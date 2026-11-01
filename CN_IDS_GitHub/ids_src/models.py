"""models.py - RF, XGBoost, Isolation Forest, Autoencoder, calibration and the unified HybridIntrusionDetector."""
from __future__ import annotations
import itertools, time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from core import Config, FeatureSchema, Preprocessor, get_logger

log = get_logger("ids.models")
BENIGN = "BENIGN"; UNKNOWN = "UNKNOWN_ANOMALY"

# ------------------------------------------------------------------ base detectors
class RandomForestDetector:
    def __init__(self, seed=42, n=80, **kw):
        self.m = RandomForestClassifier(n_estimators=n, max_depth=20, min_samples_leaf=5, n_jobs=-1, random_state=seed, class_weight="balanced_subsample", **kw)
    def fit(self, X, y): self.m.fit(X, y); return self
    def predict_proba(self, X): return self.m.predict_proba(X)
    @property
    def importances(self): return self.m.feature_importances_

class XGBoostDetector:
    def __init__(self, n_classes: int, seed=42):
        import xgboost as xgb
        self.m = xgb.XGBClassifier(objective="multi:softprob", n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.8,
                                   colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=seed, early_stopping_rounds=15, eval_metric="mlogloss")
    def fit(self, X, y, Xv, yv):
        self.m.fit(X, y, sample_weight=compute_sample_weight("balanced", y), eval_set=[(Xv, yv)], verbose=False); return self
    def predict_proba(self, X): return self.m.predict_proba(X)

class IsolationForestDetector:
    """Unsupervised; trained on benign TRAIN flows only. score = -score_samples (higher = more anomalous)."""
    def __init__(self, seed=42): self.m = IsolationForest(n_estimators=150, max_samples=512, contamination="auto", n_jobs=-1, random_state=seed)
    def fit(self, Xb): self.m.fit(Xb); return self
    def score(self, X): return -self.m.score_samples(X)

class AutoencoderDetector:
    """Dense autoencoder (d->32->16->8->16->32->d) trained on benign flows; reconstruction MSE = anomaly score."""
    def __init__(self, d: int, seed=42): self.d, self.seed, self.m = d, seed, None
    def _build(self):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(self.seed); L = tf.keras.layers
        m = tf.keras.Sequential([L.Input((self.d,)), L.Dense(32, activation="relu"), L.Dense(16, activation="relu"), L.Dense(8, activation="relu"),
                                 L.Dense(16, activation="relu"), L.Dense(32, activation="relu"), L.Dense(self.d)])
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse"); return m
    def fit(self, Xb, Xbv, epochs=40):
        import tensorflow as tf
        self.m = self._build()
        self.history = self.m.fit(Xb, Xb, validation_data=(Xbv, Xbv), epochs=epochs, batch_size=512, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_loss")]).history; return self
    def score(self, X): R = self.m.predict(X, batch_size=4096, verbose=0); return ((X - R) ** 2).mean(1)

# ------------------------------------------------------------------ calibration helpers
class BenignCDF:
    """Maps a raw anomaly score to its quantile among BENIGN calibration scores (0..1), putting IF and AE on one scale."""
    def fit(self, benign_scores): self.s = np.sort(np.asarray(benign_scores)); return self
    def __call__(self, x): return np.searchsorted(self.s, np.asarray(x), side="right") / len(self.s)

class ThresholdCalibrator:
    """All thresholds come from VALIDATION data only."""
    @staticmethod
    def benign_quantile(scores_benign: np.ndarray, target_fpr: float) -> float: return float(np.quantile(scores_benign, 1 - target_fpr))
    @staticmethod
    def tpr_at_fpr(score: np.ndarray, y_att: np.ndarray, fpr: float) -> Tuple[float, float]:
        thr = ThresholdCalibrator.benign_quantile(score[y_att == 0], fpr); return float((score[y_att == 1] > thr).mean()), thr
    @staticmethod
    def tune_class_bias(P: np.ndarray, y: np.ndarray, grid=(0.5, 0.75, 1, 1.5, 2, 3, 5)) -> np.ndarray:
        """Per-class probability multipliers (coordinate ascent on validation macro-F1)."""
        b = np.ones(P.shape[1]); best = f1_score(y, (P * b).argmax(1), average="macro")
        for _ in range(2):
            for k in range(P.shape[1]):
                for g in grid:
                    t = b.copy(); t[k] = g; f = f1_score(y, (P * t).argmax(1), average="macro")
                    if f > best + 1e-6: best, b = f, t
        return b

# ------------------------------------------------------------------ unified hybrid
class HybridIntrusionDetector:
    """ONE inference object: shared preprocessing -> RF+XGB ensemble -> calibrated S; IF + AE -> benign-CDF I, A -> risk R -> 3-level decision.
    R is chosen on validation among: weighted sum (alpha*S+beta*I+gamma*A), logistic blend, and max-rule."""
    def __init__(self, cfg: Config): self.cfg = cfg; self.classes: List[str] = []; self.cfg_dict: Dict = {}
    # ---- raw component outputs
    def components(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        prf, pxg = self.rf.predict_proba(X), self.xgb.predict_proba(X); ir, ar = self.ifo.score(X), self.ae.score(X)
        P = self.w_rf * prf + (1 - self.w_rf) * pxg; P = P * self.class_bias; P = P / P.sum(1, keepdims=True)
        att = 1 - P[:, self.bi]; S = self.iso.predict(att)
        return dict(prf=prf, pxg=pxg, P=P, S=S, I=self.cdf_if(ir), A=self.cdf_ae(ar), I_raw=ir, A_raw=ar)
    def risk(self, S, I, A) -> np.ndarray:
        f = self.combiner
        if f["kind"] == "weighted": return f["alpha"] * S + f["beta"] * I + f["gamma"] * A
        if f["kind"] == "blend": return f["lr"].predict_proba(np.c_[S, I, A])[:, 1]
        return np.maximum(S, np.minimum(I, A))
    # ---- fitting on validation (val1: calibration, val2: combiner/thresholds)
    def fit_calibration(self, X1, y1, X2, y2):
        c = self.components_uncal(X1); self.class_bias = np.ones(len(self.classes)); self.w_rf = 0.5
        best = (-1, 0.5)
        for w in np.linspace(0, 1, 11):
            f = f1_score(y1, (w * c["prf"] + (1 - w) * c["pxg"]).argmax(1), average="macro"); best = max(best, (f, w))
        self.w_rf = float(best[1]); P = self.w_rf * c["prf"] + (1 - self.w_rf) * c["pxg"]; self.class_bias = ThresholdCalibrator.tune_class_bias(P, y1)
        ya = (y1 != self.bi).astype(int); Pn = P * self.class_bias; Pn /= Pn.sum(1, keepdims=True)
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(1 - Pn[:, self.bi], ya)
        self.cdf_if = BenignCDF().fit(self.ifo.score(X1[y1 == self.bi])); self.cdf_ae = BenignCDF().fit(self.ae.score(X1[y1 == self.bi]))
        c2 = self.components(X2); ya2 = (y2 != self.bi).astype(int); S, I, A = c2["S"], c2["I"], c2["A"]; fpr = self.cfg.target_fpr_review
        cands = {}
        best_w = (-1, None)
        for a, b in itertools.product(np.arange(0, 1.01, 0.1), repeat=2):
            if a + b > 1.0001: continue
            t, _ = ThresholdCalibrator.tpr_at_fpr(a * S + b * I + (1 - a - b) * A, ya2, fpr)
            if t > best_w[0] + 1e-9: best_w = (t, dict(kind="weighted", alpha=float(a), beta=float(b), gamma=float(1 - a - b)))
        cands["weighted"] = best_w
        Z = np.c_[S, I, A]; lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
        cv = cross_val_predict(lr, Z, ya2, cv=StratifiedKFold(5, shuffle=True, random_state=self.cfg.seed), method="predict_proba")[:, 1]   # out-of-fold
        cands["blend"] = (ThresholdCalibrator.tpr_at_fpr(cv, ya2, fpr)[0], dict(kind="blend", lr=lr.fit(Z, ya2)))
        cands["maxrule"] = (ThresholdCalibrator.tpr_at_fpr(np.maximum(S, np.minimum(I, A)), ya2, fpr)[0], dict(kind="maxrule"))
        self.combiner_scores = {k: float(v[0]) for k, v in cands.items()}; k = max(cands, key=lambda z: cands[z][0]); self.combiner = cands[k][1]
        R = self.risk(S, I, A); rb = R[ya2 == 0]
        self.tau_review = ThresholdCalibrator.benign_quantile(rb, self.cfg.target_fpr_review); self.tau_high = ThresholdCalibrator.benign_quantile(rb, self.cfg.target_fpr_high)
        self.s_thr = max(0.5, ThresholdCalibrator.benign_quantile(S[ya2 == 0], 0.01)); self.sig_q = self.cfg.signal_quantile
        self.cfg_dict = dict(selected=k, combiner_val_tpr=self.combiner_scores, w_rf=self.w_rf, class_bias=self.class_bias.tolist(), tau_review=self.tau_review,
                             tau_high=self.tau_high, s_thr=self.s_thr, signal_quantile=self.sig_q, combiner={a: b for a, b in self.combiner.items() if a != "lr"})
        log.info("hybrid calibrated: %s", self.cfg_dict); return self
    def components_uncal(self, X):
        return dict(prf=self.rf.predict_proba(X), pxg=self.xgb.predict_proba(X))
    # ---- inference ---------------------------------------------------------------------
    def predict_arrays(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        c = self.components(X); R = self.risk(c["S"], c["I"], c["A"])
        sup, ifs, aes = c["S"] >= self.s_thr, c["I"] >= self.sig_q, c["A"] >= self.sig_q; n = sup.astype(int) + ifs + aes
        high = (R >= self.tau_high) & (n >= 2) & sup
        review = (R >= self.tau_review) | (~sup & ifs & aes)
        status = np.where(high, "HIGH", np.where(review, "REVIEW", "NORMAL"))
        att = np.array([i for i in range(len(self.classes)) if i != self.bi]); top = att[c["P"][:, att].argmax(1)]
        cat = np.where(status == "NORMAL", BENIGN, np.where(sup, np.array(self.classes)[top], UNKNOWN))
        c.update(R=R, status=status, category=cat, sup=sup, ifs=ifs, aes=aes, n_signals=n); return c
    def predict(self, flows: pd.DataFrame) -> pd.DataFrame:
        """Validate schema -> shared preprocessing -> unified result per flow (adds columns to a copy of the metadata)."""
        Xdf = FeatureSchema.clean(flows)
        if not FeatureSchema.valid_mask(Xdf).all(): raise ValueError("Invalid (NaN/Inf/negative) feature values in flows")
        if self.pre.schema_hash != FeatureSchema.version_hash(): raise ValueError("Schema hash mismatch between model and runtime")
        c = self.predict_arrays(self.pre.transform(Xdf)); out = flows.drop(columns=FeatureSchema.names, errors="ignore").copy()
        att = [i for i in range(len(self.classes)) if i != self.bi]
        out["status"], out["category"], out["risk"] = c["status"], c["category"], c["R"]
        out["S_sup"], out["I_iforest"], out["A_ae"] = c["S"], c["I"], c["A"]; out["ae_recon_error"], out["if_raw_score"] = c["A_raw"], c["I_raw"]
        out["rf_attack_p"], out["xgb_attack_p"] = 1 - c["prf"][:, self.bi], 1 - c["pxg"][:, self.bi]
        out["rf_class"] = np.array(self.classes)[c["prf"].argmax(1)]; out["xgb_class"] = np.array(self.classes)[c["pxg"].argmax(1)]
        out["signals"] = c["n_signals"]
        out["agreement"] = [f"RF={a}, XGB={b}, IF={'anomalous' if i else 'normal'}, AE={'elevated' if e else 'normal'}" + (" [UNANIMOUS]" if (a != BENIGN) == (b != BENIGN) == bool(i) == bool(e) else " [DISAGREE]")
                            for a, b, i, e in zip(out["rf_class"], out["xgb_class"], c["ifs"], c["aes"])]
        out["action"] = [self.recommend(s, k) for s, k in zip(out["status"], out["category"])]
        for j, k in enumerate(self.classes): out[f"P_{k}"] = c["P"][:, j]
        return out
    @staticmethod
    def recommend(status: str, cat: str) -> str:
        if status == "NORMAL": return "Allow; continue monitoring"
        if status == "REVIEW": return "Warn + raise monitoring + preserve evidence; administrator review (no automatic blocking)"
        return {"PortScan": "Block source IP (ACL) after review; restrict scanned ports", "DoS": "Rate-limit / block source IP; verify service health", "DDoS": "Block participating sources; upstream filtering",
                "FTP-Patator": "Block source on port 21; enforce lockout", "SSH-Patator": "Block source on port 22; enforce key-only auth", "Bot": "Isolate host; inspect C2 destination",
                "WebAttack": "Block source / WAF rule; inspect requests"}.get(cat, "Block or isolate after administrator confirmation")
    # ---- explainability ----------------------------------------------------------------
    def fit_explainer(self, Xtrain_benign: np.ndarray):
        self.base_mu = Xtrain_benign.mean(0); self.base_sd = Xtrain_benign.std(0) + 1e-6; self.imp = self.rf.importances; return self
    def explain(self, flow_row: pd.Series, k: int = 5) -> List[str]:
        """Importance-weighted deviation from the benign baseline (always available). SHAP is used separately when installed."""
        x = self.pre.transform(FeatureSchema.clean(pd.DataFrame([flow_row[FeatureSchema.names].astype(float)])))[0]
        z = (x - self.base_mu) / self.base_sd; sc = np.abs(z) * (self.imp / self.imp.max()); idx = np.argsort(-sc)[:k]
        return [f"{'High' if z[i] > 0 else 'Low'} {FeatureSchema.names[i]} (benign z={z[i]:+.1f}, RF importance={self.imp[i]:.3f})" for i in idx]
    def shap_top(self, X: np.ndarray, cls: str, k: int = 5):
        try:
            import shap; ex = shap.TreeExplainer(self.xgb.m); sv = ex.shap_values(X); sv = np.asarray(sv)
            j = self.classes.index(cls); v = sv[:, :, j] if sv.ndim == 3 and sv.shape[2] == len(self.classes) else sv[j]
            return [[(FeatureSchema.names[i], float(r[i])) for i in np.argsort(-np.abs(r))[:k]] for r in v]
        except Exception as e: log.warning("SHAP unavailable: %s", e); return None
    # ---- persistence ---------------------------------------------------------------------
    def save(self, path: str):
        xm, aem = self.xgb.m, self.ae.m; v = (getattr(self, "val1", None), getattr(self, "val2", None)); self.xgb.m = None; self.ae.m = None; self.val1 = self.val2 = None
        try: joblib.dump(self, path + "/hybrid.joblib", compress=3)
        finally: self.xgb.m, self.ae.m = xm, aem; self.val1, self.val2 = v
        xm.save_model(path + "/xgb.json"); aem.save(path + "/ae.keras")
    @staticmethod
    def load(path: str) -> "HybridIntrusionDetector":
        import xgboost as xgb, tensorflow as tf
        h = joblib.load(path + "/hybrid.joblib"); h.xgb.m = xgb.XGBClassifier(); h.xgb.m.load_model(path + "/xgb.json"); h.ae.m = tf.keras.models.load_model(path + "/ae.keras")
        if h.pre.schema_hash != FeatureSchema.version_hash(): raise ValueError("Saved model was trained with a different FeatureSchema")
        return h

class ModelTrainer:
    """Trains RF, XGB, IF, AE on TRAIN only (preprocessing fitted on TRAIN only); calibrates on VALIDATION only."""
    def __init__(self, cfg: Config): self.cfg = cfg; self.timings: Dict[str, float] = {}
    def _cap(self, df: pd.DataFrame, n: int, rng) -> pd.DataFrame:
        return pd.concat([g.sample(min(len(g), n), random_state=self.cfg.seed) for _, g in df.groupby("Label")])
    def compare_imbalance(self, tr: pd.DataFrame, va: pd.DataFrame, pre: Preprocessor, classes: List[str], n_small=80_000) -> pd.DataFrame:
        """Validation macro-F1 of 3 strategies with a small RF: class weights / capped undersampling / random oversampling."""
        rng = np.random.default_rng(self.cfg.seed); y = lambda d: d["Label"].map({c: i for i, c in enumerate(classes)}).values
        Xv, yv = pre.transform(va[FeatureSchema.names]), y(va); res = {}
        base = self._cap(tr, n_small, rng); med = int(base["Label"].value_counts().median())
        for name, d, cw in [("none", base, None), ("class_weight", base, "balanced_subsample"),
                            ("undersample(cap 5k)+none", self._cap(tr, 5_000, rng), None),
                            ("oversample(to median)", pd.concat([g.sample(med, replace=len(g) < med, random_state=1) for _, g in base.groupby("Label")]), None)]:
            m = RandomForestClassifier(n_estimators=60, max_depth=20, n_jobs=-1, random_state=self.cfg.seed, class_weight=cw).fit(pre.transform(d[FeatureSchema.names]), y(d))
            res[name] = dict(macro_f1=f1_score(yv, m.predict(Xv), average="macro"))
        return pd.DataFrame(res).T
    def train(self, sp: Dict[str, pd.DataFrame]) -> HybridIntrusionDetector:
        cfg = self.cfg; rng = np.random.default_rng(cfg.seed); tr, va = sp["train"], sp["val"]
        classes = sorted(tr["Label"].unique(), key=lambda c: (c != BENIGN, c)); h = HybridIntrusionDetector(cfg); h.classes, h.bi = classes, classes.index(BENIGN)
        enc = {c: i for i, c in enumerate(classes)}; h.pre = Preprocessor().fit(tr[FeatureSchema.names])
        trc = self._cap(tr, cfg.train_cap_per_class, rng); Xt, yt = h.pre.transform(trc[FeatureSchema.names]), trc["Label"].map(enc).values
        # validation halves per class, chronological: val1 = calibration, val2 = combiner/thresholds
        v1, v2 = [], []
        for _, g in va.groupby("Label"): g = g.sort_values("Timestamp"); k = len(g) // 2; v1.append(g.iloc[:k]); v2.append(g.iloc[k:])
        v1, v2 = pd.concat(v1), pd.concat(v2); X1, y1 = h.pre.transform(v1[FeatureSchema.names]), v1["Label"].map(enc).values; X2, y2 = h.pre.transform(v2[FeatureSchema.names]), v2["Label"].map(enc).values
        t = time.time(); h.rf = RandomForestDetector(cfg.seed).fit(Xt, yt); self.timings["rf"] = time.time() - t
        t = time.time(); h.xgb = XGBoostDetector(len(classes), cfg.seed).fit(Xt, yt, X1, y1); self.timings["xgb"] = time.time() - t
        Xb = h.pre.transform(tr.loc[tr["Label"] == BENIGN, FeatureSchema.names].sample(min(300_000, (tr["Label"] == BENIGN).sum()), random_state=cfg.seed))
        t = time.time(); h.ifo = IsolationForestDetector(cfg.seed).fit(Xb[:200_000]); self.timings["iforest"] = time.time() - t
        t = time.time(); h.ae = AutoencoderDetector(Xb.shape[1], cfg.seed).fit(Xb, X1[y1 == h.bi]); self.timings["autoencoder"] = time.time() - t
        h.fit_calibration(X1, y1, X2, y2); h.fit_explainer(Xb); h.val1, h.val2 = v1, v2; return h
