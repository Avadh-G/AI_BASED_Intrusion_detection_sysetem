"""evalx.py - measured evaluation: comparison, per-class, confusion matrices, curves, ablation, stress tests, leave-one-class-out, latency."""
from __future__ import annotations
import time
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.ensemble import RandomForestClassifier
import itertools
from core import FeatureSchema, Config
from models import BENIGN, UNKNOWN, ThresholdCalibrator

def binary_metrics(ya, flag, score=None) -> dict:
    ya, flag = np.asarray(ya).astype(int), np.asarray(flag).astype(int); tp = int(((ya == 1) & (flag == 1)).sum()); fn = int(((ya == 1) & (flag == 0)).sum())
    fp = int(((ya == 0) & (flag == 1)).sum()); tn = int(((ya == 0) & (flag == 0)).sum())
    out = dict(accuracy=(tp + tn) / max(len(ya), 1), precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), FPR=fp / max(fp + tn, 1), FNR=fn / max(fn + tp, 1), TP=tp, FP=fp, FN=fn, TN=tn)
    out["F1"] = 2 * out["precision"] * out["recall"] / max(out["precision"] + out["recall"], 1e-12)
    if score is not None and 0 < ya.sum() < len(ya): out["PR_AUC"] = float(average_precision_score(ya, score)); out["ROC_AUC"] = float(roc_auc_score(ya, score))
    return out

def _curve(ya, score, n=150):
    p, r, _ = precision_recall_curve(ya, score); f, t, _ = roc_curve(ya, score); ip = np.linspace(0, len(p) - 1, min(n, len(p))).astype(int); iff = np.linspace(0, len(f) - 1, min(n, len(f))).astype(int)
    return dict(precision=p[ip].tolist(), recall=r[ip].tolist(), fpr=f[iff].tolist(), tpr=t[iff].tolist())

def evaluate_all(h, test: pd.DataFrame, val2: pd.DataFrame) -> Dict:
    classes = h.classes; enc = {c: i for i, c in enumerate(classes)}; y = test["Label"].map(enc).values; ya = (y != h.bi).astype(int)
    X = h.pre.transform(test[FeatureSchema.names]); c = h.predict_arrays(X)
    Xv = h.pre.transform(val2[FeatureSchema.names]); yv = val2["Label"].map(enc).values; cv = h.components(Xv); fpr = h.cfg.target_fpr_review
    thr_if = ThresholdCalibrator.benign_quantile(cv["I_raw"][yv == h.bi], fpr); thr_ae = ThresholdCalibrator.benign_quantile(cv["A_raw"][yv == h.bi], fpr)   # thresholds from VALIDATION
    sys_ = {}
    def sup(name, P, score):
        pred = P.argmax(1); d = binary_metrics(ya, pred != h.bi, score); d.update(macro_F1=f1_score(y, pred, average="macro"), weighted_F1=f1_score(y, pred, average="weighted"), multiclass_accuracy=accuracy_score(y, pred)); sys_[name] = (d, pred)
    sup("Random Forest", c["prf"], 1 - c["prf"][:, h.bi]); sup("XGBoost", c["pxg"], 1 - c["pxg"][:, h.bi]); sup("Supervised ensemble", c["P"], c["S"])
    sys_["Isolation Forest"] = (binary_metrics(ya, c["I_raw"] > thr_if, c["I_raw"]), None); sys_["Autoencoder"] = (binary_metrics(ya, c["A_raw"] > thr_ae, c["A_raw"]), None)
    names = classes + [UNKNOWN]; hp = np.array([names.index(k) if k != BENIGN else h.bi for k in c["category"]])
    d = binary_metrics(ya, c["status"] != "NORMAL", c["R"]); d.update(macro_F1=f1_score(y, hp, average="macro", labels=list(range(len(classes)))), weighted_F1=f1_score(y, hp, average="weighted", labels=list(range(len(classes)))),
        multiclass_accuracy=accuracy_score(y, hp), HIGH_only_recall=float(((c["status"] == "HIGH") & (ya == 1)).sum() / max(ya.sum(), 1)), HIGH_only_FPR=float(((c["status"] == "HIGH") & (ya == 0)).sum() / max((ya == 0).sum(), 1)),
        unknown_flagged=int((c["category"] == UNKNOWN).sum())); sys_["Hybrid IDS (REVIEW+HIGH = alert)"] = (d, hp)
    comp = pd.DataFrame({k: v[0] for k, v in sys_.items()}).T
    per_class = {}
    for k, (d_, p_) in sys_.items():
        if p_ is None: continue
        per_class[k] = {cl: float(((p_ == i) & (y == i)).sum() / max((y == i).sum(), 1)) for i, cl in enumerate(classes)}
    per_class["Hybrid (any alert)"] = {cl: float((((c["status"] != "NORMAL") & (y == i)).sum()) / max((y == i).sum(), 1)) for i, cl in enumerate(classes) if i != h.bi}
    cms = {k: dict(labels=(names if k.startswith("Hybrid") else classes), matrix=confusion_matrix(y, p_, labels=list(range(len(names if k.startswith("Hybrid") else classes)))).tolist()) for k, (d_, p_) in sys_.items() if p_ is not None}
    curves = {"Random Forest": _curve(ya, 1 - c["prf"][:, h.bi]), "XGBoost": _curve(ya, 1 - c["pxg"][:, h.bi]), "Isolation Forest": _curve(ya, c["I_raw"]), "Autoencoder": _curve(ya, c["A_raw"]), "Hybrid risk R": _curve(ya, c["R"])}
    return dict(comparison=comp.reset_index().rename(columns={"index": "system"}).to_dict("records"), per_class_recall=per_class, confusion=cms, curves=curves,
                thresholds=dict(if_raw=thr_if, ae_raw=thr_ae, target_fpr=fpr), test_rows=int(len(test)), report_hybrid=classification_report(y, hp, labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0))

def ablation(h, val2: pd.DataFrame, test: pd.DataFrame) -> List[dict]:
    enc = {c: i for i, c in enumerate(h.classes)}; fpr = h.cfg.target_fpr_review; out = []
    cv, ct = h.components(h.pre.transform(val2[FeatureSchema.names])), h.components(h.pre.transform(test[FeatureSchema.names]))
    yv, yt = (val2["Label"].map(enc).values != h.bi).astype(int), (test["Label"].map(enc).values != h.bi).astype(int)
    for subset in [("S",), ("I",), ("A",), ("S", "I"), ("S", "A"), ("I", "A"), ("S", "I", "A")]:
        best = (-1, None)
        for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=len(subset)):
            if abs(sum(w) - 1) > 1e-6: continue
            r = sum(wi * cv[k] for wi, k in zip(w, subset)); t, thr = ThresholdCalibrator.tpr_at_fpr(r, yv, fpr)
            if t > best[0] + 1e-9: best = (t, (w, thr))
        w, thr = best[1]; rt = sum(wi * ct[k] for wi, k in zip(w, subset)); m = binary_metrics(yt, rt > thr, rt)
        out.append(dict(components="+".join(subset), weights=[round(float(x), 2) for x in w], val_TPR=best[0], test_recall=m["recall"], test_FPR=m["FPR"], PR_AUC=m.get("PR_AUC"), ROC_AUC=m.get("ROC_AUC")))
    return out

def stress_feature_mixing(h, test: pd.DataFrame, lambdas=(0.0, 0.25, 0.5, 0.75), seed=0) -> List[dict]:
    """Attack flows are blended toward random benign flows in log-feature space (attack resembles legitimate traffic). Perturbation test, not real data."""
    rng = np.random.default_rng(seed); enc = {c: i for i, c in enumerate(h.classes)}; y = test["Label"].map(enc).values
    Xa = np.log1p(test.loc[y != h.bi, FeatureSchema.names].values.astype(float)); Xb = np.log1p(test.loc[y == h.bi, FeatureSchema.names].values.astype(float)); res = []
    for lam in lambdas:
        mix = (1 - lam) * Xa + lam * Xb[rng.integers(0, len(Xb), len(Xa))]; Z = h.pre.transform(pd.DataFrame(np.expm1(mix), columns=FeatureSchema.names)); c = h.predict_arrays(Z)
        res.append(dict(mix_lambda=lam, hybrid_recall=float((c["status"] != "NORMAL").mean()), supervised_only_recall=float((c["S"] >= h.s_thr).mean()), anomaly_only_recall=float((c["ifs"] | c["aes"]).mean()),
                        hybrid_FNR=float((c["status"] == "NORMAL").mean())))
    return res

def leave_one_class_out(h, train: pd.DataFrame, test: pd.DataFrame, cfg: Config, cap=20_000) -> List[dict]:
    """Retrain a small RF WITHOUT class k, then ask: does the supervised path miss k, and does the IF/AE path (never trained on attacks) catch it?"""
    enc = {c: i for i, c in enumerate(h.classes)}; res = []
    for k in [c for c in h.classes if c != BENIGN]:
        tr = pd.concat([g.sample(min(len(g), cap), random_state=cfg.seed) for l, g in train.groupby("Label") if l != k]); y = tr["Label"].map(enc).values
        rf = RandomForestClassifier(n_estimators=60, max_depth=20, n_jobs=-1, random_state=cfg.seed, class_weight="balanced_subsample").fit(h.pre.transform(tr[FeatureSchema.names]), y)
        te = test[test["Label"] == k]; Xk = h.pre.transform(te[FeatureSchema.names]); c = h.predict_arrays(Xk); S_rf = 1 - rf.predict_proba(Xk)[:, h.bi]
        sup = S_rf >= 0.5; anom = c["ifs"] & c["aes"]; anyan = c["ifs"] | c["aes"]
        res.append(dict(held_out_class=k, n=len(te), supervised_detects=float(sup.mean()), both_anomaly_signals=float(anom.mean()), either_anomaly_signal=float(anyan.mean()), caught_by_hybrid_logic=float((sup | anom).mean())))
    return res

def latency_throughput(h, test: pd.DataFrame, n=20000) -> dict:
    X = h.pre.transform(test[FeatureSchema.names].sample(min(n, len(test)), random_state=0)); t = time.time(); h.predict_arrays(X); thr = len(X) / (time.time() - t)
    one = X[:1]; ts = []
    for i in range(30): t = time.time(); h.predict_arrays(X[i:i + 1]); ts.append(time.time() - t)
    return dict(throughput_flows_per_s=float(thr), single_flow_latency_ms=float(np.median(ts) * 1000))

def ks_domain_gap(cic: pd.DataFrame, sim: pd.DataFrame, label_map: Dict[str, str]) -> pd.DataFrame:
    """Per-class Kolmogorov-Smirnov statistic between simulator flows and CIC test flows (log1p). Large = domain gap."""
    from scipy.stats import ks_2samp
    rows = []
    for sl, cl in label_map.items():
        a, b = sim[sim["true_label"] == sl], cic[cic["Label"] == cl]
        if len(a) < 5 or len(b) < 5: continue
        ks = {f: ks_2samp(np.log1p(a[f].astype(float)), np.log1p(b[f].astype(float))).statistic for f in FeatureSchema.names}; rows.append(dict(sim_class=sl, cic_class=cl, n_sim=len(a), n_cic=len(b), mean_KS=float(np.mean(list(ks.values()))), worst_features=sorted(ks, key=ks.get)[-3:]))
    return pd.DataFrame(rows)
