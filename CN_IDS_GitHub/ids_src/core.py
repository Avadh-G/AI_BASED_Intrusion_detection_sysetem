"""core.py - configuration, logging and the single FeatureSchema used everywhere."""
from __future__ import annotations
import hashlib, json, logging, os
from dataclasses import dataclass, field, asdict
from typing import Dict, List
import numpy as np
import pandas as pd

def get_logger(name: str = "ids") -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        lg.addHandler(h); lg.setLevel(logging.INFO)
    return lg

@dataclass
class Config:
    """Central configuration (all tunables live here)."""
    seed: int = 42
    work_dir: str = "/content/ids_artifacts"
    data_dir: str = "/content/cicids2017"
    max_rows_per_class: int = 200_000      # Colab Free RAM guard (time-ordered systematic sampling)
    train_cap_per_class: int = 40_000      # capped undersampling for supervised training (RAM/time)
    min_class_rows: int = 200              # classes below this are excluded from supervised training
    split: tuple = (0.6, 0.2, 0.2)         # chronological blocks per class
    split_gap: float = 0.01                # fraction of rows dropped between blocks (leakage buffer)
    target_fpr_review: float = 0.02
    target_fpr_high: float = 0.002
    signal_quantile: float = 0.99          # anomaly signal = above 99th pct of benign validation scores
    flow_timeout_s: float = 120.0
    activity_timeout_us: float = 5_000_000.0
    temporal_windows_s: tuple = (10.0, 60.0, 300.0)
    temporal_consecutive: int = 3
    auto_response: bool = False
    rule_ttl_s: float = 600.0

# --------------------------------------------------------------------------- schema
# (name, unit, kind) ; kind in {count,bytes,time_us,rate,ratio,flag}. Names are the CIC-IDS2017 names.
_FEATURES = [
 ("Flow Duration","us","time_us"),("Total Fwd Packets","pkts","count"),("Total Backward Packets","pkts","count"),
 ("Total Length of Fwd Packets","bytes","bytes"),("Total Length of Bwd Packets","bytes","bytes"),
 ("Fwd Packet Length Max","bytes","bytes"),("Fwd Packet Length Min","bytes","bytes"),
 ("Fwd Packet Length Mean","bytes","bytes"),("Fwd Packet Length Std","bytes","bytes"),
 ("Bwd Packet Length Max","bytes","bytes"),("Bwd Packet Length Min","bytes","bytes"),
 ("Bwd Packet Length Mean","bytes","bytes"),("Bwd Packet Length Std","bytes","bytes"),
 ("Flow Bytes/s","bytes/s","rate"),("Flow Packets/s","pkts/s","rate"),
 ("Flow IAT Mean","us","time_us"),("Flow IAT Std","us","time_us"),("Flow IAT Max","us","time_us"),("Flow IAT Min","us","time_us"),
 ("Fwd IAT Total","us","time_us"),("Fwd IAT Mean","us","time_us"),("Fwd IAT Std","us","time_us"),("Fwd IAT Max","us","time_us"),("Fwd IAT Min","us","time_us"),
 ("Bwd IAT Total","us","time_us"),("Bwd IAT Mean","us","time_us"),("Bwd IAT Std","us","time_us"),("Bwd IAT Max","us","time_us"),("Bwd IAT Min","us","time_us"),
 ("Fwd PSH Flags","count","flag"),("Fwd Packets/s","pkts/s","rate"),("Bwd Packets/s","pkts/s","rate"),
 ("Min Packet Length","bytes","bytes"),("Max Packet Length","bytes","bytes"),("Packet Length Mean","bytes","bytes"),
 ("Packet Length Std","bytes","bytes"),("Packet Length Variance","bytes^2","bytes"),
 ("FIN Flag Count","count","flag"),("SYN Flag Count","count","flag"),("RST Flag Count","count","flag"),
 ("PSH Flag Count","count","flag"),("ACK Flag Count","count","flag"),("URG Flag Count","count","flag"),
 ("Down/Up Ratio","ratio","ratio"),("Average Packet Size","bytes","bytes"),
 ("Avg Fwd Segment Size","bytes","bytes"),("Avg Bwd Segment Size","bytes","bytes"),
 ("Active Mean","us","time_us"),("Active Std","us","time_us"),("Active Max","us","time_us"),("Active Min","us","time_us"),
 ("Idle Mean","us","time_us"),("Idle Std","us","time_us"),("Idle Max","us","time_us"),("Idle Min","us","time_us"),
]
RATE_COLS = ["Flow Bytes/s","Flow Packets/s","Fwd Packets/s","Bwd Packets/s"]

class FeatureSchema:
    """Single source of truth for model features (Tier A: exactly derivable from simulated packets).

    Excluded on purpose (documented, never zero-filled): Destination Port (shortcut feature), header-length,
    Init_Win_bytes_*, bulk/subflow features, act_data_pkt_fwd, min_seg_size_forward, ECE/CWE flags.
    Zero-duration rule (shared by dataset + extractor): rates are 0 when Flow Duration <= 0.
    Units follow CICFlowMeter: durations/IATs in microseconds, rates per second, lengths = payload bytes.
    """
    names: List[str] = [f[0] for f in _FEATURES]
    units: Dict[str, str] = {f[0]: f[1] for f in _FEATURES}
    kinds: Dict[str, str] = {f[0]: f[2] for f in _FEATURES}
    dtype = np.float64
    EXCLUDED = ["Destination Port","Fwd Header Length","Init_Win_bytes_forward","Init_Win_bytes_backward",
                "act_data_pkt_fwd","min_seg_size_forward","Bwd Header Length","ECE Flag Count","CWE Flag Count"]

    @classmethod
    def version_hash(cls) -> str:
        return hashlib.sha256(json.dumps(cls.names).encode()).hexdigest()[:12]

    @classmethod
    def formulas(cls) -> Dict[str, str]:
        return {"Flow Duration":"last_ts-first_ts (us)","Flow IAT *":"diff of sorted packet timestamps (us)",
                "Fwd/Bwd IAT *":"diff of timestamps within a direction (us)","Flow Bytes/s":"(fwd_len+bwd_len)/(duration/1e6)",
                "Flow Packets/s":"n_packets/(duration/1e6)","*Flag Count":"sum over packets of the TCP flag bit",
                "Packet Length Std":"sample std (ddof=1) over payload lengths","Down/Up Ratio":"floor(bwd_pkts/fwd_pkts)",
                "Active/Idle *":"gaps > activity timeout split active/idle periods"}

    # ---- shared preprocessing step 1 (training AND inference) --------------------------------
    @classmethod
    def validate(cls, df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
        """Return df restricted to schema columns, in schema order, float64. Raises on missing columns."""
        missing = [c for c in cls.names if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        return df[cls.names].astype(cls.dtype)

    @classmethod
    def apply_zero_duration_rule(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        z = df["Flow Duration"] <= 0
        for c in RATE_COLS:
            df.loc[z | ~np.isfinite(df[c]), c] = 0.0
        return df

    @classmethod
    def clean(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Identical cleaning for CIC rows and simulator rows: schema order, zero-duration rule, finite check."""
        x = cls.apply_zero_duration_rule(cls.validate(df))
        x = x.replace([np.inf, -np.inf], np.nan)
        return x

    @classmethod
    def valid_mask(cls, x: pd.DataFrame) -> pd.Series:
        """Rows that are finite and non-negative (negatives are CICFlowMeter artifacts)."""
        return x.notna().all(axis=1) & (x >= 0).all(axis=1)

class Preprocessor:
    """Shared preprocessing: log1p -> standardize. Fit on TRAIN only; same object at inference."""
    def __init__(self): self.mu = None; self.sd = None; self.schema_hash = FeatureSchema.version_hash()
    @staticmethod
    def _log(x: np.ndarray) -> np.ndarray: return np.log1p(np.clip(x, 0, None))
    def fit(self, X: pd.DataFrame):
        L = self._log(FeatureSchema.validate(X).values)
        self.mu = L.mean(0); self.sd = L.std(0); self.sd[self.sd < 1e-9] = 1.0
        return self
    def transform(self, X) -> np.ndarray:
        v = FeatureSchema.validate(X).values if isinstance(X, pd.DataFrame) else np.asarray(X, float)
        if not np.isfinite(v).all(): raise ValueError("Non-finite values reached preprocessing")
        return np.clip((self._log(v) - self.mu) / self.sd, -10, 10).astype(np.float32)
