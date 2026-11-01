"""data.py - CIC-IDS2017 loading, inspection, cleaning and leakage-safe chronological splitting."""
from __future__ import annotations
import glob, os
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from core import Config, FeatureSchema, get_logger

log = get_logger("ids.data")

def normalize_label(s: str) -> str:
    """Map raw CIC labels to the merged taxonomy used by this project (mapping is documented in the report)."""
    t = str(s).strip().lower()
    if "web attack" in t: return "WebAttack"
    if t.startswith("dos"): return "DoS"
    if t == "ddos": return "DDoS"
    if t == "portscan": return "PortScan"
    if "ftp" in t: return "FTP-Patator"
    if "ssh" in t: return "SSH-Patator"
    if t == "bot": return "Bot"
    if t == "benign": return "BENIGN"
    if "infiltration" in t: return "Infiltration"
    if "heartbleed" in t: return "Heartbleed"
    return str(s).strip()

class DatasetProcessor:
    """Loads TrafficLabelling CSVs, cleans them with the SAME rules as the simulator path, and splits chronologically."""
    def __init__(self, cfg: Config): self.cfg = cfg; self.report: Dict[str, object] = {}
    def find_files(self) -> list:
        # Recursively find all CSV files
        fs = sorted(glob.glob(os.path.join(self.cfg.data_dir, "**", "*.csv"), recursive=True))
        # Loosen matching: accept files containing working hours, or ending with ISCX.csv, or typical MachineLearningCSV names
        fs = [f for f in fs if ("WorkingHours" in os.path.basename(f) or f.endswith("pcap_ISCX.csv") or "csv" in os.path.basename(f).lower()) and "Maths" not in os.path.basename(f)]
        if not fs:
            raise FileNotFoundError(f"No CIC-IDS2017 CSV files found under {self.cfg.data_dir} (recursive search). Please check the path and upload the CSVs.")
        return fs
    def inspect_file(self, path: str, nrows: int = 5000) -> pd.DataFrame:
        raw = pd.read_csv(path, nrows=nrows, encoding="latin1"); raw.columns = [c.strip() for c in raw.columns]
        return raw
    def load(self) -> pd.DataFrame:
        """Chunked, downcast load. Keeps schema features + Timestamp + Label. Caps rows per class (time-ordered systematic sampling)."""
        want = set(FeatureSchema.names) | {"Label", "Timestamp"}; parts, raw_n, order = [], 0, 0
        for f in self.find_files():
            log.info("loading %s", os.path.basename(f))
            for ch in pd.read_csv(f, usecols=lambda c: c.strip() in want, chunksize=250_000, encoding="latin1", low_memory=False):
                ch.columns = [c.strip() for c in ch.columns]; ch = ch.loc[:, ~ch.columns.duplicated()]
                raw_n += len(ch); ch["Label"] = ch["Label"].map(normalize_label)
                for c in FeatureSchema.names: ch[c] = pd.to_numeric(ch[c], errors="coerce").astype("float64")
                ch["Timestamp"] = self._parse_ts(ch["Timestamp"])
                ch["_order"] = np.arange(order, order + len(ch)); order += len(ch); parts.append(ch)
        df = pd.concat(parts, ignore_index=True); del parts
        missing = [c for c in FeatureSchema.names if c not in df.columns]
        if missing: raise ValueError(f"Dataset lacks schema features: {missing}")
        self.report["raw_rows"] = raw_n; self.report["raw_label_counts"] = df["Label"].value_counts().to_dict()
        return self.clean(df)
    @staticmethod
    def _parse_ts(s: pd.Series) -> pd.Series:
        """Parse CIC timestamps (month-first). Files record 12-hour clock times without AM/PM; captures ran ~08:00-17:00, so hours 1-7 are PM (+12h). Heuristic, documented."""
        t = pd.to_datetime(s, errors="coerce")
        bad = t.isna() & s.notna()
        if bad.any(): t[bad] = pd.to_datetime(s[bad], errors="coerce", format="mixed")
        pm = t.notna() & (t.dt.hour < 8); t[pm] = t[pm] + pd.Timedelta(hours=12); return t
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        r = self.report; n0 = len(df)
        x = FeatureSchema.apply_zero_duration_rule(df[FeatureSchema.names]); df[FeatureSchema.names] = x
        arr = df[FeatureSchema.names].replace([np.inf, -np.inf], np.nan)
        r["nan_or_inf_cells"] = int(arr.isna().sum().sum()); r["negative_cells"] = int((arr < 0).sum().sum())
        r["constant_features"] = [c for c in FeatureSchema.names if df[c].nunique(dropna=True) <= 1]
        ok = FeatureSchema.valid_mask(arr); r["rows_dropped_invalid"] = int((~ok).sum()); df = df[ok.values].copy()
        n1 = len(df); df = df.drop_duplicates(subset=FeatureSchema.names + ["Label"]); r["duplicates_removed"] = n1 - len(df)
        if df["Timestamp"].isna().mean() > 0.2: log.warning("Timestamps unparsable -> falling back to file order"); df["Timestamp"] = pd.to_datetime(df["_order"], unit="s")
        df["Timestamp"] = df["Timestamp"].fillna(pd.Timestamp("1970-01-01"))
        df[FeatureSchema.names] = df[FeatureSchema.names].astype("float32")
        cnt = df["Label"].value_counts(); keep = cnt[cnt >= self.cfg.min_class_rows].index
        r["excluded_rare_classes"] = {k: int(v) for k, v in cnt.items() if k not in keep}
        self.rare = df[~df["Label"].isin(keep)].copy(); df = df[df["Label"].isin(keep)]
        df = self._cap(df); r["clean_rows"] = len(df); r["clean_label_counts"] = df["Label"].value_counts().to_dict()
        log.info("clean rows %d (from %d)", len(df), n0); return df.reset_index(drop=True)
    def _cap(self, df):
        out = []
        for lab, g in df.groupby("Label"):
            g = g.sort_values(["Timestamp", "_order"])
            if len(g) > self.cfg.max_rows_per_class: g = g.iloc[np.linspace(0, len(g) - 1, self.cfg.max_rows_per_class).astype(int)]
            out.append(g)
        return pd.concat(out)
    def split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Per-class chronological blocks train/val/test with a gap buffer between blocks (no random row-level split)."""
        a, b, _ = self.cfg.split; gap = self.cfg.split_gap; parts = {"train": [], "val": [], "test": []}
        for lab, g in df.groupby("Label"):
            g = g.sort_values(["Timestamp", "_order"]); n = len(g); i, j, gp = int(n * a), int(n * (a + b)), max(1, int(n * gap))
            parts["train"].append(g.iloc[: i - gp]); parts["val"].append(g.iloc[i: j - gp]); parts["test"].append(g.iloc[j:])
        out = {k: pd.concat(v).sort_values(["Timestamp", "_order"]).reset_index(drop=True) for k, v in parts.items()}
        self.leakage_check(out); return out
    @staticmethod
    def leakage_check(sp: Dict[str, pd.DataFrame]):
        # row-id disjointness and strict time ordering per class
        ids = [set(sp[k]["_order"]) for k in ("train", "val", "test")]
        assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2]), "row overlap between splits"
        for lab in sp["train"]["Label"].unique():
            t = {k: sp[k].loc[sp[k]["Label"] == lab, "Timestamp"] for k in sp}
            assert t["train"].max() <= t["val"].min() and t["val"].max() <= t["test"].min(), f"temporal leakage in class {lab}"
        log.info("leakage checks passed (disjoint rows, per-class chronological order)")
