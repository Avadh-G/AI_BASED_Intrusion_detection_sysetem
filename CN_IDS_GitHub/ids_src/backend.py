"""backend.py - live-style pipeline used by the Streamlit dashboard and the notebook demo: simulator -> flows -> hybrid IDS -> temporal -> alerts -> response."""
from __future__ import annotations
import json, os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from core import Config, FeatureSchema, get_logger
from sim import NetworkSimulator, ScenarioController, FlowFeatureExtractor
from models import HybridIntrusionDetector, UNKNOWN
from ids import TemporalDetectionEngine, FalsePositiveManager, AlertManager, ResponseEngine

log = get_logger("ids.backend")

class IDSBackend:
    def __init__(self, hybrid: HybridIntrusionDetector, cfg: Config, seed: Optional[int] = None):
        self.h, self.cfg = hybrid, cfg; self.seed = cfg.seed if seed is None else seed; self.build()
    def build(self):
        self.sim = NetworkSimulator(self.seed); self.ctl = ScenarioController(self.sim, self.seed); self.fx = FlowFeatureExtractor(self.cfg)
        self.temporal = TemporalDetectionEngine(self.cfg, self.h.tau_review); self.fp = FalsePositiveManager(); self.alerts = AlertManager(); self.resp = ResponseEngine(self.sim, self.cfg)
        self.flows = pd.DataFrame(); self.packets_seen = 0; self.log: List[str] = []; self.first_alert_time: Dict[str, float] = {}
    def reset(self, seed: Optional[int] = None):
        if seed is not None: self.seed = seed
        self.build()
    # ---- main loop -------------------------------------------------------------------
    def step(self, dt: float = 2.0, force_flush: bool = False) -> pd.DataFrame:
        self.ctl.tick(dt); self.resp.sweep(); return self._process(force_flush)
    def run(self, seconds: float, dt: float = 2.0, flush: bool = True) -> pd.DataFrame:
        out = []
        for _ in range(int(np.ceil(seconds / dt))): out.append(self.step(dt))
        if flush: out.append(self._process(True))
        out = [o for o in out if len(o)]; return pd.concat(out) if out else pd.DataFrame()
    def _process(self, force: bool) -> pd.DataFrame:
        pk = self.sim.harvest(self.sim.clock if not force else None); self.packets_seen += len(pk); self.fx.ingest(pk)
        rows = self.fx.pop_finished(self.sim.clock, force)
        if not rows: return pd.DataFrame()
        fl = pd.DataFrame(rows); fl = fl[FeatureSchema.clean(fl).pipe(FeatureSchema.valid_mask).values]
        if fl.empty: return pd.DataFrame()
        res = self.h.predict(fl); res = res.sort_values("start_ts").reset_index(drop=True); recs = res.to_dict("records"); fin_status, notes, tsup, aids = [], [], [], []
        full = fl.set_index("flow_id")
        for r in recs:
            tmp = self.temporal.observe(r); st = tmp["status"]; fpd = self.fp.apply(r["src_ip"], st); st = fpd["status"]
            ev = []; aid = None
            if st != "NORMAL":
                key = (r["src_ip"], r["category"], r["dst_ip"]); old = self.alerts.open.get(key)
                if old is None or r["risk"] > self.alerts.alerts[old]["risk"] or tmp["escalated"]:
                    ev = self.h.explain(full.loc[r["flow_id"]]) + [f"Destination-port diversity {tmp['dst_port_diversity_10s']} in 10 s from {r['src_ip']}",
                         f"Connection frequency {tmp['conn_per_s_10s']:.1f} flows/s (10 s window)", f"{tmp['consecutive']} consecutive suspicious flows, EWMA risk {tmp['ewma_risk']:.2f}"]
                    if tmp["escalated"]: ev.append("Escalated REVIEW->HIGH by temporal engine")
                    if fpd["severity_note"]: ev.append(fpd["severity_note"])
                aid = self.alerts.ingest(r, st, ev or self.alerts.alerts.get(old, {}).get("evidence", []), fpd["severity_note"])
                if aid and r["category"] not in self.first_alert_time.get(r["src_ip"], {}): self.first_alert_time.setdefault(r["src_ip"], {})[r["category"]] = r["end_ts"]
                if aid:
                    msg = self.resp.maybe_auto(self.alerts.alerts[aid], fpd["auto_ok"], tmp["temporal_support"])
                    if msg: self.log.append(msg)
            fin_status.append(st); tsup.append(tmp["temporal_support"]); aids.append(aid or 0)
        res = pd.DataFrame(recs); res["final_status"] = fin_status; res["temporal_support"] = tsup; res["alert_id"] = aids
        self.flows = pd.concat([self.flows, res]).tail(50_000).reset_index(drop=True); return res
    # ---- dashboard-facing helpers ------------------------------------------------------
    def summary(self) -> dict:
        f = self.flows; n = len(f); return dict(packets=self.packets_seen, flows=n, normal=int((f["final_status"] == "NORMAL").sum()) if n else 0,
            suspicious=int((f["final_status"] != "NORMAL").sum()) if n else 0, alerts=len(self.alerts.alerts), clock=self.sim.clock, active_scenarios=[s.name for s in self.ctl.scenarios if s.active])
    def detection_summary(self, true_label: str) -> dict:
        f = self.flows[self.flows["true_label"] == true_label] if len(self.flows) else self.flows
        if f.empty: return {}
        return dict(flows=len(f), alerted=float((f["final_status"] != "NORMAL").mean()), high=float((f["final_status"] == "HIGH").mean()),
                    categories=f["category"].value_counts().to_dict())
