"""ids.py - temporal detection, alert management, false-positive management and the response engine."""
from __future__ import annotations
import itertools
from collections import deque, defaultdict
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from core import Config, get_logger
from sim import NetworkSimulator
from models import UNKNOWN

log = get_logger("ids.ids")

class TemporalDetectionEngine:
    """Per-source-host rolling windows. Computes port diversity / connection frequency (host-level features that are NOT in the flow
    schema), EWMA risk, consecutive-suspicious counts and a per-host flow-rate baseline. May escalate REVIEW -> HIGH, never downgrade."""
    def __init__(self, cfg: Config, tau_review: float, ewma_alpha: float = 0.3):
        self.cfg, self.tau, self.a = cfg, tau_review, ewma_alpha; self.h: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5000))
        self.ewma: Dict[str, float] = {}; self.consec: Dict[str, int] = defaultdict(int); self.base: Dict[str, list] = defaultdict(lambda: [0, 0.0, 0.0])
    def observe(self, row: dict) -> dict:
        src, t, R, st = row["src_ip"], row["start_ts"], row["risk"], row["status"]
        self.h[src].append((t, R, st, row["dst_ip"], row["dst_port"])); self.ewma[src] = self.a * R + (1 - self.a) * self.ewma.get(src, R)
        self.consec[src] = self.consec[src] + 1 if st != "NORMAL" else 0
        feats = {}
        for w in self.cfg.temporal_windows_s:
            rec = [r for r in self.h[src] if t - r[0] <= w]; feats[f"w{int(w)}"] = dict(flows=len(rec), conn_per_s=len(rec) / w, dst_ports=len({r[4] for r in rec}), dst_hosts=len({r[3] for r in rec}),
                                                                                           mean_risk=float(np.mean([r[1] for r in rec])), suspicious=sum(r[2] != "NORMAL" for r in rec))
        rate = feats[f"w{int(self.cfg.temporal_windows_s[0])}"]["conn_per_s"]; n, m, M2 = self.base[src]
        z = (rate - m) / (np.sqrt(M2 / n) + 1e-6) if n > 5 else 0.0
        if st == "NORMAL":   # baseline updated on normal flows only (Welford)
            n += 1; d = rate - m; m += d / n; M2 += d * (rate - m); self.base[src] = [n, m, M2]
        w0 = feats[f"w{int(self.cfg.temporal_windows_s[0])}"]
        support = (self.consec[src] >= self.cfg.temporal_consecutive and self.ewma[src] >= self.tau) or (w0["dst_ports"] >= 20 and st != "NORMAL")
        out = dict(status=st, temporal_support=bool(support), ewma_risk=float(self.ewma[src]), consecutive=int(self.consec[src]), rate_z=float(z),
                   dst_port_diversity_10s=w0["dst_ports"], conn_per_s_10s=w0["conn_per_s"], windows=feats, escalated=False)
        if st == "REVIEW" and support and row["category"] != UNKNOWN: out["status"], out["escalated"] = "HIGH", True
        return out

class FalsePositiveManager:
    """Whitelisting (alters severity/response eligibility, never disables logging) and administrator feedback (never auto-retrains)."""
    def __init__(self): self.whitelist: Dict[str, str] = {}; self.feedback: List[dict] = []
    def add_whitelist(self, ip: str, reason: str = ""): self.whitelist[ip] = reason
    def remove_whitelist(self, ip: str): self.whitelist.pop(ip, None)
    def apply(self, src_ip: str, status: str) -> dict:
        if src_ip in self.whitelist and status != "NORMAL":
            return dict(status="REVIEW" if status == "HIGH" else status, severity_note=f"whitelisted ({self.whitelist[src_ip]}): downgraded, auto-response disabled", auto_ok=False)
        return dict(status=status, severity_note="", auto_ok=True)
    def record(self, alert_id: int, verdict: str, note: str = "", category: str = ""):
        assert verdict in ("confirmed", "false_positive", "investigating"); self.feedback.append(dict(alert_id=alert_id, verdict=verdict, note=note, category=category))
    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.feedback) if self.feedback else pd.DataFrame(columns=["alert_id", "verdict", "note", "category"])
    def export_for_review(self, path: str): self.summary().to_csv(path, index=False)   # for MANUAL retraining review only

class AlertManager:
    """Aggregates flow decisions into alerts (key = src, category, dst). All fields originate from model outputs/features."""
    def __init__(self, merge_s: float = 30.0): self.alerts: Dict[int, dict] = {}; self._id = itertools.count(1); self.merge_s = merge_s; self.open: Dict[tuple, int] = {}
    def ingest(self, row: dict, status: str, evidence: List[str], note: str = "") -> Optional[int]:
        if status == "NORMAL": return None
        key = (row["src_ip"], row["category"], row["dst_ip"]); aid = self.open.get(key); sev = "CRITICAL" if status == "HIGH" else "WARNING"
        if aid is not None and row["start_ts"] - self.alerts[aid]["last_time"] <= self.merge_s:
            a = self.alerts[aid]; a["count"] += 1; a["last_time"] = row["start_ts"]
            if row["risk"] > a["risk"]: a.update(risk=row["risk"], evidence=evidence, models=self._m(row))
            if sev == "CRITICAL" and a["severity"] != "CRITICAL": a.update(severity="CRITICAL", action=row["action"])
            return aid
        aid = next(self._id); self.open[key] = aid
        self.alerts[aid] = dict(id=aid, severity=sev, src_ip=row["src_ip"], dst_ip=row["dst_ip"], category=row["category"], status_level=status, risk=row["risk"], count=1,
            first_time=row["start_ts"], last_time=row["start_ts"], evidence=evidence, models=self._m(row), agreement=row["agreement"], action=row["action"],
            investigation="new", note=note, true_label=row.get("true_label", ""))
        return aid
    @staticmethod
    def _m(r): return {k: r[k] for k in ("rf_class", "xgb_class", "rf_attack_p", "xgb_attack_p", "I_iforest", "A_ae", "ae_recon_error", "S_sup", "signals") if k in r}
    def set_status(self, aid: int, st: str): self.alerts[aid]["investigation"] = st
    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.alerts.values()) if self.alerts else pd.DataFrame()

class ResponseEngine:
    """Maps IDS decisions to simulated controls on the router: ACL/IP block, port restriction, host isolation, expiry, verification, incident log."""
    def __init__(self, sim: NetworkSimulator, cfg: Config): self.sim, self.cfg = sim, cfg; self.incidents: List[dict] = []; self.rules: Dict[int, dict] = {}
    def _log(self, action, target, outcome, **kw): self.incidents.append(dict(time=self.sim.clock, action=action, target=target, outcome=outcome, **kw))
    def _dev(self, ip): return self.sim.by_ip.get(ip)
    def block_ip(self, ip: str, ttl: Optional[float] = None, comment: str = "", alert_id=None, auto=False) -> int:
        r = self.sim.router.add_rule("deny", now=self.sim.clock, ttl=ttl if ttl is not None else self.cfg.rule_ttl_s, src=ip, hard=True, comment=comment or "IDS block")
        self.rules[r.rule_id] = dict(kind="block_ip", target=ip); self._log("block_ip", ip, "applied", rule_id=r.rule_id, alert_id=alert_id, auto=auto); return r.rule_id
    def restrict_port(self, dst_ip: str, port: int, proto: str = "TCP", src: str = "any", ttl: Optional[float] = None, alert_id=None) -> int:
        r = self.sim.router.add_rule("deny", now=self.sim.clock, ttl=ttl, src=src, dst=dst_ip, proto=proto, dport=port, hard=True, comment=f"restrict {proto}/{port}")
        self.rules[r.rule_id] = dict(kind="port", target=f"{dst_ip}:{port}"); self._log("restrict_port", f"{dst_ip}:{port}", "applied", rule_id=r.rule_id, alert_id=alert_id); return r.rule_id
    def isolate(self, ip: str, alert_id=None, auto=False):
        d = self._dev(ip)
        if d is None: raise KeyError(ip)
        self.sim.isolate(d, True); self._log("isolate_host", ip, "applied", alert_id=alert_id, auto=auto)
    def release(self, ip: str):
        d = self._dev(ip); self.sim.isolate(d, False); self.sim.dev[d].security_status = "normal"; self._log("release_host", ip, "applied")
    def unblock(self, rule_id: int):
        ok = self.sim.router.remove_rule(rule_id); self.rules.pop(rule_id, None); self._log("unblock", str(rule_id), "removed" if ok else "not-found")
    def sweep(self):
        for r in self.sim.router.expire(self.sim.clock): self._log("rule_expired", str(r.rule_id), "expired"); self.rules.pop(r.rule_id, None)
    def verify(self, src_ip: str, dst_ip: str, proto="TCP", dport=80) -> dict:
        """Mitigation verification: try the communication again in the simulator."""
        s, d = self._dev(src_ip), self._dev(dst_ip); res = self.sim.probe(s, d, proto, dport)
        self._log("verify", f"{src_ip}->{dst_ip}:{dport}", "BLOCKED" if not res["reachable"] else "STILL REACHABLE", reason=res["reason"]); return res
    def maybe_auto(self, alert: dict, auto_ok: bool, temporal_support: bool) -> Optional[str]:
        """Safeguards: auto-response only if enabled, CRITICAL, supervised category (not UNKNOWN), >=2 signals, temporal support, not whitelisted."""
        if not (self.cfg.auto_response and auto_ok and alert["severity"] == "CRITICAL" and alert["category"] != UNKNOWN and alert["models"].get("signals", 0) >= 2 and temporal_support): return None
        if any(r["kind"] == "block_ip" and r["target"] == alert["src_ip"] for r in self.rules.values()): return None
        rid = self.block_ip(alert["src_ip"], alert_id=alert["id"], auto=True, comment=f"auto:{alert['category']}"); return f"auto-blocked {alert['src_ip']} (rule {rid})"
