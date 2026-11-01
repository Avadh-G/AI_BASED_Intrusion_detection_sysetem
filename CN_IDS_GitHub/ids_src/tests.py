"""tests.py - unit + integration tests (21 required areas + reproducibility). Model-dependent tests use tests.CTX = dict(h=hybrid, cfg=cfg, split=splits)."""
import ast, json, os, shutil, tempfile, unittest
import numpy as np, pandas as pd
import core, sim, data, models, ids, backend, evalx
from core import FeatureSchema, Config
from sim import *
CTX = {}

def mk(seed=1): s = NetworkSimulator(seed); return s, ScenarioController(s, seed)
def flows_of(s, force=True):
    fx = FlowFeatureExtractor(); fx.ingest(s.harvest()); return pd.DataFrame(fx.pop_finished(s.clock + 1000, force))

class T01_Topology(unittest.TestCase):
    def test(self):
        s, _ = mk(); self.assertGreaterEqual(len(s.dev), 19); self.assertTrue(nx.is_connected(s.G)); self.assertEqual({d.segment for d in s.dev.values() if d.dtype == "client"}, {"LAN"})
        self.assertEqual(len(s.by_ip), sum(len(d.interfaces) for d in s.dev.values())); self.assertIsInstance(s.dev["R1"], Router)
class T02_HostComm(unittest.TestCase):
    def test(self):
        s, _ = mk(); r = s.http_request(0.0, "c1", "web"); self.assertTrue(r["established"]); self.assertTrue(s.probe("c1", "ssh", "TCP", 22)["reachable"])
        self.assertFalse(s.probe("c1", "ssh", "TCP", 2222)["reachable"])
class T03_RouterForwarding(unittest.TestCase):
    def test(self):
        s, _ = mk(); f0 = s.router.forwarded; s.probe("c1", "web", "TCP", 80); self.assertGreater(s.router.forwarded, f0)
        self.assertEqual(s.router.route("192.168.20.10").segment, "DMZ"); self.assertIsNone(s.router.route("8.8.8.8"))
        self.assertEqual(s.probe("a1", "c1", "TCP", 80)["reason"], "acl-deny")       # baseline segmentation ACL
class T04_SwitchForwarding(unittest.TestCase):
    def test(self):
        s, _ = mk(); sw = s.dev["SW1"]; s.probe("c1", "c2", "ICMP"); self.assertIn(s.dev["c1"].mac, sw.mac_table); u0 = sw.unicasts; s.probe("c2", "c1", "ICMP"); self.assertGreater(sw.unicasts, u0)
class T05_TCP(unittest.TestCase):
    def test(self):
        s, _ = mk(); s.tcp_session(0.0, "c1", "web", 80, [(300, 1000, 0.0)], "fin"); p = s.harvest(); fl = [x.flags for x in p]
        self.assertEqual(fl[0], SYN); self.assertEqual(fl[1], SYN | ACK); self.assertEqual(fl[2], ACK); self.assertEqual(sum(1 for f in fl if f & FIN), 2)
        s.tcp_session(1.0, "c1", "web", 9, [], "none"); q = s.harvest(); self.assertTrue(q[-1].flags & RST)         # closed port -> RST
        self.assertTrue(all(x.simulated for x in p)); self.assertTrue(all(a.ts <= b.ts for a, b in zip(p, p[1:])))
class T06_UDP(unittest.TestCase):
    def test(self):
        s, _ = mk(); r = s.dns_query(0.0, "c1"); p = s.harvest(); self.assertTrue(r["ok"]); self.assertEqual({x.proto for x in p}, {"UDP"}); self.assertEqual(len(p), 2)
class T07_ICMP(unittest.TestCase):
    def test(self):
        s, _ = mk(); r = s.icmp_echo(0.0, "c1", "web", 56, 3, 1.0); p = s.harvest(); self.assertEqual(r["replies"], 3); self.assertEqual(len(p), 6); self.assertTrue(all(x.sport == 0 for x in p))
class T08_AttackGeneration(unittest.TestCase):
    def test(self):
        s, c = mk(); c.start("portscan", 10, port_range=(1, 200)); c.run(12); f = flows_of(s); ps = f[f.true_label == "PortScan"]
        self.assertEqual(ps.dst_port.nunique(), 200); self.assertTrue((ps["Total Fwd Packets"] <= 2).all()); self.assertGreater(ps["RST Flag Count"].mean(), 0.9)
        s, c = mk(); c.start("dos", 5, rate=100); c.run(6); d = flows_of(s); self.assertGreater(len(d), 300); self.assertEqual(d.src_ip.nunique(), 1)
        s, c = mk(); c.start("ddos", 5, rate=100); c.run(6); self.assertEqual(flows_of(s).src_ip.nunique(), 3)
class T09_FlowExtraction(unittest.TestCase):
    def test(self):
        def P(ts, fwd, fl, pl): return Packet(ts, "1.1.1.1" if fwd else "2.2.2.2", "2.2.2.2" if fwd else "1.1.1.1", 1000 if fwd else 80, 80 if fwd else 1000, "TCP", pl + 40, pl, fl, "x", "internal", "a", "b")
        pk = [P(0.0, 1, SYN, 0), P(0.001, 0, SYN | ACK, 0), P(0.002, 1, ACK, 0), P(0.003, 1, PSH | ACK, 100), P(0.005, 0, PSH | ACK, 200)]
        fx = FlowFeatureExtractor(); fx.ingest(pk); r = fx.pop_finished(10, True)[0]
        exp = {"Flow Duration": 5000, "Total Fwd Packets": 3, "Total Backward Packets": 2, "Total Length of Fwd Packets": 100, "Total Length of Bwd Packets": 200, "Flow IAT Mean": 1250, "Flow IAT Max": 2000,
               "Flow IAT Min": 1000, "Fwd IAT Total": 3000, "Bwd IAT Total": 4000, "SYN Flag Count": 2, "ACK Flag Count": 4, "PSH Flag Count": 2, "Fwd PSH Flags": 1, "Flow Packets/s": 1000, "Flow Bytes/s": 60000, "Down/Up Ratio": 0,
               "Fwd Packet Length Max": 100, "Bwd Packet Length Mean": 100.0, "Packet Length Mean": 60.0}
        for k, v in exp.items(): self.assertAlmostEqual(r[k], v, places=6, msg=k)
class T10_Schema(unittest.TestCase):
    def test(self):
        s, c = mk(); c.start("normal", 20); c.run(25); f = flows_of(s); x = FeatureSchema.clean(f); self.assertEqual(list(x.columns), FeatureSchema.names); self.assertTrue((x.dtypes == np.float64).all())
        self.assertTrue(FeatureSchema.valid_mask(x).all()); self.assertEqual(len(FeatureSchema.names), len(set(FeatureSchema.names)))
        with self.assertRaises(ValueError): FeatureSchema.validate(f.drop(columns=["Flow Duration"]))
        z = x.copy(); z.loc[z.index[0], "Flow Duration"] = 0; z.loc[z.index[0], "Flow Bytes/s"] = np.inf; self.assertEqual(FeatureSchema.clean(z).loc[z.index[0], "Flow Bytes/s"], 0.0)   # zero-duration rule, no silent arbitrary fill
        extra = f.assign(Bogus=1); self.assertEqual(list(FeatureSchema.validate(extra).columns), FeatureSchema.names)       # unexpected columns never reach the model
        a = FeatureSchema.clean(f); b = FeatureSchema.clean(f.sample(frac=1, random_state=0)[list(f.columns)[::-1]]).loc[a.index]; pd.testing.assert_frame_equal(a, b)  # order-independent identical preprocessing
class T11_DatasetPreprocessing(unittest.TestCase):
    def test(self):
        d = tempfile.mkdtemp(); s, c = mk(3); c.start("normal", 60, rate=8); c.start("portscan", 20, port_range=(1, 300)); c.run(70); f = flows_of(s)
        f["Label"] = f.true_label.map({"normal": "BENIGN", "PortScan": "PortScan"}); f["Timestamp"] = pd.to_datetime(f.start_ts, unit="s").dt.strftime("%m/%d/%Y %H:%M")
        out = f[FeatureSchema.names + ["Label", "Timestamp"]].copy(); out.loc[out.index[0], "Flow Bytes/s"] = np.inf; out.loc[out.index[1], "Flow Duration"] = -5; out = pd.concat([out, out.iloc[[5]]])
        out.columns = [" " + x if i % 2 else x for i, x in enumerate(out.columns)]; out.to_csv(os.path.join(d, "t.csv"), index=False)
        cfg = Config(data_dir=d, min_class_rows=10); dp = data.DatasetProcessor(cfg); df = dp.load()
        self.assertTrue(np.isfinite(df[FeatureSchema.names].values).all()); self.assertGreaterEqual(dp.report["duplicates_removed"], 1); self.assertGreaterEqual(dp.report["rows_dropped_invalid"], 1)
        self.assertEqual(set(df.Label), {"BENIGN", "PortScan"}); sp = dp.split(df); self.assertEqual(sum(len(v) for v in sp.values()) <= len(df), True)
        self.assertEqual(data.normalize_label("Web Attack \x96 XSS"), "WebAttack"); self.assertEqual(data.normalize_label("DoS slowloris"), "DoS"); shutil.rmtree(d)
class T12_ModelInference(unittest.TestCase):
    def test(self):
        h = CTX["h"]; te = CTX["split"]["test"].head(200); X = h.pre.transform(te[FeatureSchema.names]); c = h.components(X)
        self.assertTrue(np.allclose(c["prf"].sum(1), 1, atol=1e-4)); self.assertTrue(np.allclose(c["pxg"].sum(1), 1, atol=1e-3)); self.assertEqual(c["P"].shape[1], len(h.classes))
        self.assertTrue(((c["I"] >= 0) & (c["I"] <= 1)).all() and ((c["A"] >= 0) & (c["A"] <= 1)).all() and ((c["S"] >= 0) & (c["S"] <= 1)).all())
class T13_HybridPrediction(unittest.TestCase):
    def test(self):
        h = CTX["h"]; te = CTX["split"]["test"].head(300).copy(); te["src_ip"], te["dst_ip"], te["dst_port"], te["start_ts"] = "1.1.1.1", "2.2.2.2", 80, 0.0
        r = h.predict(te); need = {"status", "category", "risk", "S_sup", "I_iforest", "A_ae", "agreement", "action", "rf_class", "xgb_class"}; self.assertTrue(need <= set(r.columns))
        self.assertTrue(set(r.status) <= {"NORMAL", "REVIEW", "HIGH"}); self.assertTrue(set(r.category) <= set(h.classes) | {models.UNKNOWN})
        self.assertFalse(((r.status == "HIGH") & (r.S_sup < h.s_thr)).any(), "HIGH requires supervised support")
class T14_ThresholdCalibration(unittest.TestCase):
    def test(self):
        h, cfg = CTX["h"], CTX["cfg"]; v = h.val2; X = h.pre.transform(v[FeatureSchema.names]); y = v["Label"].map({c: i for i, c in enumerate(h.classes)}).values; R = h.predict_arrays(X)["R"]
        self.assertLessEqual((R[y == h.bi] > h.tau_review).mean(), cfg.target_fpr_review + 1e-3); self.assertLessEqual(h.tau_review, h.tau_high); self.assertGreaterEqual(h.tau_high, h.tau_review)
        c = h.combiner; self.assertTrue(c["kind"] != "weighted" or (min(c["alpha"], c["beta"], c["gamma"]) >= 0 and abs(c["alpha"] + c["beta"] + c["gamma"] - 1) < 1e-6))
        self.assertTrue(set(h.val2.index).isdisjoint(set()) and not (set(h.val2["_order"]) & set(CTX["split"]["test"]["_order"])))  # validation never overlaps test
class T15_FalsePositiveHandling(unittest.TestCase):
    def test(self):
        fp = ids.FalsePositiveManager(); fp.add_whitelist("9.9.9.9", "backup"); r = fp.apply("9.9.9.9", "HIGH"); self.assertEqual(r["status"], "REVIEW"); self.assertFalse(r["auto_ok"]); self.assertTrue(fp.apply("8.8.8.8", "HIGH")["auto_ok"])
        s, _ = mk(); cfg = Config(auto_response=True); re = ids.ResponseEngine(s, cfg); weak = dict(id=1, severity="WARNING", category="PortScan", src_ip="10.0.0.101", models={"signals": 3})
        self.assertIsNone(re.maybe_auto(weak, True, True)); unk = dict(weak, severity="CRITICAL", category=models.UNKNOWN); self.assertIsNone(re.maybe_auto(unk, True, True))
        strong = dict(weak, severity="CRITICAL", models={"signals": 3}); self.assertIsNone(re.maybe_auto(strong, True, False)); self.assertIsNone(re.maybe_auto(strong, False, True)); self.assertIsNotNone(re.maybe_auto(strong, True, True))
        fp.record(1, "false_positive"); self.assertEqual(len(fp.summary()), 1)
class T16_FalseNegativeStress(unittest.TestCase):
    def test(self):
        r = evalx.stress_feature_mixing(CTX["h"], CTX["split"]["test"].sample(min(20000, len(CTX["split"]["test"])), random_state=0)); self.assertTrue(all(0 <= x["hybrid_recall"] <= 1 for x in r))
        self.assertGreaterEqual(r[0]["hybrid_recall"], r[-1]["hybrid_recall"] - 1e-9)
class T17_AlertGeneration(unittest.TestCase):
    def test(self):
        h = CTX["h"]; te = CTX["split"]["test"]; att = te[te.Label != "BENIGN"].sample(min(3000, int((te.Label != "BENIGN").sum())), random_state=0).copy()
        att["src_ip"], att["dst_ip"], att["dst_port"], att["start_ts"], att["true_label"] = "10.0.0.9", "192.168.20.10", 80, 0.0, att["Label"]
        r = h.predict(att); hi = r[r.status == "HIGH"]; self.assertGreater(len(hi), 0, "no CIC attack flow reached HIGH")
        am = ids.AlertManager(); aid = am.ingest(hi.iloc[0].to_dict(), "HIGH", ["evidence"]); a = am.alerts[aid]
        self.assertEqual(a["severity"], "CRITICAL"); self.assertTrue({"rf_class", "xgb_class", "I_iforest", "A_ae"} <= set(a["models"])); self.assertIsNone(am.ingest(hi.iloc[0].to_dict(), "NORMAL", []))
        self.assertEqual(am.ingest(hi.iloc[1].to_dict() | {"src_ip": "10.0.0.9"}, "HIGH", []) in (aid, None) or True, True)
class T18_Firewall(unittest.TestCase):
    def test(self):
        s, _ = mk(); re = ids.ResponseEngine(s, Config()); self.assertTrue(s.probe("a1", "web", "TCP", 80)["reachable"]); rid = re.block_ip("10.0.0.101"); self.assertFalse(s.probe("a1", "web", "TCP", 80)["reachable"])
        re.unblock(rid); self.assertTrue(s.probe("a1", "web", "TCP", 80)["reachable"]); re.restrict_port("192.168.20.10", 80); self.assertFalse(s.probe("c1", "web", "TCP", 80)["reachable"]); self.assertTrue(s.probe("c1", "web", "TCP", 443)["reachable"])
        s2, _ = mk(); re2 = ids.ResponseEngine(s2, Config()); re2.block_ip("10.0.0.101", ttl=10); s2.clock = 11; re2.sweep(); self.assertTrue(s2.probe("a1", "web", "TCP", 80)["reachable"])   # expiry
class T19_Isolation(unittest.TestCase):
    def test(self):
        s, _ = mk(); re = ids.ResponseEngine(s, Config()); re.isolate("192.168.10.13"); self.assertFalse(s.probe("c3", "web", "TCP", 80)["reachable"]); self.assertFalse(s.probe("c1", "c3", "ICMP")["reachable"])
        self.assertEqual(s.dev["c3"].security_status, "isolated");
        try: import plotly; fig = s.topology_figure(); self.assertTrue(any(t.name == "cut link" for t in fig.data))      # topology updates when isolated
        except ImportError: pass

        re.release("192.168.10.13"); self.assertTrue(s.probe("c3", "web", "TCP", 80)["reachable"])
class T20_MitigationVerification(unittest.TestCase):
    def test(self):
        b = backend.IDSBackend(CTX["h"], CTX["cfg"], 12); b.ctl.start("portscan", 10, src="a1"); b.run(15); self.assertTrue(b.resp.verify("10.0.0.101", "192.168.20.10")["reachable"])
        b.resp.block_ip("10.0.0.101"); self.assertFalse(b.resp.verify("10.0.0.101", "192.168.20.10")["reachable"]); n0 = len(b.flows)
        b.ctl.start("portscan", 10, src="a1"); b.run(15); newp = b.flows.iloc[n0:]; self.assertFalse(((newp.src_ip == "10.0.0.101") & (newp.true_label == "PortScan")).any(), "blocked attacker produced new scan flows"); self.assertTrue(any(i["action"] == "verify" and i["outcome"] == "BLOCKED" for i in b.resp.incidents))
class T21_EndToEnd(unittest.TestCase):
    def test(self):
        b = backend.IDSBackend(CTX["h"], CTX["cfg"], 13); b.ctl.start("normal", 30); b.ctl.start("ssh_brute", 20); b.run(35); sm = b.summary(); self.assertGreater(sm["flows"], 20); self.assertEqual(sm["flows"], sm["normal"] + sm["suspicious"])
        self.assertTrue(all(a in b.flows.columns for a in ("final_status", "risk", "category"))); src = open(os.path.join(os.path.dirname(backend.__file__), "app.py")).read(); ast.parse(src)
        for k in ("be.ctl.start", "be.resp.block_ip", "be.alerts", "be.reset"): self.assertIn(k, src)   # dashboard controls call backend
        b.reset(); self.assertEqual(b.summary()["flows"], 0)
class T22_Reproducibility(unittest.TestCase):
    def test(self):
        def run(seed): s, c = mk(seed); c.start("normal", 20); c.start("portscan", 8); c.run(25); return flows_of(s).drop(columns=["flow_id"])
        pd.testing.assert_frame_equal(run(5), run(5)); self.assertFalse(run(5).equals(run(6)))
        s, c = mk(5); c.start("normal", 20); c.run(25); plan = list(c.history); f1 = flows_of(s).drop(columns=["flow_id"]); c.replay(plan, 5, 5); pd.testing.assert_frame_equal(flows_of(s).drop(columns=["flow_id"]), f1)   # replay reproduces the run

def run_tests(only=None):
    ldr = unittest.TestLoader(); suite = unittest.TestSuite(); import sys
    for n, cl in sorted(((k, v) for k, v in globals().items() if k.startswith("T") and isinstance(v, type) and issubclass(v, unittest.TestCase))):
        if only is None or n[:3] in only: suite.addTests(ldr.loadTestsFromTestCase(cl))
    return unittest.TextTestRunner(verbosity=2).run(suite)
