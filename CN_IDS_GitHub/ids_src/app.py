"""app.py - SOC-style Streamlit dashboard. Every control calls the real backend (IDSBackend); metrics come from saved evaluation JSON."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, streamlit as st, plotly.express as px, plotly.graph_objects as go
from core import Config, FeatureSchema
from backend import IDSBackend
from models import HybridIntrusionDetector

ART = os.environ.get("IDS_ARTIFACTS", "/content/ids_artifacts")
st.set_page_config(page_title="AI-IDS SOC", layout="wide", page_icon="🛡️")

@st.cache_resource
def load_model(): return HybridIntrusionDetector.load(ART)
def metrics():
    p = os.path.join(ART, "metrics.json"); return json.load(open(p)) if os.path.exists(p) else None

cfg = Config(work_dir=ART)
if "be" not in st.session_state: st.session_state.be = IDSBackend(load_model(), cfg, seed=cfg.seed); st.session_state.auth = False
be: IDSBackend = st.session_state.be

st.sidebar.title("🛡️ AI-Based IDS")
st.sidebar.caption("All traffic is SIMULATED inside a custom Python network model.")
pw = os.environ.get("IDS_PASSWORD", "")
if not st.session_state.auth:
    if pw == "": st.session_state.auth = False; st.sidebar.warning("No IDS_PASSWORD set: control pages are locked (read-only mode).")
    else:
        x = st.sidebar.text_input("Admin password", type="password")
        if x and x == pw: st.session_state.auth = True; st.rerun()
        elif x: st.sidebar.error("Wrong password")
else:
    st.sidebar.success("Admin unlocked")
page = st.sidebar.radio("Page", ["1 Network Overview", "2 Live Traffic", "3 Attack Simulation", "4 Hybrid AI Monitoring", "5 Security Alerts", "6 Firewall & Response", "7 Model Evaluation"])
s = be.summary(); st.sidebar.metric("Sim clock (s)", f"{s['clock']:.0f}"); st.sidebar.metric("Flows / alerts", f"{s['flows']} / {s['alerts']}")
def locked():
    if not st.session_state.auth: st.warning("🔒 Read-only: unlock with the admin password to use controls."); return True
    return False
F = be.flows
COLS = ["start_ts", "src_ip", "dst_ip", "src_port", "dst_port", "proto", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Flow Duration", "category", "risk", "final_status"]

if page.startswith("1"):
    st.header("Network Overview"); c = st.columns(5)
    c[0].metric("Packet events", s["packets"]); c[1].metric("Flows", s["flows"]); c[2].metric("Normal", s["normal"]); c[3].metric("Suspicious", s["suspicious"]); c[4].metric("Alerts", s["alerts"])
    st.plotly_chart(be.sim.topology_figure(), use_container_width=True)
    a, b = st.columns(2)
    a.subheader("Devices"); a.dataframe(pd.DataFrame([d.to_dict() for d in be.sim.dev.values()]).drop(columns=["services"]), height=300)
    b.subheader("Recent connections"); b.dataframe(F[COLS].tail(15) if len(F) else F, height=300)
elif page.startswith("2"):
    st.header("Live Traffic Monitoring")
    if F.empty: st.info("No flows yet. Use 'Attack Simulation' to start traffic.")
    else:
        c = st.columns(4); hosts = c[0].multiselect("Host (src or dst)", sorted(set(F.src_ip) | set(F.dst_ip))); pr = c[1].multiselect("Protocol", sorted(F.proto.unique()))
        pd_ = c[2].multiselect("Prediction", sorted(F.category.unique())); win = c[3].slider("Last N sim-seconds", 10, int(max(20, s["clock"])), int(max(20, s["clock"])))
        d = F[F.start_ts >= s["clock"] - win]
        if hosts: d = d[d.src_ip.isin(hosts) | d.dst_ip.isin(hosts)]
        if pr: d = d[d.proto.isin(pr)]
        if pd_: d = d[d.category.isin(pd_)]
        d = d.assign(packets=d["Total Fwd Packets"] + d["Total Backward Packets"], bytes=d["Total Length of Fwd Packets"] + d["Total Length of Bwd Packets"])
        st.dataframe(d[["start_ts", "src_ip", "src_port", "dst_ip", "dst_port", "proto", "packets", "bytes", "Flow Duration", "category", "risk", "final_status"]].tail(500), height=450)
        st.plotly_chart(px.histogram(d, x="start_ts", color="final_status", nbins=60, title="Flows over simulated time"), use_container_width=True)
elif page.startswith("3"):
    st.header("Attack Simulation (controls call the real simulator)")
    if not locked():
        dur = st.slider("Scenario duration (sim s)", 5, 120, 30); inten = st.slider("Intensity multiplier", 0.1, 2.0, 1.0, 0.1)
        ramp = st.slider("Gradual ramp-up (s, 0 = immediate)", 0, 60, 0)
        btn = {"Start normal traffic": "normal", "Start DoS": "dos", "Start DDoS": "ddos", "Start port scan": "portscan", "Start FTP brute-force": "ftp_brute",
               "Start SSH brute-force": "ssh_brute", "Start botnet beaconing": "botnet", "Start suspicious web requests": "web_attack", "Legit burst (FP test)": "burst", "Unusual protocol (unknown)": "unusual"}
        cols = st.columns(3)
        for i, (lab, k) in enumerate(btn.items()):
            if cols[i % 3].button(lab):
                extra = {} if k in ("portscan", "botnet", "unusual") else {"ramp": float(ramp)}
                be.ctl.start(k, float(dur), intensity=float(inten), **extra) if k not in ("normal",) else be.ctl.start(k, float(dur))
                with st.spinner("Simulating + detecting..."): be.run(dur + 5, dt=2.0, flush=True)
                st.success(f"{lab}: {be.summary()['flows']} flows, {be.summary()['alerts']} alerts"); st.rerun()
        c2 = st.columns(3)
        if c2[0].button("Advance 10 s"): be.run(10, flush=False); st.rerun()
        if c2[1].button("Stop scenarios"): be.ctl.stop(); st.info("Active scenarios stopped");
        if c2[2].button("Reset network"): be.reset(); st.rerun()
        st.write("Active:", s["active_scenarios"]); st.write("Scenario history:", pd.DataFrame(be.ctl.history).drop(columns=["params"], errors="ignore"))
elif page.startswith("4"):
    st.header("Hybrid AI Monitoring")
    if F.empty: st.info("No data yet.")
    else:
        d = F.tail(1000); st.dataframe(d[["start_ts", "src_ip", "dst_ip", "category", "final_status", "risk", "rf_class", "rf_attack_p", "xgb_class", "xgb_attack_p", "I_iforest", "ae_recon_error", "A_ae", "S_sup", "agreement"]].tail(300), height=380)
        st.plotly_chart(px.scatter(d, x="start_ts", y="risk", color="final_status", hover_data=["src_ip", "category"], title="Hybrid risk score"), use_container_width=True)
        a, b = st.columns(2); a.plotly_chart(px.histogram(d, x="I_iforest", nbins=40, title="Isolation Forest (benign-CDF quantile)"), use_container_width=True)
        b.plotly_chart(px.histogram(d, x="ae_recon_error", nbins=40, log_y=True, title="Autoencoder reconstruction error"), use_container_width=True)
        dis = d[d.agreement.str.contains("DISAGREE")]; st.metric("Model-disagreement flows (last 1000)", len(dis))
        sel = st.selectbox("Explain flow", d[d.final_status != "NORMAL"].flow_id.tolist()[-200:] or ["-"])
        if sel != "-":
            row = be.flows[be.flows.flow_id == sel].iloc[0]; st.json({k: (v if not isinstance(v, float) else round(v, 4)) for k, v in row.items() if k in ("src_ip", "dst_ip", "category", "final_status", "risk", "agreement", "action")})
            st.write("\n".join("• " + e for e in be.h.explain(row)))
elif page.startswith("5"):
    st.header("Security Alerts"); t = be.alerts.table()
    if t.empty: st.info("No alerts.")
    else:
        st.dataframe(t[["id", "severity", "src_ip", "dst_ip", "category", "risk", "count", "first_time", "last_time", "agreement", "action", "investigation"]], height=300)
        aid = st.selectbox("Alert", t.id.tolist()); a = be.alerts.alerts[aid]
        st.subheader(f"Alert {aid}: {a['severity']} – {a['category']}"); st.write(f"SOURCE {a['src_ip']} → DESTINATION {a['dst_ip']} | RISK {a['risk']:.3f} | {a['count']} flows")
        st.write("**Models:**", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in a["models"].items()}); st.write("**Agreement:**", a["agreement"]); st.write("**Recommended:**", a["action"])
        st.write("**Evidence:**"); st.write("\n".join("• " + e for e in a["evidence"]))
        if not locked():
            c = st.columns(4)
            for col, (lab, v) in zip(c, [("Acknowledge", "acknowledged"), ("Confirmed attack", "confirmed"), ("False positive", "false_positive"), ("Investigating", "investigating")]):
                if col.button(lab):
                    be.alerts.set_status(aid, v)
                    if v in ("confirmed", "false_positive", "investigating"): be.fp.record(aid, v, category=a["category"])
                    st.rerun()
            ip = st.text_input("Whitelist source IP (reduces severity/auto-response, keeps logging)", a["src_ip"])
            if st.button("Add to whitelist"): be.fp.add_whitelist(ip, "admin"); st.success("Whitelisted")
        st.subheader("Feedback history (never auto-retrained)"); st.dataframe(be.fp.summary())
elif page.startswith("6"):
    st.header("Firewall & Response")
    st.dataframe(pd.DataFrame([vars(r) for r in be.sim.router.acl]), height=220)
    if not locked():
        be.cfg.auto_response = st.checkbox("Enable automatic response (guarded: CRITICAL + supervised category + ≥2 signals + temporal support + not whitelisted)", be.cfg.auto_response)
        c = st.columns(2); ip = c[0].text_input("IP", "10.0.0.101")
        if c[0].button("Block IP"): be.resp.block_ip(ip); st.rerun()
        if c[0].button("Isolate host"): be.resp.isolate(ip); st.rerun()
        if c[0].button("Release host"): be.resp.release(ip); st.rerun()
        rid = c[1].number_input("Rule id", 1, 9999, 1)
        if c[1].button("Unblock (remove rule)"): be.resp.unblock(int(rid)); st.rerun()
        pip = c[1].text_input("Restrict dst IP", "192.168.20.10"); pp = c[1].number_input("Port", 1, 65535, 80)
        if c[1].button("Restrict port"): be.resp.restrict_port(pip, int(pp)); st.rerun()
        st.subheader("Mitigation verification"); v = st.columns(4); a_ = v[0].text_input("From IP", "10.0.0.101"); b_ = v[1].text_input("To IP", "192.168.20.10"); pr = v[2].selectbox("Proto", ["TCP", "UDP", "ICMP"]); po = v[3].number_input("dport", 1, 65535, 80)
        if st.button("Verify now"): r = be.resp.verify(a_, b_, pr, int(po)); (st.success if not r["reachable"] else st.error)(f"{'BLOCKED' if not r['reachable'] else 'STILL REACHABLE'} ({r['reason']})")
    st.subheader("Incident history"); st.dataframe(pd.DataFrame(be.resp.incidents))
    if be.log: st.write(be.log)
    st.plotly_chart(be.sim.topology_figure(), use_container_width=True)
else:
    st.header("Model Evaluation (CIC-IDS2017 held-out test, measured)"); m = metrics()
    if m is None: st.error("metrics.json not found – run the evaluation cells in the notebook first.")
    else:
        st.dataframe(pd.DataFrame(m["comparison"]).set_index("system").round(4)); st.caption(f"Test rows: {m['test_rows']}. Thresholds were fixed on validation data only.")
        k = st.selectbox("Confusion matrix", list(m["confusion"])); cm = m["confusion"][k]; st.plotly_chart(px.imshow(np.array(cm["matrix"]), x=cm["labels"], y=cm["labels"], text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted", y="True")), use_container_width=True)
        st.plotly_chart(px.bar(pd.DataFrame(m["per_class_recall"]).reset_index().melt("index"), x="index", y="value", color="variable", barmode="group", title="Per-class recall"), use_container_width=True)
        fig = go.Figure(); fig2 = go.Figure()
        for n, c in m["curves"].items(): fig.add_scatter(x=c["recall"], y=c["precision"], name=n); fig2.add_scatter(x=c["fpr"], y=c["tpr"], name=n)
        fig.update_layout(title="Precision-Recall", xaxis_title="Recall", yaxis_title="Precision"); fig2.update_layout(title="ROC", xaxis_title="FPR", yaxis_title="TPR")
        a, b = st.columns(2); a.plotly_chart(fig, use_container_width=True); b.plotly_chart(fig2, use_container_width=True)
        for key, title in [("ablation", "Ablation"), ("stress_feature_mixing", "Stress: attacks blended toward benign"), ("leave_one_class_out", "Leave-one-class-out (novel attack proxy)"), ("sim_stress", "Simulator stress (tier 2)"), ("imbalance", "Imbalance strategies (val macro-F1)")]:
            if key in m: st.subheader(title); st.dataframe(pd.DataFrame(m[key]))
        st.subheader("Latency / throughput"); st.json(m.get("latency", {})); st.subheader("Selected hybrid formulation"); st.json(m.get("hybrid_config", {}))
        st.subheader("Simulator-vs-CIC domain gap (KS)"); st.dataframe(pd.DataFrame(m.get("domain_gap", [])))
