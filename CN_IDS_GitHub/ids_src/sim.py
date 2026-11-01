"""sim.py - custom Python network simulator (SIMULATED traffic only; no OS packets are ever created)."""
from __future__ import annotations
import heapq, ipaddress, itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import networkx as nx
from core import get_logger, Config

log = get_logger("ids.sim")
FIN, SYN, RST, PSH, ACK, URG = 1, 2, 4, 8, 16, 32
HDR = {"TCP": 40, "UDP": 28, "ICMP": 28}
MSS = 1460

# ------------------------------------------------------------------ data structures
@dataclass
class Packet:
    """One simulated packet event as captured by the monitoring node."""
    ts: float; src_ip: str; dst_ip: str; sport: int; dport: int; proto: str
    size: int; payload: int; flags: int; conn_id: str; direction: str
    src_dev: str; dst_dev: str; label: str = "normal"; meta: Dict[str, Any] = field(default_factory=dict)
    simulated: bool = True

@dataclass
class Interface:
    name: str; ip: str; mac: str; segment: str; prefix: int = 24

class Device:
    """Base network device."""
    dtype = "device"
    def __init__(self, dev_id: str, name: str, segment: str):
        self.id, self.name, self.segment = dev_id, name, segment
        self.interfaces: List[Interface] = []
        self.services: Dict[int, str] = {}
        self.connection_state = "up"          # up | down
        self.security_status = "normal"       # normal | blocked | isolated | compromised
    @property
    def ip(self) -> Optional[str]: return self.interfaces[0].ip if self.interfaces else None
    @property
    def mac(self) -> Optional[str]: return self.interfaces[0].mac if self.interfaces else None
    def add_interface(self, name, ip, mac, segment, prefix=24): self.interfaces.append(Interface(name, ip, mac, segment, prefix))
    def to_dict(self): return dict(id=self.id, name=self.name, type=self.dtype, ip=self.ip, mac=self.mac, segment=self.segment,
                                    status=self.security_status, state=self.connection_state, services=self.services)

class Host(Device): dtype = "host"
class Client(Host): dtype = "client"
class Server(Host): dtype = "server"
class Attacker(Host): dtype = "attacker"
class MonitoringNode(Device):
    dtype = "monitor"
    def __init__(self, *a, **k): super().__init__(*a, **k); self.captured = 0

class Switch(Device):
    """Learning switch: MAC table, unicast when known else flood."""
    dtype = "switch"
    def __init__(self, *a, **k): super().__init__(*a, **k); self.mac_table: Dict[str, str] = {}; self.floods = 0; self.unicasts = 0
    def forward(self, src_mac: str, dst_mac: str, ingress: str) -> str:
        self.mac_table[src_mac] = ingress
        if dst_mac in self.mac_table: self.unicasts += 1; return "unicast"
        self.floods += 1; return "flood"

@dataclass
class ACLRule:
    rule_id: int; action: str; src: str = "any"; dst: str = "any"; proto: str = "any"; dport: Optional[int] = None
    hard: bool = False            # hard rules also drop packets of already-established connections
    expires: Optional[float] = None; comment: str = ""; created: float = 0.0; hits: int = 0
    def matches(self, p: Packet) -> bool:
        def inn(ip, net): return net == "any" or ipaddress.ip_address(ip) in ipaddress.ip_network(net, strict=False)
        return (inn(p.src_ip, self.src) and inn(p.dst_ip, self.dst) and self.proto in ("any", p.proto)
                and (self.dport is None or self.dport == p.dport))

class Router(Device):
    """Router with longest-prefix routing and an ordered ACL (first match wins, default permit)."""
    dtype = "router"
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.acl: List[ACLRule] = []; self._ids = itertools.count(1); self.forwarded = 0; self.denied = 0
    def route(self, dst_ip: str) -> Optional[Interface]:
        best, bl = None, -1
        for itf in self.interfaces:
            net = ipaddress.ip_network(f"{itf.ip}/{itf.prefix}", strict=False)
            if ipaddress.ip_address(dst_ip) in net and net.prefixlen > bl: best, bl = itf, net.prefixlen
        return best
    def add_rule(self, action, now=0.0, ttl=None, **kw) -> ACLRule:
        r = ACLRule(next(self._ids), action, created=now, expires=(now + ttl) if ttl else None, **kw); self.acl.append(r); return r
    def remove_rule(self, rule_id: int) -> bool:
        n = len(self.acl); self.acl = [r for r in self.acl if r.rule_id != rule_id]; return len(self.acl) < n
    def expire(self, now: float) -> List[ACLRule]:
        gone = [r for r in self.acl if r.expires is not None and r.expires <= now]
        self.acl = [r for r in self.acl if r not in gone]; return gone
    def check(self, p: Packet, established: bool) -> Tuple[str, Optional[ACLRule]]:
        for r in self.acl:
            if established and not r.hard: continue
            if r.matches(p): r.hits += 1; return r.action, r
        return "permit", None

# ------------------------------------------------------------------ simulator
class NetworkSimulator:
    """Topology (NetworkX) + packet forwarding + connection tracking + capture at the monitoring tap."""
    def __init__(self, seed: int = 42, loss: float = 0.0, jitter: float = 0.1):
        self.seed, self.loss, self.jitter = seed, loss, jitter
        self.reset(seed)
    def reset(self, seed: Optional[int] = None):
        if seed is not None: self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.G = nx.Graph(); self.dev: Dict[str, Device] = {}; self.by_ip: Dict[str, str] = {}
        self.clock = 0.0; self.pending: List[Tuple[float, int, Packet]] = []; self._seq = itertools.count()
        self.conn_table: Dict[str, bool] = {}; self._cid = itertools.count(1); self.drop_log: List[dict] = []
        self.stats = dict(delivered=0, dropped=0, sessions=0); self._path_cache: Dict[Tuple[str, str], Optional[List[str]]] = {}
        self.build_default_topology()

    # ---- topology ---------------------------------------------------------------------
    def _add(self, d: Device, **pos):
        self.dev[d.id] = d; self.G.add_node(d.id, **pos)
        for i in d.interfaces: self.by_ip[i.ip] = d.id
    def _link(self, a, b, lat): self.G.add_edge(a, b, latency=lat); self._path_cache.clear()
    @staticmethod
    def _mac(n): return "02:00:00:00:%02x:%02x" % (n // 256, n % 256)
    def build_default_topology(self):
        n = itertools.count(1)
        r = Router("R1", "Router-1", "core")
        for nm, ip, seg in [("eth0", "192.168.10.1", "LAN"), ("eth1", "192.168.20.1", "DMZ"), ("eth2", "10.0.0.1", "EXT"), ("eth3", "192.168.30.1", "MGMT")]:
            r.add_interface(nm, ip, self._mac(next(n)), seg)
        self._add(r, x=0, y=0)
        for sid, seg, x, y in [("SW1", "LAN", -3, -1.5), ("SW2", "DMZ", 0, -3), ("SW3", "EXT", 3, -1.5)]:
            s = Switch(sid, sid, seg); s.add_interface("sw", f"0.0.0.{len(self.dev)}", self._mac(next(n)), seg); self._add(s, x=x, y=y)
        m = MonitoringNode("MON", "IDS-Monitor", "MGMT"); m.add_interface("eth0", "192.168.30.2", self._mac(next(n)), "MGMT"); self._add(m, x=0, y=1.6)
        for i in range(1, 7):
            c = Client(f"c{i}", f"Client-{i}", "LAN"); c.add_interface("eth0", f"192.168.10.{10+i}", self._mac(next(n)), "LAN")
            self._add(c, x=-5 + (i - 1) * 0.55, y=-3.4)
        for sid, nm, last, svc, x in [("web", "Web-Server", 10, {80: "http", 443: "https"}, -1.6), ("dns", "DNS-Server", 11, {53: "dns"}, -0.55),
                                       ("ftp", "FTP-Server", 12, {21: "ftp"}, 0.55), ("ssh", "SSH-Server", 13, {22: "ssh"}, 1.6)]:
            s = Server(sid, nm, "DMZ"); s.add_interface("eth0", f"192.168.20.{last}", self._mac(next(n)), "DMZ"); s.services = svc; self._add(s, x=x, y=-4.6)
        for i in range(1, 4):
            a = Attacker(f"a{i}", f"Attacker-{i}", "EXT"); a.add_interface("eth0", f"10.0.0.{100+i}", self._mac(next(n)), "EXT"); self._add(a, x=4.5 + (i - 1) * 0.7, y=-3.4)
        c2 = Server("cnc", "Botnet-C2", "EXT"); c2.add_interface("eth0", "10.0.0.50", self._mac(next(n)), "EXT"); c2.services = {8080: "c2-http"}; self._add(c2, x=6.8, y=-0.5)
        for i in range(1, 7): self._link(f"c{i}", "SW1", 0.0001)
        for s in ("web", "dns", "ftp", "ssh"): self._link(s, "SW2", 0.0001)
        for a in ("a1", "a2", "a3", "cnc"): self._link(a, "SW3", 0.0002)
        self._link("SW1", "R1", 0.0002); self._link("SW2", "R1", 0.0002); self._link("SW3", "R1", 0.008); self._link("MON", "R1", 0.00005)
        R = self.dev["R1"]
        R.add_rule("deny", src="10.0.0.0/24", dst="192.168.10.0/24", comment="baseline: EXT cannot initiate to LAN")
        R.add_rule("deny", src="10.0.0.0/24", dst="192.168.30.0/24", comment="baseline: EXT cannot reach MGMT")
        self.router = R

    # ---- device / path helpers ---------------------------------------------------------
    def _active(self, d: str) -> bool:
        x = self.dev[d]; return x.connection_state == "up" and x.security_status != "isolated"
    def path(self, a: str, b: str) -> Optional[List[str]]:
        k = (a, b)
        if k in self._path_cache: return self._path_cache[k]
        H = self.G.subgraph([n for n in self.G.nodes if self._active(n)])
        try: p = nx.shortest_path(H, a, b, weight="latency")
        except (nx.NetworkXNoPath, nx.NodeNotFound): p = None
        self._path_cache[k] = p; return p
    def isolate(self, dev_id: str, on: bool = True):
        self.dev[dev_id].security_status = "isolated" if on else "normal"; self._path_cache.clear()
    def one_way(self, a: str, b: str) -> Optional[float]:
        p = self.path(a, b)
        return None if p is None else sum(self.G[u][v]["latency"] for u, v in zip(p, p[1:]))
    def segment_of(self, ip): return self.dev[self.by_ip[ip]].segment
    def new_cid(self, proto, a, ap, b, bp): return f"{proto}-{a}:{ap}-{b}:{bp}-{next(self._cid)}"

    # ---- packet delivery ----------------------------------------------------------------
    def deliver(self, t: float, src: str, dst: str, sport: int, dport: int, proto: str, flags: int, payload: int,
                cid: str, label: str = "normal", meta: Optional[dict] = None) -> Tuple[bool, float, str]:
        """Forward one packet hop by hop. Returns (delivered, arrival_time, reason). Captures at the tap after ACL."""
        sd, dd = self.dev[src], self.dev[dst]
        size = payload + HDR[proto]
        def drop(reason, rule=None):
            self.stats["dropped"] += 1
            self.drop_log.append(dict(ts=t, src_ip=sd.ip, dst_ip=dd.ip, dport=dport, proto=proto, reason=reason, rule=getattr(rule, "rule_id", None)))
            return False, t, reason
        if not (self._active(src) and self._active(dst)): return drop("isolated-or-down")
        p = self.path(src, dst)
        if p is None: return drop("no-route")
        if self.loss and self.rng.random() < self.loss: return drop("loss")
        sip, dip = sd.ip, dd.ip
        direction = ("internal" if self.segment_of(sip) == self.segment_of(dip) or {self.segment_of(sip), self.segment_of(dip)} <= {"LAN", "DMZ"}
                     else ("inbound" if self.segment_of(sip) == "EXT" else "outbound"))
        pkt = Packet(0.0, sip, dip, sport, dport, proto, size, payload, flags, cid, direction, src, dst, label, meta or {})
        established = cid in self.conn_table
        now, captured, prev = t, False, p[0]
        for node in p[1:-1]:
            now += self.G[prev][node]["latency"] * max(0.2, 1 + self.rng.normal(0, self.jitter))
            d = self.dev[node]
            if isinstance(d, Switch): d.forward(sd.mac, dd.mac, prev)
            elif isinstance(d, Router):
                if d.route(dip) is None: return drop("no-route-to-host")
                act, rule = d.check(pkt, established)
                if act == "deny": d.denied += 1; return drop("acl-deny", rule)
                d.forwarded += 1
            if not captured: pkt.ts = now; captured = True
            prev = node
        now += self.G[prev][dst]["latency"]
        if not captured: pkt.ts = now
        is_new = (proto == "TCP" and (flags & SYN) and not (flags & ACK)) or (proto != "TCP" and not established)
        if is_new: self.conn_table[cid] = True
        heapq.heappush(self.pending, (pkt.ts, next(self._seq), pkt))
        self.stats["delivered"] += 1; self.dev["MON"].captured += 1
        return True, now, "ok"

    def harvest(self, until: Optional[float] = None) -> List[Packet]:
        """Pop captured packets with ts <= until (all if None), in timestamp order."""
        out = []
        while self.pending and (until is None or self.pending[0][0] <= until): out.append(heapq.heappop(self.pending)[2])
        return out

    # ---- protocol behaviours ------------------------------------------------------------
    def eph(self) -> int: return int(self.rng.integers(32768, 61000))
    def _segs(self, n: int) -> List[int]:
        if n <= 0: return [0]
        out = [MSS] * (n // MSS); r = n % MSS
        if r: out.append(r)
        return out
    def tcp_session(self, t0: float, src: str, dst: str, dport: int, exchanges: List[Tuple[int, int, float]], close: str = "fin",
                    label: str = "normal", sport: Optional[int] = None, handshake: bool = True, meta: Optional[dict] = None,
                    proc: float = 0.00005) -> dict:
        """Full TCP lifecycle: 3-way handshake, request/response exchanges (req_bytes, resp_bytes, think_s), teardown (fin|rst|none).
        Closed port -> RST/ACK. ACL drop -> no reply. Returns summary dict incl. 'established' and 'end' time."""
        sp = sport or self.eph(); dd = self.dev[dst]; cid = self.new_cid("TCP", self.dev[src].ip, sp, dd.ip, dport)
        self.stats["sessions"] += 1
        def S(t, a, b, s, d, fl, pl): return self.deliver(t, a, b, s, d, "TCP", fl, pl, cid, label, meta)
        ok, t, why = S(t0, src, dst, sp, dport, SYN, 0)
        if not ok: return dict(established=False, reason=why, end=t0, cid=cid)
        t += proc
        if dport not in dd.services:                                  # closed port
            ok, t, why = S(t, dst, src, dport, sp, RST | ACK, 0)
            return dict(established=False, reason="closed-port", end=t, cid=cid)
        if not handshake:                                             # half-open scan: SYN/ACK then client RST
            ok, t, _ = S(t, dst, src, dport, sp, SYN | ACK, 0); t += proc
            ok, t, _ = S(t, src, dst, sp, dport, RST, 0)
            return dict(established=False, reason="half-open", end=t, cid=cid)
        ok, t, _ = S(t, dst, src, dport, sp, SYN | ACK, 0); t += proc
        ok, t, _ = S(t, src, dst, sp, dport, ACK, 0); t += proc
        for req, resp, think in exchanges:
            t += think
            segs = self._segs(req)
            for i, s in enumerate(segs):
                ok, t2, _ = S(t, src, dst, sp, dport, (PSH | ACK) if i == len(segs) - 1 else ACK, s); t = t2 + proc * 0.2
            ok, t, _ = S(t + proc, dst, src, dport, sp, ACK, 0); t += proc
            if resp > 0:
                segs = self._segs(resp)
                for i, s in enumerate(segs):
                    ok, t2, _ = S(t, dst, src, dport, sp, (PSH | ACK) if i == len(segs) - 1 else ACK, s); t = t2 + proc * 0.2
                    if i % 2 == 1 or i == len(segs) - 1: ok, t3, _ = S(t2 + proc, src, dst, sp, dport, ACK, 0); t = max(t, t3) + proc * 0.2
        if close == "fin":
            ok, t, _ = S(t + proc, src, dst, sp, dport, FIN | ACK, 0); t += proc
            ok, t, _ = S(t, dst, src, dport, sp, ACK, 0); t += proc
            ok, t, _ = S(t, dst, src, dport, sp, FIN | ACK, 0); t += proc
            ok, t, _ = S(t, src, dst, sp, dport, ACK, 0)
        elif close == "rst":
            ok, t, _ = S(t + proc, src, dst, sp, dport, RST | ACK, 0)
        return dict(established=True, reason="ok", end=t, cid=cid, sport=sp)

    def udp_exchange(self, t0, src, dst, dport, req, resp, label="normal", sport=None, rtt_extra=0.0005, meta=None) -> dict:
        sp = sport or self.eph(); dd = self.dev[dst]; cid = self.new_cid("UDP", self.dev[src].ip, sp, dd.ip, dport)
        ok, t, why = self.deliver(t0, src, dst, sp, dport, "UDP", 0, req, cid, label, meta)
        if not ok: return dict(ok=False, reason=why, end=t0)
        if dport in dd.services and resp > 0:
            ok, t, _ = self.deliver(t + rtt_extra, dst, src, dport, sp, "UDP", 0, resp, cid, label, meta)
        return dict(ok=ok, reason="ok", end=t)
    def icmp_echo(self, t0, src, dst, payload=56, count=1, interval=1.0, label="normal", meta=None) -> dict:
        cid = self.new_cid("ICMP", self.dev[src].ip, 0, self.dev[dst].ip, 0); t = t0; got = 0
        for _ in range(count):
            ok, ta, why = self.deliver(t, src, dst, 0, 0, "ICMP", 0, payload, cid, label, meta)
            if ok:
                ok2, _, _ = self.deliver(ta + 0.00002, dst, src, 0, 0, "ICMP", 0, payload, cid, label, meta); got += int(ok2)
            t += interval
        return dict(ok=got > 0, replies=got, end=t)
    def dns_query(self, t0, client, server="dns", label="normal"):
        return self.udp_exchange(t0, client, server, 53, int(self.rng.integers(28, 70)), int(self.rng.integers(60, 400)), label, meta={"app": "dns"})
    def http_request(self, t0, client, server="web", req=400, resp=5000, label="normal", close="fin", port=80):
        return self.tcp_session(t0, client, server, port, [(req, resp, 0.0)], close, label, meta={"app": "http"})

    def probe(self, src: str, dst: str, proto: str = "TCP", dport: int = 80) -> dict:
        """Mitigation verification: attempt communication now and report whether it works. Packets are captured (label 'probe')."""
        t = self.clock + 1e-3
        if proto == "ICMP": r = self.icmp_echo(t, src, dst, 32, 1, label="probe"); return dict(reachable=r["ok"], reason="ok" if r["ok"] else "no-reply")
        if proto == "UDP": r = self.udp_exchange(t, src, dst, dport, 40, 80, "probe"); return dict(reachable=bool(r["ok"]), reason=r["reason"])
        r = self.tcp_session(t, src, dst, dport, [(100, 100, 0.0)], "fin", "probe"); return dict(reachable=r["established"], reason=r["reason"])

    # ---- visualization ----------------------------------------------------------------
    def topology_figure(self, highlight_recent: int = 0):
        import plotly.graph_objects as go
        col = dict(client="#3b82f6", server="#10b981", attacker="#ef4444", router="#f59e0b", switch="#8b5cf6", monitor="#14b8a6", host="#64748b")
        ex, ey = [], []
        for u, v in self.G.edges:
            if self._active(u) and self._active(v): ex += [self.G.nodes[u]["x"], self.G.nodes[v]["x"], None]; ey += [self.G.nodes[u]["y"], self.G.nodes[v]["y"], None]
        dx, dy = [], []
        for u, v in self.G.edges:
            if not (self._active(u) and self._active(v)): dx += [self.G.nodes[u]["x"], self.G.nodes[v]["x"], None]; dy += [self.G.nodes[u]["y"], self.G.nodes[v]["y"], None]
        fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=1.5, color="#94a3b8"), hoverinfo="none", name="active link"))
        if dx: fig.add_trace(go.Scatter(x=dx, y=dy, mode="lines", line=dict(width=1.5, color="#ef4444", dash="dot"), hoverinfo="none", name="cut link"))
        for dt in col:
            ids = [i for i, d in self.dev.items() if d.dtype == dt]
            if not ids: continue
            fig.add_trace(go.Scatter(x=[self.G.nodes[i]["x"] for i in ids], y=[self.G.nodes[i]["y"] for i in ids], mode="markers+text",
                text=[self.dev[i].name for i in ids], textposition="bottom center",
                hovertext=[f"{self.dev[i].name}<br>{self.dev[i].ip}<br>{self.dev[i].segment}<br>status={self.dev[i].security_status}" for i in ids],
                marker=dict(size=[26 if self.dev[i].dtype == "router" else 18 for i in ids], color=col[dt],
                            line=dict(width=[4 if self.dev[i].security_status != "normal" else 1 for i in ids],
                                      color=["#dc2626" if self.dev[i].security_status != "normal" else "#0f172a" for i in ids])), name=dt))
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10), xaxis=dict(visible=False), yaxis=dict(visible=False),
                          title="Simulated network topology (red outline = isolated/blocked; dotted = cut link)", legend=dict(orientation="h"))
        return fig

# ------------------------------------------------------------------ traffic + attack scenarios
@dataclass
class Scenario:
    name: str; kind: str; params: dict; start: float; end: float; state: dict = field(default_factory=dict); active: bool = True

def _ln(rng, mean, sigma, lo=1, hi=10**7): return int(np.clip(rng.lognormal(np.log(mean), sigma), lo, hi))

class TrafficGenerator:
    """Benign traffic mixes (web, DNS, ICMP, file transfer, SSH, UDP). Rates are sessions/second."""
    def __init__(self, sim: NetworkSimulator, rng: np.random.Generator): self.sim, self.rng = sim, rng
    def web(self, t, c, srv="web", heavy=False):
        r = self.rng; n = int(r.integers(1, 5)); port = 443 if r.random() < 0.4 else 80
        ex = [(_ln(r, 450, 0.4, 80, 3000), _ln(r, 9000 if not heavy else 80000, 1.0, 100, 400000), float(r.exponential(0.25))) for _ in range(n)]
        return self.sim.tcp_session(t, c, srv, port, ex, "fin" if r.random() < 0.9 else "rst", "normal", meta={"app": "http"})
    def file_transfer(self, t, c, size=None):
        r = self.rng; up = r.random() < 0.3; size = size or _ln(r, 400_000, 0.8, 20_000, 1_500_000)
        ex = [(60, 50, 0.01), (30, 40, 0.05)] + ([(size, 60, 0.1)] if up else [(40, size, 0.1)])
        return self.sim.tcp_session(t, c, "ftp", 21, ex, "fin", "normal", meta={"app": "ftp-transfer"})
    def ssh(self, t, c):
        r = self.rng; ex = [(21, 21, 0.0), (_ln(r, 900, 0.2), _ln(r, 900, 0.2), 0.01)] + [(int(r.integers(36, 90)), int(r.integers(36, 400)), float(r.exponential(0.6))) for _ in range(int(r.integers(8, 40)))]
        return self.sim.tcp_session(t, c, "ssh", 22, ex, "fin", "normal", meta={"app": "ssh"})
    def ping(self, t, c, dst=None):
        dst = dst or str(self.rng.choice(["web", "dns", "ftp", "ssh"])); return self.sim.icmp_echo(t, c, dst, 56, int(self.rng.integers(1, 5)), 1.0)
    def udp(self, t, c): return self.sim.udp_exchange(t, c, "dns", 53, int(self.rng.integers(30, 80)), int(self.rng.integers(80, 300)), meta={"app": "dns"})
    def session(self, t, c):
        u = self.rng.random()
        if u < 0.45: return self.web(t, c)
        if u < 0.65: return self.sim.dns_query(t, c)
        if u < 0.72: return self.ping(t, c)
        if u < 0.82: return self.file_transfer(t, c)
        if u < 0.90: return self.ssh(t, c)
        return self.udp(t, c)
    def emit_normal(self, sc: Scenario, t0, t1):
        clients = sc.params.get("clients") or [f"c{i}" for i in range(1, 7)]
        for t in sorted(t0 + self.rng.random(self.rng.poisson(sc.params.get("rate", 3.0) * (t1 - t0))) * (t1 - t0)):
            self.session(float(t), str(self.rng.choice(clients)))

class AttackSimulator:
    """Controlled attack-like traffic inside the simulator. Flow features are DERIVED from the packets it generates."""
    def __init__(self, sim: NetworkSimulator, rng: np.random.Generator, tg: TrafficGenerator): self.sim, self.rng, self.tg = sim, rng, tg
    def _rate(self, sc, t):  # intensity scaling + gradual ramp
        ramp = sc.params.get("ramp", 0.0); f = min(1.0, (t - sc.start) / ramp) if ramp > 0 else 1.0
        return sc.params.get("rate", 10.0) * sc.params.get("intensity", 1.0) * max(f, 0.02)
    def _times(self, sc, t0, t1):
        lam = self._rate(sc, (t0 + t1) / 2) * (t1 - t0); n = self.rng.poisson(lam)
        return sorted(float(x) for x in t0 + self.rng.random(n) * (t1 - t0))
    def emit(self, sc: Scenario, t0, t1):
        getattr(self, "_" + sc.kind)(sc, t0, t1)
    def _srcs(self, sc): s = sc.params.get("src", "a1"); return s if isinstance(s, list) else [s]
    def _dos(self, sc, t0, t1):
        dst, r = sc.params.get("dst", "web"), self.rng; srcs = self._srcs(sc)
        for t in self._times(sc, t0, t1):
            s = str(r.choice(srcs))
            if sc.params.get("variant") == "slow":      # slowloris-like: tiny header chunks every ~10 s
                ex = [(5, 0, float(r.uniform(8, 14))) for _ in range(int(r.integers(3, 7)))]
                self.sim.tcp_session(t, s, dst, 80, ex, "none", "DoS", meta={"app": "slow-http"})
            else:
                self.sim.tcp_session(t, s, dst, 80, [(_ln(r, 380, 0.25, 200, 900), int(r.choice([0, 0, 180, 500])), 0.0)], "rst" if r.random() < 0.6 else "none", "DoS", meta={"app": "http-flood"})
    def _ddos(self, sc, t0, t1):
        dst, r = sc.params.get("dst", "web"), self.rng; srcs = self._srcs(sc)
        for t in self._times(sc, t0, t1):
            s = str(r.choice(srcs)); self.sim.tcp_session(t, s, dst, 80, [(int(r.integers(18, 40)), int(r.choice([0, 0, 120])), 0.0)], "rst" if r.random() < 0.5 else "fin", "DDoS", meta={"app": "http-flood"})
    def _portscan(self, sc, t0, t1):
        p = sc.params; r = self.rng; s = self._srcs(sc)[0]; dst = p.get("dst", "web")
        if "order" not in sc.state:
            lo, hi = p.get("port_range", (1, 1024)); sc.state.update(order=list(r.permutation(np.arange(lo, hi + 1))), i=0, t=sc.start)
        t = max(sc.state["t"], t0); gap = p.get("interval", 0.004) / max(p.get("intensity", 1.0), 0.01)
        while t < t1 and sc.state["i"] < len(sc.state["order"]):
            port = int(sc.state["order"][sc.state["i"]]); sc.state["i"] += 1
            self.sim.tcp_session(t, s, dst, port, [], "none", "PortScan", handshake=False, meta={"app": "syn-scan"}); t += float(r.exponential(gap))
        sc.state["t"] = t
        if sc.state["i"] >= len(sc.state["order"]): sc.active = False
    def _ftp_brute(self, sc, t0, t1):
        r = self.rng; dst = sc.params.get("dst", "ftp")
        for t in self._times(sc, t0, t1):
            ex = [(0, 40, 0.01), (14, 33, 0.02), (16, 24, 0.02), (6, 14, 0.01)]
            self.sim.tcp_session(t, self._srcs(sc)[0], dst, 21, ex, "fin", "FTP-Patator", meta={"app": "ftp-login"})
    def _ssh_brute(self, sc, t0, t1):
        r = self.rng; dst = sc.params.get("dst", "ssh")
        for t in self._times(sc, t0, t1):
            ex = [(21, 21, 0.0), (_ln(r, 1000, 0.05, 800, 1200), _ln(r, 1000, 0.05, 800, 1200), 0.02), (48, 60, 0.15), (96, 80, 0.4), (48, 48, 0.2)]
            self.sim.tcp_session(t, self._srcs(sc)[0], dst, 22, ex, "fin", "SSH-Patator", meta={"app": "ssh-login"})
    def _web_attack(self, sc, t0, t1):
        r = self.rng; dst = sc.params.get("dst", "web")
        for t in self._times(sc, t0, t1):
            ex = [(_ln(r, 1800, 0.5, 700, 6000), _ln(r, 350, 0.3, 120, 900), float(r.exponential(0.05))) for _ in range(int(r.integers(2, 6)))]
            self.sim.tcp_session(t, self._srcs(sc)[0], dst, 80, ex, "fin", "WebAttack", meta={"app": "http-suspicious"})
    def _botnet(self, sc, t0, t1):
        r = self.rng; per = sc.params.get("period", 5.0); s = self._srcs(sc)[0]; dst = sc.params.get("dst", "cnc")
        t = sc.state.get("t", sc.start)
        self.sim.dev[s].security_status = "compromised" if self.sim.dev[s].security_status == "normal" else self.sim.dev[s].security_status
        while t < t1:
            if t >= t0:
                ex = [(int(r.normal(120, 6)), int(r.normal(90, 5)), 0.0)]
                if r.random() < 0.1: ex.append((int(r.integers(2000, 9000)), 40, 0.05))   # occasional exfil-like burst
                self.sim.tcp_session(t, s, dst, 8080, ex, "fin", "Bot", meta={"app": "c2-beacon"})
            t += max(0.5, float(r.normal(per, per * 0.1)) / max(sc.params.get("intensity", 1.0), 0.05))
        sc.state["t"] = t
    def _burst(self, sc, t0, t1):   # abnormal connection frequency / legitimate-but-unusual burst (benign-looking sessions)
        s = self._srcs(sc)[0]
        for t in self._times(sc, t0, t1): self.tg.web(t, s, sc.params.get("dst", "web"), heavy=sc.params.get("heavy", False))
    def _unusual(self, sc, t0, t1):  # protocol behaviour absent from the known attack classes
        r = self.rng; s = self._srcs(sc)[0]; var = sc.params.get("variant", "icmp_tunnel")
        if var == "icmp_tunnel":
            t = sc.state.get("t", sc.start); dst = sc.params.get("dst", "cnc")
            while t < t1:
                if t >= t0: self.sim.icmp_echo(t, s, dst, int(r.integers(900, 1300)), 1, label="Unknown")
                t += float(r.normal(0.2, 0.01))
            sc.state["t"] = t
        else:   # xmas-like: FIN|PSH|URG single packets without handshake
            for t in self._times(sc, t0, t1):
                port = int(r.choice([80, 22, 21, 443])); sd, dd = self.sim.dev[s], self.sim.dev[sc.params.get("dst", "web")]
                cid = self.sim.new_cid("TCP", sd.ip, self.sim.eph(), dd.ip, port)
                self.sim.deliver(t, s, dd.id, self.sim.eph(), port, "TCP", FIN | PSH | URG, int(r.integers(0, 20)), cid, "Unknown", {"app": "xmas"})

class ScenarioController:
    """Starts / stops / resets / replays scenarios. Dashboard buttons call these methods directly."""
    KINDS = {"normal": "normal", "dos": "dos", "ddos": "ddos", "portscan": "portscan", "ftp_brute": "ftp_brute", "ssh_brute": "ssh_brute",
             "web_attack": "web_attack", "botnet": "botnet", "burst": "burst", "unusual": "unusual"}
    DEFAULTS = {"normal": dict(rate=3.0), "dos": dict(src="a1", dst="web", rate=120.0), "ddos": dict(src=["a1", "a2", "a3"], dst="web", rate=200.0),
                "portscan": dict(src="a1", dst="web", port_range=(1, 1024), interval=0.004), "ftp_brute": dict(src="a2", dst="ftp", rate=3.0),
                "ssh_brute": dict(src="a2", dst="ssh", rate=2.0), "web_attack": dict(src="a3", dst="web", rate=8.0),
                "botnet": dict(src="c3", dst="cnc", period=5.0), "burst": dict(src="c5", dst="web", rate=30.0),
                "unusual": dict(src="c4", dst="cnc", variant="icmp_tunnel")}
    def __init__(self, sim: NetworkSimulator, seed: int = 42):
        self.sim = sim; self.seed = seed; self.rng = np.random.default_rng(seed + 1)
        self.tg = TrafficGenerator(sim, self.rng); self.atk = AttackSimulator(sim, self.rng, self.tg); self.scenarios: List[Scenario] = []; self.history: List[dict] = []
    def start(self, kind: str, duration: float = 30.0, **params) -> Scenario:
        if kind not in self.KINDS: raise ValueError(f"unknown scenario {kind}")
        p = {**self.DEFAULTS[kind], **params}; sc = Scenario(f"{kind}#{len(self.history)+1}", kind, p, self.sim.clock, self.sim.clock + duration)
        self.scenarios.append(sc); self.history.append(dict(name=sc.name, kind=kind, params=p, start=sc.start, duration=duration)); log.info("start %s %s", sc.name, p); return sc
    def stop(self, name: Optional[str] = None):
        for s in self.scenarios:
            if name is None or s.name == name or s.kind == name: s.active = False
    def tick(self, dt: float = 2.0):
        """Advance the simulated clock by dt, emitting events of every active scenario in [clock, clock+dt)."""
        t0, t1 = self.sim.clock, self.sim.clock + dt
        for s in self.scenarios:
            if not s.active: continue
            a, b = max(t0, s.start), min(t1, s.end)
            if b > a: (self.tg.emit_normal(s, a, b) if s.kind == "normal" else self.atk.emit(s, a, b))
            if s.end <= t1: s.active = False
        self.sim.clock = t1; self.sim.router.expire(t1)
    def run(self, seconds: float, dt: float = 2.0):
        n = int(np.ceil(seconds / dt))
        for _ in range(n): self.tick(dt)
    def reset(self, seed: Optional[int] = None):
        """Reset the whole network (devices, ACLs, clock, capture) and the scenario state with a fresh seed."""
        if seed is not None: self.seed = seed
        self.sim.reset(self.seed); self.rng = np.random.default_rng(self.seed + 1)
        self.tg = TrafficGenerator(self.sim, self.rng); self.atk = AttackSimulator(self.sim, self.rng, self.tg)
        self.scenarios, self.history = [], []
    def replay(self, plan: List[dict], seed: Optional[int] = None, extra: float = 10.0):
        """Deterministically re-run a recorded plan [{kind, params, start, duration}] on a fresh network."""
        self.reset(seed)
        for it in sorted(plan, key=lambda x: x["start"]):
            while self.sim.clock < it["start"]: self.tick(min(2.0, it["start"] - self.sim.clock))
            self.start(it["kind"], it["duration"], **it["params"])
        self.run(max(it["start"] + it["duration"] for it in plan) - self.sim.clock + extra)

# ------------------------------------------------------------------ flow extraction
class _Flow:
    __slots__ = ("key", "src", "dst", "sp", "dp", "proto", "ts", "ln", "fw", "fl", "labels", "devs", "term", "fin_f", "fin_b", "cid", "last")
class FlowFeatureExtractor:
    """CICFlowMeter-style bidirectional flow assembly. Forward = direction of the first packet of the flow.
    Flow ends on RST, on FIN from both sides (after a short linger), on timeout, or on forced flush."""
    LINGER = 2.0
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config(); self.active: Dict[tuple, _Flow] = {}; self.done: List[dict] = []; self._n = itertools.count(1)
    @staticmethod
    def _key(p: Packet):
        a, b = (p.src_ip, p.sport), (p.dst_ip, p.dport); return (p.proto,) + ((a, b) if a <= b else (b, a))
    def ingest(self, packets: List[Packet]):
        for p in packets:
            k = self._key(p); f = self.active.get(k)
            if f is not None:
                new_conn = p.proto == "TCP" and (p.flags & SYN) and not (p.flags & ACK) and f.term
                if p.ts - f.last > self.cfg.flow_timeout_s or new_conn: self._finish(k)
                f = self.active.get(k)
            if f is None:
                f = _Flow(); f.key, f.src, f.dst, f.sp, f.dp, f.proto = k, p.src_ip, p.dst_ip, p.sport, p.dport, p.proto
                f.ts, f.ln, f.fw, f.fl, f.labels, f.devs = [], [], [], [], [], (p.src_dev, p.dst_dev)
                f.term, f.fin_f, f.fin_b, f.cid, f.last = False, False, False, p.conn_id, p.ts; self.active[k] = f
            fwd = p.src_ip == f.src and p.sport == f.sp
            f.ts.append(p.ts); f.ln.append(p.payload); f.fw.append(fwd); f.fl.append(p.flags); f.labels.append(p.label); f.last = p.ts
            if p.proto == "TCP":
                if p.flags & RST: f.term = True
                if p.flags & FIN: (setattr(f, "fin_f", True) if fwd else setattr(f, "fin_b", True)); f.term = f.term or (f.fin_f and f.fin_b)
    def _finish(self, k):
        f = self.active.pop(k); self.done.append(self._features(f))
    def pop_finished(self, now: float, force: bool = False) -> List[dict]:
        for k, f in list(self.active.items()):
            if force or (f.term and now - f.last >= self.LINGER) or now - f.last > self.cfg.flow_timeout_s: self._finish(k)
        out, self.done = self.done, []; return out
    @staticmethod
    def _stats(a: np.ndarray):
        if a.size == 0: return 0.0, 0.0, 0.0, 0.0
        return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0, float(a.max()), float(a.min())
    def _features(self, f: _Flow) -> dict:
        ts = np.asarray(f.ts) * 1e6; ln = np.asarray(f.ln, float); fw = np.asarray(f.fw, bool); fl = np.asarray(f.fl, int)
        o = np.argsort(ts, kind="stable"); ts, ln, fw, fl = ts[o], ln[o], fw[o], fl[o]
        dur = float(ts[-1] - ts[0]); n = len(ts); nf, nb = int(fw.sum()), int((~fw).sum())
        lf, lb = ln[fw], ln[~fw]; iat = np.diff(ts); fi = np.diff(ts[fw]); bi = np.diff(ts[~fw])
        m, s, mx, mn = self._stats(iat); fm, fs, fmx, fmn = self._stats(fi); bm, bs, bmx, bmn = self._stats(bi)
        lfm, lfs, lfx, lfn = self._stats(lf); lbm, lbs, lbx, lbn = self._stats(lb); pm, ps, px, pn = self._stats(ln)
        sec = dur / 1e6
        act, idl, start = [], [], ts[0]
        for i in range(1, n):
            gap = ts[i] - ts[i - 1]
            if gap > self.cfg.activity_timeout_us:
                if ts[i - 1] - start > 0: act.append(ts[i - 1] - start)
                idl.append(gap); start = ts[i]
        if ts[-1] - start > 0: act.append(ts[-1] - start)
        am, as_, ax, an = self._stats(np.asarray(act)); im, is_, ix, in_ = self._stats(np.asarray(idl))
        cnt = lambda bit: int(((fl & bit) > 0).sum())
        row = {
         "Flow Duration": dur, "Total Fwd Packets": nf, "Total Backward Packets": nb, "Total Length of Fwd Packets": float(lf.sum()),
         "Total Length of Bwd Packets": float(lb.sum()), "Fwd Packet Length Max": lfx, "Fwd Packet Length Min": lfn, "Fwd Packet Length Mean": lfm,
         "Fwd Packet Length Std": lfs, "Bwd Packet Length Max": lbx, "Bwd Packet Length Min": lbn, "Bwd Packet Length Mean": lbm, "Bwd Packet Length Std": lbs,
         "Flow Bytes/s": float(ln.sum() / sec) if sec > 0 else 0.0, "Flow Packets/s": n / sec if sec > 0 else 0.0,
         "Flow IAT Mean": m, "Flow IAT Std": s, "Flow IAT Max": mx, "Flow IAT Min": mn,
         "Fwd IAT Total": float(fi.sum()), "Fwd IAT Mean": fm, "Fwd IAT Std": fs, "Fwd IAT Max": fmx, "Fwd IAT Min": fmn,
         "Bwd IAT Total": float(bi.sum()), "Bwd IAT Mean": bm, "Bwd IAT Std": bs, "Bwd IAT Max": bmx, "Bwd IAT Min": bmn,
         "Fwd PSH Flags": int(((fl[fw] & PSH) > 0).sum()), "Fwd Packets/s": nf / sec if sec > 0 else 0.0, "Bwd Packets/s": nb / sec if sec > 0 else 0.0,
         "Min Packet Length": pn, "Max Packet Length": px, "Packet Length Mean": pm, "Packet Length Std": ps, "Packet Length Variance": ps ** 2,
         "FIN Flag Count": cnt(FIN), "SYN Flag Count": cnt(SYN), "RST Flag Count": cnt(RST), "PSH Flag Count": cnt(PSH), "ACK Flag Count": cnt(ACK), "URG Flag Count": cnt(URG),
         "Down/Up Ratio": float(nb // nf) if nf else 0.0, "Average Packet Size": float(ln.mean()), "Avg Fwd Segment Size": lfm, "Avg Bwd Segment Size": lbm,
         "Active Mean": am, "Active Std": as_, "Active Max": ax, "Active Min": an, "Idle Mean": im, "Idle Std": is_, "Idle Max": ix, "Idle Min": in_}
        lab = max(set(f.labels), key=f.labels.count)
        row.update(flow_id=f"F{next(self._n)}", src_ip=f.src, dst_ip=f.dst, src_port=f.sp, dst_port=f.dp, proto=f.proto,
                   start_ts=float(ts[0] / 1e6), end_ts=float(ts[-1] / 1e6), true_label=lab, conn_id=f.cid)
        return row
