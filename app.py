
import json
import pickle
import math
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. PAGE CONFIGURATION - KEEP THIS AS THE ONLY CALL
# ============================================================

st.set_page_config(
    page_title="GSA-DT | Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 2. CUSTOM UI DESIGN
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #07111f 0%,
        #101c31 48%,
        #0b1728 100%
    );
    color: #edf4ff;
}

[data-testid="stHeader"] {
    background: rgba(7,17,31,0.85);
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 30px;
    border: 1px solid rgba(120,170,255,.25);
    border-radius: 22px;
    background: linear-gradient(
        120deg,
        rgba(23,55,91,.95),
        rgba(19,34,58,.92)
    );
    box-shadow: 0 14px 40px rgba(0,0,0,.22);
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero p {
    color: #c6d8f2;
    font-size: 1rem;
}

.section-card {
    border: 1px solid rgba(137,176,230,.20);
    border-radius: 16px;
    padding: 20px;
    background: rgba(17,34,57,.78);
    margin: 12px 0 18px 0;
}

div[data-testid="stMetric"] {
    background: rgba(22,44,72,.82);
    border: 1px solid rgba(137,176,230,.22);
    padding: 16px;
    border-radius: 15px;
}

div[data-testid="stMetricLabel"] {
    color: #b8cce8;
}

div[data-testid="stMetricValue"] {
    color: white;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 10px;
    min-height: 2.8rem;
    font-weight: 700;
    border: 1px solid #4e9bff;
    background: linear-gradient(
        90deg,
        #1769c2,
        #2789e8
    );
    color: white;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: #9ccaff;
    color: white;
}

[data-testid="stFileUploader"] {
    background: rgba(17,34,57,.78);
    border: 1px dashed #5b91cb;
    border-radius: 16px;
    padding: 12px;
}

h2, h3 {
    color: #eaf3ff !important;
}

.footer {
    text-align: center;
    color: #9fb5d2;
    font-size: .95rem;
    padding: 25px 10px 10px;
    margin-top: 40px;
    border-top: 1px solid rgba(137,176,230,.2);
}

.footer strong {
    color: #78baff;
    letter-spacing: .5px;
    font-size: 1rem;
}

.small-note {
    color: #a9bfdc;
    font-size: .9rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. ARTIFACT PATH
# ============================================================

BASE = Path(__file__).parent
ART = BASE / "artifacts"


# ============================================================
# 4. LOAD MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    with open(ART / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    def load_pickle(name):
        with open(ART / name, "rb") as f:
            return pickle.load(f)

    flow_scaler = load_pickle("flow_scaler.pkl")
    node_scaler = load_pickle("node_scaler.pkl")
    graph = load_pickle("graph_state.pkl")
    label_encoder = load_pickle("label_encoder.pkl")
    tree = load_pickle("gsa_dt_decision_tree.pkl")

    state = torch.load(
        ART / "gsa_dt_gcn_attention.pt",
        map_location="cpu",
        weights_only=True
    )

    return (
        cfg,
        flow_scaler,
        node_scaler,
        graph,
        label_encoder,
        tree,
        state
    )


# ============================================================
# 5. SELF ATTENTION
# ============================================================

class SelfAttention(nn.Module):

    def __init__(self, d, e):
        super().__init__()

        self.e = e

        self.embed = nn.Parameter(
            torch.randn(d, e) * 0.1
        )

        self.Wq = nn.Linear(e, e, bias=False)
        self.Wk = nn.Linear(e, e, bias=False)
        self.Wv = nn.Linear(e, e, bias=False)

        self.proj = nn.Linear(e, 1)

    def forward(self, h):

        x = h.unsqueeze(-1) * self.embed

        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        attention = torch.softmax(
            q @ k.transpose(1, 2) / math.sqrt(self.e),
            dim=-1
        )

        return self.proj(
            attention @ v
        ).squeeze(-1)


# ============================================================
# 6. GSA-DT GCN MODEL
# ============================================================

class GsaNet(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden,
        attention_dim,
        n_classes,
        dropout,
        slope
    ):
        super().__init__()

        self.dropout = dropout

        self.lin = nn.ModuleList([
            nn.Linear(input_dim, hidden),
            nn.Linear(hidden, hidden)
        ])

        self.sa = SelfAttention(
            hidden,
            attention_dim
        )

        self.head = nn.Linear(
            2 * hidden,
            n_classes
        )

        self.slope = slope

    def embed(self, x, adj):

        h = x

        for layer in self.lin:

            h = F.dropout(
                h,
                self.dropout,
                self.training
            )

            h = F.leaky_relu(
                torch.sparse.mm(adj, layer(h)),
                negative_slope=self.slope
            )

        return self.sa(h)


# ============================================================
# 7. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    cfg, fs, ns, graph, le, tree, weights = load_artifacts()

    model = GsaNet(
        graph["node_features"].shape[1],
        cfg["hidden"],
        cfg["attention_dim"],
        len(cfg["classes"]),
        cfg["dropout"],
        cfg["leaky_slope"]
    )

    model.load_state_dict(weights)
    model.eval()

    return model


# ============================================================
# 8. HELPER FUNCTIONS
# ============================================================

def _numeric(df, col, default=0.0):

    if col not in df:
        return np.full(
            len(df),
            default,
            dtype=float
        )

    return pd.to_numeric(
        df[col],
        errors="coerce"
    ).fillna(default).to_numpy(dtype=float)


def _port(df, col):

    if col not in df:
        return np.zeros(len(df), dtype=float)

    s = df[col].astype(str).str.strip()

    out = pd.to_numeric(
        s,
        errors="coerce"
    ).astype(float)

    mask = s.str.lower().str.startswith("0x")

    out.loc[mask] = s.loc[mask].map(
        lambda v: int(v, 16)
    )

    return out.fillna(0).to_numpy(dtype=float)


# ============================================================
# 9. PCAP AND PCAPNG PROCESSING
# ============================================================

def pcap_to_dataframe(upload):

    """
    Convert PCAP/PCAPNG packets into approximate
    bidirectional network flows.

    Supported protocols:
    TCP, UDP and ICMP.

    PCAP does not directly contain the Argus State field.
    State is approximated as CON for TCP and INT for
    UDP/ICMP.
    """

    try:

        from scapy.utils import rdpcap
        from scapy.layers.inet import IP, TCP, UDP, ICMP
        from scapy.layers.inet6 import IPv6

    except ImportError:

        raise RuntimeError(
            "PCAP support requires Scapy. "
            "Install it using: pip install scapy"
        )

    try:

        packets = rdpcap(
            BytesIO(upload.getvalue())
        )

    except Exception as exc:

        raise ValueError(
            f"Could not read this PCAP/PCAPNG file: {exc}"
        )

    flows = {}

    for pkt in packets:

        if IP in pkt:

            ip = pkt[IP]
            src = str(ip.src)
            dst = str(ip.dst)

        elif IPv6 in pkt:

            ip = pkt[IPv6]
            src = str(ip.src)
            dst = str(ip.dst)

        else:
            continue

        if TCP in pkt:

            layer = pkt[TCP]
            proto = "tcp"

        elif UDP in pkt:

            layer = pkt[UDP]
            proto = "udp"

        elif ICMP in pkt:

            layer = pkt[ICMP]
            proto = "icmp"

        else:
            continue

        sport = int(getattr(layer, "sport", 0))
        dport = int(getattr(layer, "dport", 0))

        a = (src, sport)
        b = (dst, dport)

        endpoints = tuple(sorted((a, b)))
        key = (proto, endpoints)

        ts = float(pkt.time)
        size = int(len(pkt))

        if key not in flows:

            flows[key] = {
                "SrcAddr": src,
                "DstAddr": dst,
                "Sport": sport,
                "Dport": dport,
                "Proto": proto,
                "State": "CON" if proto == "tcp" else "INT",
                "first": ts,
                "last": ts,
                "TotPkts": 0,
                "TotBytes": 0,
                "SrcBytes": 0
            }

        row = flows[key]

        row["TotPkts"] += 1
        row["TotBytes"] += size

        if (
            src == row["SrcAddr"]
            and sport == row["Sport"]
        ):
            row["SrcBytes"] += size

        row["first"] = min(
            row["first"],
            ts
        )

        row["last"] = max(
            row["last"],
            ts
        )

    records = []

    for row in flows.values():

        records.append({

            "SrcAddr": row["SrcAddr"],
            "DstAddr": row["DstAddr"],

            "Dur": max(
                0.0,
                row["last"] - row["first"]
            ),

            "TotPkts": row["TotPkts"],
            "TotBytes": row["TotBytes"],
            "SrcBytes": row["SrcBytes"],

            "Sport": row["Sport"],
            "Dport": row["Dport"],

            "Proto": row["Proto"],
            "State": row["State"]

        })

    if not records:

        raise ValueError(
            "No supported IPv4/IPv6 TCP, UDP or ICMP "
            "packets were found in this capture."
        )

    return pd.DataFrame(records)


# ============================================================
# 10. PREPARE NETWORK FLOWS
# ============================================================

def prepare_flows(upload):

    if isinstance(upload, pd.DataFrame):

        raw = upload.copy()

    else:

        raw = pd.read_csv(
            upload,
            sep=None,
            engine="python",
            on_bad_lines="skip"
        )

    raw.columns = [
        str(c).strip()
        for c in raw.columns
    ]

    required = {
        "SrcAddr",
        "DstAddr"
    }

    if not required.issubset(raw.columns):

        raise ValueError(
            "Input file must contain SrcAddr and DstAddr "
            "columns (standard Argus/BINETFLOW format)."
        )

    total = _numeric(
        raw,
        "TotBytes"
    )

    src_bytes = _numeric(
        raw,
        "SrcBytes",
        -1
    )

    src_bytes = np.where(
        src_bytes < 0,
        total / 2,
        src_bytes
    )

    proto = (
        raw["Proto"].astype(str).str.strip().str.lower()
        if "Proto" in raw
        else pd.Series(["other"] * len(raw))
    )

    state = (
        raw["State"].astype(str).str.strip().str.lower()
        if "State" in raw
        else pd.Series(["none"] * len(raw))
    )

    d = pd.DataFrame({

        "src": raw["SrcAddr"].astype(str).str.strip(),
        "dst": raw["DstAddr"].astype(str).str.strip(),

        "dur": _numeric(raw, "Dur"),
        "tot_pkts": _numeric(raw, "TotPkts"),
        "tot_bytes": total,

        "out_bytes": src_bytes,

        "in_bytes": np.clip(
            total - src_bytes,
            0,
            None
        ),

        "sport": _port(raw, "Sport"),
        "dport": _port(raw, "Dport")

    })

    d["proto_tcp"] = (
        proto == "tcp"
    ).astype(float)

    d["proto_udp"] = (
        proto == "udp"
    ).astype(float)

    d["proto_arp"] = (
        proto == "arp"
    ).astype(float)

    d["proto_icmp"] = (
        proto == "icmp"
    ).astype(float)

    states = [
        "con",
        "srpa_spa",
        "fspa_fspa",
        "s_",
        "fspa_fsrpa",
        "int",
        "s_ra",
        "spa_fsrpa"
    ]

    for s in states:

        d["state_" + s] = (
            state == s
        ).astype(float)

    features = [
        "dur",
        "tot_pkts",
        "tot_bytes",
        "out_bytes",
        "in_bytes",
        "sport",
        "dport",
        "proto_tcp",
        "proto_udp",
        "proto_arp",
        "proto_icmp"
    ] + [
        "state_" + s
        for s in states
    ]

    d = d.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna(
        subset=["src", "dst"] + features
    )

    d = d[
        (d.src != "") &
        (d.dst != "")
    ].drop_duplicates(
        subset=["src", "dst"] + features
    ).reset_index(drop=True)

    if d.empty:

        raise ValueError(
            "No usable flows found in the uploaded file."
        )

    return d, features


# ============================================================
# 11. GSA-DT PREDICTION
# ============================================================

def predict(uploaded):

    cfg, flow_scaler, node_scaler, graph, le, tree, _ = load_artifacts()

    model = load_model()

    flows, feat_cols = prepare_flows(uploaded)

    expected = cfg["flow_features"]

    if feat_cols != expected:

        raise ValueError(
            f"Feature mismatch. App features: {feat_cols}; "
            f"model expects: {expected}"
        )

    # Flow feature transformation

    xr = flows[expected].to_numpy(
        dtype=np.float32
    )

    xr = (
        np.sign(xr) *
        np.log1p(np.abs(xr))
    )

    xf = flow_scaler.transform(
        xr
    ).astype(np.float32)

    # Map IP addresses to graph nodes

    old_ips = [
        str(v)
        for v in graph["node_ips"]
    ]

    ip_to_id = {
        ip: i
        for i, ip in enumerate(old_ips)
    }

    n_old = len(old_ips)

    new_ips = sorted(
        set(flows.src).union(flows.dst).difference(ip_to_id)
    )

    for ip in new_ips:

        ip_to_id[ip] = len(ip_to_id)

    n_nodes = len(ip_to_id)

    src = flows.src.map(
        ip_to_id
    ).to_numpy(dtype=np.int64)

    dst = flows.dst.map(
        ip_to_id
    ).to_numpy(dtype=np.int64)

    # Recover node statistics

    node_raw = node_scaler.inverse_transform(
        graph["node_features"]
    ).astype(np.float64)

    old_out_n = np.expm1(
        np.maximum(node_raw[:, 38], 0)
    )

    old_in_n = np.expm1(
        np.maximum(node_raw[:, 39], 0)
    )

    old_deg = np.expm1(
        np.maximum(node_raw[:, 40], 0)
    )

    out_sum = (
        node_raw[:, :19] *
        old_out_n[:, None]
    )

    in_sum = (
        node_raw[:, 19:38] *
        old_in_n[:, None]
    )

    out_n = old_out_n.copy()
    in_n = old_in_n.copy()

    if n_nodes > n_old:

        pad = n_nodes - n_old

        out_sum = np.vstack([
            out_sum,
            np.zeros((pad, 19))
        ])

        in_sum = np.vstack([
            in_sum,
            np.zeros((pad, 19))
        ])

        out_n = np.r_[
            out_n,
            np.zeros(pad)
        ]

        in_n = np.r_[
            in_n,
            np.zeros(pad)
        ]

        old_deg = np.r_[
            old_deg,
            np.zeros(pad)
        ]

    # Update node statistics

    for node, row in zip(src, xf):

        out_sum[node] += row
        out_n[node] += 1

    for node, row in zip(dst, xf):

        in_sum[node] += row
        in_n[node] += 1

    out_mean = np.divide(
        out_sum,
        out_n[:, None],
        out=np.zeros_like(out_sum),
        where=out_n[:, None] > 0
    )

    in_mean = np.divide(
        in_sum,
        in_n[:, None],
        out=np.zeros_like(in_sum),
        where=in_n[:, None] > 0
    )

    # Extend graph

    A = graph["adjacency"].tocsr().astype(np.float32)

    if n_nodes > n_old:

        A = sp.block_diag(
            (
                A,
                sp.csr_matrix(
                    (n_nodes - n_old, n_nodes - n_old)
                )
            ),
            format="csr"
        )

    add = sp.coo_matrix(
        (
            np.ones(len(src), dtype=np.float32),
            (src, dst)
        ),
        shape=(n_nodes, n_nodes)
    ).tocsr()

    A = (
        (A + add + add.T) > 0
    ).astype(np.float32).tocsr()

    A.setdiag(0)
    A.eliminate_zeros()

    degree = np.asarray(
        A.sum(axis=1)
    ).ravel().astype(np.float64)

    node_raw_new = np.column_stack([

        out_mean,
        in_mean,

        np.log1p(out_n),
        np.log1p(in_n),
        np.log1p(degree)

    ]).astype(np.float32)

    x = node_scaler.transform(
        node_raw_new
    ).astype(np.float32)

    # Normalize adjacency

    Bhat = A + sp.eye(
        n_nodes,
        dtype=np.float32,
        format="csr"
    )

    deg_hat = np.asarray(
        Bhat.sum(axis=1)
    ).ravel()

    inv = np.power(
        np.maximum(deg_hat, 1),
        -0.5
    )

    S = (
        sp.diags(inv) @
        Bhat @
        sp.diags(inv)
    ).tocoo()

    indices = torch.from_numpy(
        np.vstack([
            S.row,
            S.col
        ]).astype(np.int64)
    )

    values = torch.from_numpy(
        S.data.astype(np.float32)
    )

    adj_t = torch.sparse_coo_tensor(
        indices,
        values,
        S.shape
    ).coalesce()

    x_t = torch.from_numpy(x)

    # GCN + Self Attention

    with torch.no_grad():

        O = model.embed(
            x_t,
            adj_t
        )

        units = torch.cat([

            O[torch.from_numpy(src)],
            O[torch.from_numpy(dst)]

        ], dim=1).numpy()

    # Decision Tree prediction

    pred_ids = tree.predict(units)

    labels = le.inverse_transform(
        pred_ids
    )

    result = flows[
        [
            "src",
            "dst",
            "dur",
            "tot_pkts",
            "tot_bytes",
            "sport",
            "dport"
        ]
    ].copy()

    result["prediction"] = labels

    result["is_malicious"] = (
        result.prediction.ne(
            cfg["normal_class"]
        )
    )

    return result, len(new_ips)


# ============================================================
# MAIN USER INTERFACE
# ============================================================

st.title("🛡️ GSA-DT Network Intrusion Detection")

st.write(
    "Intelligent network traffic analysis using "
    "Graph Convolutional Networks, Self-Attention "
    "and a Gini-based Decision Tree."
)

st.divider()

st.subheader("🔍 Analyse Your Network Traffic")

st.write(
    "Upload a supported traffic file to identify "
    "normal and potentially malicious network flows."
)

st.info(
    "Supported formats: CSV | BINETFLOW | TXT | PCAP | PCAPNG"
)


# ============================================================
# 13. INPUT INFORMATION
# ============================================================

with st.expander(
    "ℹ️ Input requirements and PCAP notes"
):

    st.write(
        "NetFlow files should contain SrcAddr and DstAddr. "
        "For best compatibility include Dur, TotPkts, "
        "TotBytes, SrcBytes, Sport, Dport, Proto and State."
    )

    st.write(
        "PCAP/PCAPNG files are converted into approximate "
        "bidirectional TCP/UDP/ICMP flows. PCAP-derived "
        "State is approximated because packet captures "
        "do not directly contain the Argus State field. "
        "Treat PCAP predictions as experimental."
    )


# ============================================================
# 14. FILE UPLOAD
# ============================================================

upload = st.file_uploader(
    "Choose a network traffic file",
    type=None,
    help=(
        "Accepted extensions: .csv, .binetflow, "
        ".txt, .pcap, .pcapng"
    )
)


if upload is not None:

    ext = (
        upload.name.rsplit(".", 1)[-1].lower()
        if "." in upload.name
        else ""
    )

    supported = {
        "csv",
        "binetflow",
        "txt",
        "pcap",
        "pcapng"
    }

    if ext not in supported:

        st.error(
            "Unsupported file format. "
            "Please upload only: .csv, .binetflow, "
            ".txt, .pcap or .pcapng"
        )

        st.stop()

    st.success(
        f"File ready: {upload.name}"
    )

    st.caption(
        f"File size: {upload.size / 1024:.1f} KB "
        f"· Format: .{ext}"
    )

    # ========================================================
    # 15. ANALYSE BUTTON
    # ========================================================

    if st.button(
        "🚀 Analyse traffic",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Processing traffic and generating predictions..."
            ):

                if ext in {"pcap", "pcapng"}:

                    flows_df = pcap_to_dataframe(
                        upload
                    )

                    st.info(
                        f"Extracted {len(flows_df):,} "
                        "approximate bidirectional flows."
                    )

                    result, unseen = predict(
                        flows_df
                    )

                else:

                    result, unseen = predict(
                        upload
                    )

            # =================================================
            # 16. DETECTION OVERVIEW
            # =================================================

            total = len(result)

            malicious = int(
                result.is_malicious.sum()
            )

            normal = total - malicious

            rate = (
                malicious / total * 100
                if total
                else 0
            )

            st.markdown(
                "### 📊 Detection overview"
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Total flows analysed",
                f"{total:,}"
            )

            c2.metric(
                "Malicious flows",
                f"{malicious:,}"
            )

            c3.metric(
                "Normal flows",
                f"{normal:,}"
            )

            c4.metric(
                "Malicious flow rate",
                f"{rate:.1f}%"
            )

            # =================================================
            # 17. ATTACK CLASS CHART
            # =================================================

            st.markdown(
                "### 🧠 Predicted attack classes"
            )

            counts = (
                result.prediction
                .value_counts()
                .rename_axis("Attack class")
                .to_frame("Number of flows")
            )

            st.bar_chart(counts)

            # =================================================
            # 18. DETAILED RESULTS
            # =================================================

            st.markdown(
                "### 🔎 Detailed prediction results"
            )

            filter_choice = st.selectbox(
                "Filter results",
                [
                    "All flows",
                    "Malicious only",
                    "Normal only"
                ]
            )

            if filter_choice == "Malicious only":

                display_result = result[
                    result.is_malicious
                ]

            elif filter_choice == "Normal only":

                display_result = result[
                    ~result.is_malicious
                ]

            else:

                display_result = result

            st.dataframe(
                display_result,
                use_container_width=True,
                hide_index=True
            )

            # =================================================
            # 19. DOWNLOAD RESULTS
            # =================================================

            st.download_button(

                "⬇️ Download prediction results (CSV)",

                result.to_csv(
                    index=False
                ).encode("utf-8"),

                "gsa_dt_predictions.csv",

                "text/csv",

                use_container_width=True

            )

            if ext in {"pcap", "pcapng"}:

                st.warning(
                    "PCAP-derived features are approximate. "
                    "In particular, the Argus State field is "
                    "approximated. Validate these results "
                    "against labeled captures before drawing "
                    "security conclusions."
                )

            st.caption(
                f"{unseen:,} IP addresses were not present "
                "in the saved training graph and were added "
                "for this inference run."
            )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    "<div style='text-align:center; color:#9fb5d2;'>"
    "GSA-DT Network Intrusion Detection System"
    "</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div style='text-align:center; color:#78baff; "
    "font-size:18px; font-weight:bold; padding:10px;'>"
    "Developed by Pavan Kumar Choudari"
    "</div>",
    unsafe_allow_html=True
)