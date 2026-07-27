import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.graph_objects as go
import string

# Dynamically add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src', 'field_segmentation'))
sys.path.append(src_dir)

from segmentation_engine import ProtocolSegmentationEngine

st.set_page_config(
    page_title="Protocol Field Segmentation Engine", 
    page_icon="⚡", 
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E222B;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #4F8BF9;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Network Protocol Segmentation Engine")
st.markdown("Unsupervised structural boundary detection integrating **N-Gram Tokenization** and **Mutual Information**.")

# --- SIDEBAR ---
st.sidebar.header("Protocol Configuration")
protocol_choice = st.sidebar.selectbox("Select Target Protocol", ("ARP", "Modbus TCP", "DNP3"))

st.sidebar.divider()
st.sidebar.header("AI Upgrade Parameters")
ngram_size = st.sidebar.slider(
    "N-Gram Tokenization Size (Bytes)", 
    min_value=1, max_value=8, value=2, step=1,
    help="Forces structural boundaries in static, zero-variance spans. (e.g., 2 forces 16-bit word alignment)."
)

import plotly.graph_objects as go
import plotly.express as px

def plot_3tier_comparison(bounds_baseline, bounds_mi, bounds_dbscan, total_bytes):
    """
    Renders a multi-bar horizontal stacked plot with distinct qualitative colors
    per field unit (matching the Ground Truth visual style).
    """
    fig = go.Figure()

    # Use Plotly's qualitative color palette cycle
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Bold

    tiers = [
        ("1. Baseline Literature", bounds_baseline),
        ("2. MI & N-Gram (Imp 1+2)", bounds_mi),
        ("3. DBSCAN Clustering (Imp 3)", bounds_dbscan)
    ]

    for tier_name, bounds in tiers:
        # Reconstruct continuous field segments
        full_bounds = sorted(list(set([0] + list(bounds) + [total_bytes])))
        
        for i in range(len(full_bounds) - 1):
            start = full_bounds[i]
            end = full_bounds[i+1]
            length = end - start
            end_label = end - 1
            
            # Pick color cyclically based on field index
            color = palette[i % len(palette)]
            
            fig.add_trace(go.Bar(
                y=[tier_name],
                x=[length],
                name=f"Field {i:02d} [{start}:{end_label}]",
                orientation='h',
                marker=dict(
                    color=color,
                    line=dict(color='#111111', width=1.5) # Crisp dark borders between fields
                ),
                hovertemplate=(
                    f"<b>Field {i:02d}</b><br>"
                    f"<b>Range:</b> [{start} : {end_label}]<br>"
                    f"<b>Length:</b> {length} Bytes<extra></extra>"
                ),
                showlegend=False
            ))

    fig.update_layout(
        barmode='stack',
        title="<b>3-Tier Visual Schema Comparison</b>",
        xaxis=dict(
            title="Byte Offset",
            tickmode="linear",
            dtick=2, # Ticks every 2 bytes for better readability
            range=[0, total_bytes]
        ),
        yaxis=dict(autorange="reversed"), # Top-to-bottom layout
        height=380,
        margin=dict(l=20, r=20, t=50, b=40),
        template="plotly_dark"
    )
    
    return fig

@st.cache_data
def load_protocol_data(protocol_name):
    if protocol_name == "ARP":
        csv_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'raw', 'ARP1275-hex.csv'))
    else:
        csv_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'raw', f"{protocol_name.lower()}_traces.csv"))
    
    if os.path.exists(csv_path):
        try:
            raw_df = pd.read_csv(csv_path, dtype=str)
            if 'hex' not in raw_df.columns: return pd.DataFrame()

            hex_digits = set(string.hexdigits)
            byte_matrix = []
            
            for hex_string in raw_df['hex'].dropna():
                clean_str = str(hex_string).strip().replace("0x", "").replace(" ", "")
                if len(clean_str) > 0 and set(clean_str).issubset(hex_digits):
                    bytes_row = [int(clean_str[i:i+2], 16) for i in range(0, len(clean_str), 2) if len(clean_str[i:i+2]) == 2]
                    if len(bytes_row) > 0: byte_matrix.append(bytes_row)
            
            df = pd.DataFrame(byte_matrix)
            if df.empty: return pd.DataFrame()
                
            max_protocol_len = 28
            if len(df.columns) > max_protocol_len:
                df = df.iloc[:, :max_protocol_len]
                
            df.columns = list(range(len(df.columns)))
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame(np.random.randint(0, 256, size=(100, 28)))

@st.cache_data
def load_ground_truth(protocol_name):
    if protocol_name == "ARP":
        return pd.DataFrame({
            "Actual Field": ["Hardware Type", "Protocol Type", "Hardware/Protocol Size", "Opcode", "Sender MAC", "Sender IP", "Target MAC", "Target IP"],
            "Start Offset": [0, 2, 4, 6, 8, 14, 18, 24],
            "End Offset": [1, 3, 5, 7, 13, 17, 23, 27]
        })
    return pd.DataFrame()

df_packets = load_protocol_data(protocol_choice)
df_ground_truth = load_ground_truth(protocol_choice)

if not df_packets.empty:
    st.sidebar.success(f"Loaded {len(df_packets)} packets ({len(df_packets.columns)} bytes/packet)")

# --- ROADMAP PROCESS PIPELINE ---
st.markdown("### 🗺️ Algorithmic Pipeline Flow")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.info("**1. Byte Matrix Generation**\n\nConverts hex streams into 2D byte integer arrays.")
col_b.info("**2. Information Metrics**\n\nCalculates FVI & Shannon Entropy per byte offset.")
col_c.info("**3. BCS & Mutual Info**\n\nDetects candidates & prevents multi-byte field fragmentation.")
col_d.info("**4. N-Gram Tokenization**\n\nFractures massive invariant static blocks into logical words.")

st.divider()

def render_3bar_visual_comparison(schema_baseline, schema_mi, ground_truth_df, total_bytes):
    st.subheader("⚔️ 3-Tier Schema Comparison: Evolutionary Journey")
    
    fig = go.Figure()
    colors = ['#4A6FA5', '#166088', '#4CB944', '#F0A202', '#F18805', '#D95D39', '#202C59', '#8B5CF6']
    
    def add_schema_bar(schema_df, y_label):
        for idx, row in schema_df.iterrows():
            rng = row['Offset Range'].strip('[]').split(':')
            start, end = int(rng[0]), int(rng[1])
            length = end - start + 1
            color = colors[idx % len(colors)]
            
            fig.add_trace(go.Bar(
                name=row['Field ID'], x=[length], y=[y_label], orientation='h',
                marker=dict(color=color, line=dict(color='#111827', width=1.5)),
                hovertemplate=f"<b>{row['Field ID']}</b><br>Offsets: [{start}:{end}]<br>Length: {length}B<br>Typology: {row['Semantic Typology']}<extra></extra>",
                showlegend=False
            ))

    add_schema_bar(schema_mi, '3. MI & N-Gram Enhanced AI')
    add_schema_bar(schema_baseline, '2. Baseline Literature')

    if not ground_truth_df.empty:
        rfc_colors = ["#6366F1", "#EC4899", "#14B8A6", "#8B5CF6", "#F97316", "#06B6D4", "#A855F7", "#EAB308"]
        for idx, row in ground_truth_df.iterrows():
            start, end = int(row['Start Offset']), int(row['End Offset'])
            length = end - start + 1
            
            fig.add_trace(go.Bar(
                name=row['Actual Field'], x=[length], y=['1. Ground Truth (RFC)'], orientation='h',
                marker=dict(color=rfc_colors[idx % len(rfc_colors)], line=dict(color='#111827', width=1.5)),
                hovertemplate=f"<b>{row['Actual Field']}</b><br>Offsets: [{start}:{end}]<br>Length: {length}B<extra></extra>",
                showlegend=False
            ))

    fig.update_layout(
        barmode='stack', template='plotly_dark', height=300,
        xaxis_title="Byte Offset", xaxis=dict(range=[0, total_bytes], tick0=0, dtick=2),
        yaxis=dict(categoryorder='array', categoryarray=["1. Ground Truth (RFC)", "2. Baseline Literature", "3. MI & N-Gram Enhanced AI"]),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- EXECUTION SECTION ---
if st.sidebar.button("🚀 Run Segmentation Pipeline", type="primary"):
    with st.spinner("Processing protocol trace matrix with advanced heuristics..."):
        
        # Initialize engine with the selected N-Gram size
        engine = ProtocolSegmentationEngine(df_packets, ngram_size=ngram_size)
        fvi_df = engine.calculate_fvi_and_typology()
        
        # Unpack all 6 outputs returned by the 3-tier engine
        (
            schema_baseline, 
            schema_mi, 
            schema_dbscan, 
            bounds_baseline, 
            final_bounds,   # <--- Assign final_bounds here (MI + N-Gram result)
            bounds_dbscan
        ) = engine.run_pipeline()

        # --- TOP KPI METRICS ---
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        kpi1.metric("Total Bytes / Packet", f"{engine.num_bytes} B")
        kpi2.metric("Baseline Boundaries", f"{len(bounds_baseline)}")
        kpi3.metric("DBSCAN Clusters", f"{len(bounds_dbscan)}")
        kpi4.metric("Final Refined Boundaries", f"{len(final_bounds)}")

        st.divider()

        # --- VISUALIZATION 1: FVI Entropy Profile ---
        st.subheader("📈 Field Variability Index (FVI) & Discovered Boundaries")
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fvi_df['offset'], y=fvi_df['fvi'], mode='lines+markers', 
            name='FVI Variability', line=dict(color='#3B82F6', width=2), marker=dict(size=4)
        ))
        
        for b in final_bounds:
            fig.add_vline(x=b - 0.5, line_width=2, line_dash="dash", line_color="#EF4444")

        fig.update_layout(
            xaxis_title="Byte Offset", yaxis_title="Variability Index (0 = Invariant, 1 = Max Entropy)",
            template="plotly_dark", height=320, margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # --- VISUALIZATION 2: 3-Bar Comparative Schema ---
        render_3bar_visual_comparison(schema_baseline, schema_mi, df_ground_truth, len(df_packets.columns))
        st.plotly_chart(plot_3tier_comparison(bounds_baseline, final_bounds, bounds_dbscan, engine.num_bytes), use_container_width=True)

        st.divider()

        with st.expander("🎓 Architecture Briefing: Overcoming Segmentation Challenges", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            
            with c1:
                st.error("**Obstacle 1: The Binary Masking Error**")
                st.markdown("Resolved misaligned parsing where CSV binary streams were evaluated incorrectly. Applied strict `hex` series truncation to standardize header limits.")
                
            with c2:
                st.success("**Obstacle 2: Zero-Entropy Under-Segmentation**")
                st.markdown("""
                * **Symptom:** Bytes `0` through `6` clumped into a massive block.
                * **Root Cause:** Shannon Entropy collapses to $0.0$ on static constants, nullifying statistical detection.
                * **Solution:** Introduced **N-Gram Tokenization** to scan flat FVI lines and forcefully fracture them along standard 16-bit network alignments.
                """)
                
            with c3:
                st.warning("**Obstacle 3: Subnet Dynamic Fragmentation**")
                st.markdown("Implemented moving-average sliding window thresholding to prevent address fragmenting caused by static subnet prefixes masking underlying structures.")

            with c4:
                st.success("**Obstacle 4: Address Fragmentation (The AI Upgrade)**")
                st.markdown("Solved the issue of multi-byte fields splitting into fragments. Integrated **Mutual Information $I(X;Y)$** to measure joint correlation and bind adjacent address bytes.")

else:
    st.info("👈 Configure your protocol settings and click **Run Segmentation Pipeline** to execute the analysis flow.")