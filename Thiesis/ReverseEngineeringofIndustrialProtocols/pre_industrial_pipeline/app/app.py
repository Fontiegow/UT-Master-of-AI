import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.express as px
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

# Custom Styling for Metric Cards & Layout
st.markdown("""
<style>
    .metric-card {
        background-color: #1E222B;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #4F8BF9;
    }
    .status-box {
        background-color: #111827;
        padding: 12px;
        border-radius: 6px;
        border: 1px solid #374151;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Network Protocol Segmentation Engine")
st.markdown("Unsupervised structural boundary detection using information-theoretic & correlation metrics.")

# --- SIDEBAR ---
st.sidebar.header("Protocol Configuration")
protocol_choice = st.sidebar.selectbox("Select Target Protocol", ("ARP", "Modbus TCP", "DNP3"))

@st.cache_data
def load_protocol_data(protocol_name):
    if protocol_name == "ARP":
        csv_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'raw', 'ARP1275-hex.csv'))
    else:
        csv_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'raw', f"{protocol_name.lower()}_traces.csv"))
    
    if os.path.exists(csv_path):
        try:
            # Read CSV and respect the header row
            raw_df = pd.read_csv(csv_path, dtype=str)
            
            # Ensure the 'hex' column exists
            if 'hex' not in raw_df.columns:
                st.error("Missing 'hex' column in dataset.")
                return pd.DataFrame()

            hex_digits = set(string.hexdigits)
            byte_matrix = []
            
            # Only process the specific 'hex' column
            for hex_string in raw_df['hex'].dropna():
                clean_str = str(hex_string).strip().replace("0x", "").replace(" ", "")
                
                if len(clean_str) > 0 and set(clean_str).issubset(hex_digits):
                    # Convert pairs of hex characters into integer bytes
                    bytes_row = [int(clean_str[i:i+2], 16) for i in range(0, len(clean_str), 2) if len(clean_str[i:i+2]) == 2]
                    
                    if len(bytes_row) > 0:
                        byte_matrix.append(bytes_row)
            
            df = pd.DataFrame(byte_matrix)
            
            if df.empty: 
                return pd.DataFrame()
                
            # Truncate trailing Ethernet padding (zeros) to isolate the ARP layer.
            # Standard ARP is 28 bytes (offsets 0 to 27). 
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
else:
    st.sidebar.error("Failed to load dataset.")

# --- ROADMAP PROCESS PIPELINE ---
st.markdown("### 🗺️ Algorithmic Pipeline Flow")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.info("**1. Byte Matrix Generation**\n\nConverts hex streams into 2D byte integer arrays.")
col_b.info("**2. Information Metrics**\n\nCalculates FVI & Shannon Entropy per byte offset.")
col_c.info("**3. BCS & Mutual Info Filtering**\n\nDetects candidate boundaries & prunes correlated byte splits.")
col_d.info("**4. Macro Consolidation**\n\nMerges similar regions into final schema layout.")

st.divider()

# --- VISUALIZATION HELPER FUNCTIONS ---
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
                name=row['Field ID'],
                x=[length],
                y=[y_label],
                orientation='h',
                marker=dict(color=color, line=dict(color='#111827', width=1.5)),
                hovertemplate=f"<b>{row['Field ID']}</b><br>Offsets: [{start}:{end}]<br>Length: {length}B<br>Typology: {row['Semantic Typology']}<extra></extra>",
                showlegend=False
            ))

    # Bar 3: MI-Enhanced Engine (AI Upgrade)
    add_schema_bar(schema_mi, '3. MI-Enhanced AI')

    # Bar 2: Baseline Literature Engine (Entropy + KL)
    add_schema_bar(schema_baseline, '2. Baseline Literature')

    # Bar 1: Ground Truth (RFC Standard)
    if not ground_truth_df.empty:
        rfc_colors = ["#6366F1", "#EC4899", "#14B8A6", "#8B5CF6", "#F97316", "#06B6D4", "#A855F7", "#EAB308"]
        for idx, row in ground_truth_df.iterrows():
            start = int(row['Start Offset'])
            end = int(row['End Offset'])
            length = end - start + 1
            color = rfc_colors[idx % len(rfc_colors)]
            
            fig.add_trace(go.Bar(
                name=row['Actual Field'],
                x=[length],
                y=['1. Ground Truth (RFC)'],
                orientation='h',
                marker=dict(color=color, line=dict(color='#111827', width=1.5)),
                hovertemplate=f"<b>{row['Actual Field']}</b><br>Offsets: [{start}:{end}]<br>Length: {length}B<extra></extra>",
                showlegend=False
            ))

    fig.update_layout(
        barmode='stack',
        template='plotly_dark',
        height=300,
        xaxis_title="Byte Offset",
        xaxis=dict(range=[0, total_bytes], tick0=0, dtick=2),
        yaxis=dict(categoryorder='array', categoryarray=["1. Ground Truth (RFC)", "2. Baseline Literature", "3. MI-Enhanced AI"]),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --- EXECUTION SECTION ---
if st.sidebar.button("🚀 Run Segmentation Pipeline", type="primary"):
    with st.spinner("Processing protocol trace matrix with Mutual Information..."):
        
        engine = ProtocolSegmentationEngine(df_packets)
        fvi_df = engine.calculate_fvi_and_typology()
        
        # Execute pipeline and retrieve both baseline and MI-enhanced results
        schema_baseline, schema_mi, candidates, kl_verified, final_bounds = engine.run_pipeline()

        # --- TOP KPI METRICS ---
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Bytes Analyzed", f"{len(df_packets.columns)} Bytes")
        kpi2.metric("Discovered Fields (MI)", f"{len(schema_mi)} Fields")
        kpi3.metric("Raw Candidate Bounds", f"{len(candidates)}")
        kpi4.metric("MI-Pruned Boundaries", f"{len(final_bounds)}")

        st.divider()

        # --- VISUALIZATION 1: FVI Entropy Profile Line Chart ---
        st.subheader("📈 Field Variability Index (FVI) & Discovered Boundaries")
        
        fig = go.Figure()
        
        # Plot FVI Curve
        fig.add_trace(go.Scatter(
            x=fvi_df['offset'], 
            y=fvi_df['fvi'], 
            mode='lines+markers', 
            name='FVI Variability',
            line=dict(color='#3B82F6', width=2),
            marker=dict(size=4)
        ))
        
        # Overlay Discovered Boundary Vertical Lines
        for b in final_bounds:
            fig.add_vline(x=b - 0.5, line_width=2, line_dash="dash", line_color="#EF4444")

        fig.update_layout(
            xaxis_title="Byte Offset",
            yaxis_title="Variability Index (0 = Invariant, 1 = Max Entropy)",
            template="plotly_dark",
            height=320,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # --- VISUALIZATION 2: 3-Bar Comparative Schema Layout ---
        render_3bar_visual_comparison(schema_baseline, schema_mi, df_ground_truth, len(df_packets.columns))

        st.divider()

        # --- DETAILED COMPARISON TABLES ---
        with st.expander("🎓 Architecture Briefing: Overcoming Segmentation Challenges", expanded=True):
            st.markdown("""
            ### 🔬 Evolutionary Journey of the Segmentation Engine
            Transitioning from standard literature heuristics to an unsupervised machine learning architecture required identifying and solving distinct mathematical roadblocks.
            """)
            
            c1, c2, c3, c4 = st.columns(4)
            
            with c1:
                st.error("**Obstacle 1: The Binary Masking Error**")
                st.markdown("""
                * **Symptom:** Evaluated $184$ bytes instead of standard $28$-byte ARP header.
                * **Root Cause:** Raw CSV binary digits (`0000...`) parsed as hex characters.
                * **Solution:** Isolated parsing strictly to `hex` series and truncated padding to 28 bytes.
                """)
                
            with c2:
                st.warning("**Obstacle 2: Zero-Entropy Under-Segmentation**")
                st.markdown("""
                * **Symptom:** Bytes `0` through `6` merged into a massive block (`Field_00`).
                * **Root Cause:** Absolute header constants cause Shannon Entropy to collapse to $0.0$.
                * **Solution:** Applied 16-bit word boundary heuristics on zero-variance spans.
                """)
                
            with c3:
                st.warning("**Obstacle 3: Subnet Dynamic Fragmentation**")
                st.markdown("""
                * **Symptom:** Dynamic IP/MAC addresses broken across subnet prefixes.
                * **Root Cause:** Static IP prefixes (e.g. `192.168.x.x`) cause artificial FVI drops.
                * **Solution:** Implemented moving-average sliding window thresholding.
                """)

            with c4:
                st.success("**Obstacle 4: Address Fragmentation (The AI Upgrade)**")
                st.markdown("""
                * **Symptom:** Multi-byte fields split into 2-byte or 3-byte fragments.
                * **Root Cause:** KL Divergence treats adjacent bytes independently.
                * **Solution:** Integrated **Mutual Information ($I(X;Y)$)** to measure joint correlation and bind dependent address bytes back together.
                """)

else:
    st.info("👈 Click **Run Segmentation Pipeline** in the sidebar to execute the analysis flow.")