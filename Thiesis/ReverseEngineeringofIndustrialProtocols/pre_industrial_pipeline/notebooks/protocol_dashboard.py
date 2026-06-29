import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# --- 1. Page Configuration ---
st.set_page_config(page_title="Protocol Reverse Engineering Dashboard", layout="wide")
st.title("🛡️ Unsupervised Protocol Schema Extractor")
st.markdown("Interactive visualization of byte-pattern-based field segmentation.")

# --- 2. Data Loading ---
@st.cache_data
def load_research_artifacts():
    processed_dir = os.path.join("..", "data", "processed")

    entropy = np.load(os.path.join(processed_dir, "entropy_profile.npy"))
    fvi = np.clip(entropy / 8.0, 0, 1)
    schema_df = pd.read_csv(os.path.join(processed_dir, "final_protocol_schema.csv"))

    return entropy, fvi, schema_df

try:
    entropy_profile, fvi_profile, protocol_schema = load_research_artifacts()
except FileNotFoundError:
    st.error("Missing data artifacts. Please ensure Pillar 1 notebooks have been executed.")
    st.stop()

M_dimension = len(entropy_profile)

# --- 3. Interactive Plotly Rendering ---
st.subheader("Field Layout & Positional Variance Map")

fig = go.Figure()

# Add Shannon Entropy Line
fig.add_trace(go.Scatter(
    x=list(range(M_dimension)), 
    y=entropy_profile,
    mode='lines+markers',
    name='Shannon Entropy (Bits)',
    line=dict(color='blue', width=2),
    hovertemplate='Offset: %{x}<br>Entropy: %{y:.3f} bits<extra></extra>'
))

# Add FVI Line
fig.add_trace(go.Scatter(
    x=list(range(M_dimension)), 
    y=fvi_profile * 8.0, # Scaled to match the entropy axis for dual-viewing
    mode='lines',
    name='Scaled FVI',
    line=dict(color='purple', width=2, dash='dot'),
    hoverinfo='skip'
))

# Add Semantic Field Background Spans
color_map = {
    "Static Header / Padding": "rgba(44, 160, 44, 0.2)", # Green
    "Status / Opcode": "rgba(255, 127, 14, 0.2)",         # Orange
    "Dynamic Address / Payload": "rgba(214, 39, 40, 0.2)" # Red
}

for index, row in protocol_schema.iterrows():
    # Parse the string offset range '[start : end]' safely
    range_str = row["Offset Range"].strip("[]")
    start_idx, end_idx = map(int, range_str.split(":"))
    end_idx += 1 # Adjust for inclusive visual boundary

    typology = row["Semantic Typology"]
    bg_color = color_map.get(typology, "rgba(128, 128, 128, 0.2)")

    # Draw background rectangle for the field
    fig.add_vrect(
        x0=start_idx, x1=end_idx,
        fillcolor=bg_color, opacity=1,
        layer="below", line_width=1, line_color="black",
        annotation_text=row["Field ID"] if (end_idx - start_idx) > 1 else "",
        annotation_position="top left",
        annotation_font_size=10, annotation_font_color="black"
    )

fig.update_layout(
    xaxis_title="Byte Position Offset Index",
    yaxis_title="Information Variance Magnitude",
    hovermode="x unified",
    height=500,
    margin=dict(l=0, r=0, t=30, b=0)
)

st.plotly_chart(fig, use_container_width=True)

# --- 4. Tabular Schema Display ---
st.subheader("Reconstructed Protocol Specification")
st.dataframe(protocol_schema, use_container_width=True)
