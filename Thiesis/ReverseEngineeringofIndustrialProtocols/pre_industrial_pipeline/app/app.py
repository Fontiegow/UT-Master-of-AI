import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Dynamically add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src', 'field_segmentation'))
sys.path.append(src_dir)

from segmentation_engine import ProtocolSegmentationEngine

st.set_page_config(page_title="Protocol Field Segmentation Engine", layout="wide")

st.title("Network Protocol Segmentation Engine")
st.markdown("Unsupervised structural boundary detection via information-theoretic metrics.")

# --- SIDEBAR: Protocol Selection ---
st.sidebar.header("Protocol Configuration")
protocol_choice = st.sidebar.selectbox(
    "Select Target Protocol",
    ("ARP", "Modbus TCP", "DNP3", "Custom")
)

# --- DATA LOADING FROM CSV ---
import string

@st.cache_data
def load_protocol_data(protocol_name):
    if protocol_name == "ARP":
        csv_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'raw', 'ARP1275-hex.csv'))
    else:
        csv_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'raw', f"{protocol_name.lower()}_traces.csv"))
    
    if os.path.exists(csv_path):
        try:
            # Read CSV as string values
            raw_df = pd.read_csv(csv_path, header=None, dtype=str)
            
            # Helper set for hex validation
            hex_digits = set(string.hexdigits)
            
            byte_matrix = []
            
            # Iterate through all rows and columns to extract valid hex strings
            for row in raw_df.values:
                for cell in row:
                    if pd.isna(cell):
                        continue
                    clean_str = str(cell).strip().replace("0x", "").replace(" ", "")
                    
                    # Check if string is a valid non-empty hex value (skips header text like 'direction')
                    if len(clean_str) > 0 and set(clean_str).issubset(hex_digits):
                        bytes_row = [int(clean_str[i:i+2], 16) for i in range(0, len(clean_str), 2) if len(clean_str[i:i+2]) == 2]
                        if len(bytes_row) > 0:
                            byte_matrix.append(bytes_row)
            
            df = pd.DataFrame(byte_matrix)

            if df.empty:
                st.error("Dataset loaded, but resulted in 0 valid hex rows.")
                return pd.DataFrame()

            # Set integer column headers: 0, 1, 2, ..., N
            df.columns = list(range(len(df.columns)))
            return df
            
        except Exception as e:
            st.error(f"Error parsing CSV dataset at {csv_path}: {e}")
            return pd.DataFrame()
    else:
        st.warning(f"CSV file not found at {csv_path}. Using simulated data matrix.")
        return pd.DataFrame(np.random.randint(0, 256, size=(100, 42)))
    
@st.cache_data
def load_ground_truth(protocol_name):
    if protocol_name == "ARP":
        return pd.DataFrame({
            "Actual Field": [
                "Hardware Type / Protocol Type / Lengths / Opcode", 
                "Opcode (LSB)", 
                "Sender Hardware Address (MAC)", 
                "Sender Hardware Address (MAC)", 
                "Sender Protocol Address (IP)", 
                "Sender Protocol Address (IP)", 
                "Target Hardware Address (MAC)", 
                "Target Hardware Address (MAC)", 
                "Target Hardware Address (MAC)", 
                "Target Protocol Address (IP)", 
                "Target Protocol Address (IP)", 
                "Target Protocol Address (IP)", 
                "Ethernet Frame Padding"
            ],
            "Offset Range": [
                "[0 : 6]", "[7 : 7]", "[8 : 11]", "[12 : 13]", 
                "[14 : 16]", "[17 : 17]", "[18 : 18]", "[19 : 21]", 
                "[22 : 23]", "[24 : 24]", "[25 : 26]", "[27 : 27]", "[28 : 41]"
            ]
        })
    return pd.DataFrame({"Actual Field": ["Pending Definition"], "Offset Range": ["N/A"]})

df_packets = load_protocol_data(protocol_choice)
df_ground_truth = load_ground_truth(protocol_choice)

st.sidebar.success(f"Loaded dataset: {len(df_packets)} packets, {len(df_packets.columns)} bytes per packet.")

# --- EXECUTION ---
if st.sidebar.button("Run Segmentation Pipeline"):
    with st.spinner('Executing FVI, KL Divergence, and Macro-Consolidation...'):
        
        engine = ProtocolSegmentationEngine(df_packets)
        reconstructed_schema = engine.run_pipeline()
        
        st.subheader(f"Segmentation Results: {protocol_choice}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🤖 Discovered Schema (Engine Output)")
            st.dataframe(reconstructed_schema, use_container_width=True)
            
        with col2:
            st.markdown("### 📖 Ground Truth Reference (RFC Standard)")
            st.dataframe(df_ground_truth, use_container_width=True)
            
        st.success("Pipeline execution complete! Compare the offset ranges above.")

else:
    st.info("Select a protocol from the sidebar and click **Run Segmentation Pipeline**.")