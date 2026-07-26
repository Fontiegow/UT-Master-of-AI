import numpy as np
import pandas as pd
from scipy.stats import entropy
from typing import List, Tuple

class ProtocolSegmentationEngine:
    def __init__(self, dataframe: pd.DataFrame, bcs_threshold: float = 0.3, kl_threshold: float = 1.0, merge_threshold: float = 0.15):
        """
        Initializes the Multi-Stage Protocol Segmentation Pipeline.
                
        Args:
            dataframe (pd.DataFrame): Raw byte matrix (Packets x Byte Offsets).
            bcs_threshold (float): Sensitivity cutoff for candidate detection.
            kl_threshold (float): Minimum bits of relative entropy to verify boundaries.
            merge_threshold (float): Maximum FVI delta allowed when merging adjacent fields.
        
        """
        self.df = dataframe.copy()
        self.bcs_threshold = bcs_threshold
        self.kl_threshold = kl_threshold
        self.merge_threshold = merge_threshold
        
        self.num_bytes = len(self.df.columns)
        
        # Fill missing byte values safely with 0 and convert to unsigned byte integers (0-255)
        clean_df = self.df.fillna(0)
        self.matrix = clean_df.values.astype(np.uint8)
        self.schema = None

    def calculate_fvi_and_typology(self, tau_static: float = 0.05, tau_status: float = 0.40) -> pd.DataFrame:
        """
        Notebook 05: Computes the Field Variability Index (FVI) per byte offset
        and assigns discrete semantic typology labels.
        """
        fvi_data = []
        h_max = np.log2(256)  # Theoretical max entropy for 1 byte (8 bits)

        for idx, col in enumerate(self.df.columns):
            column_data = self.df[col].dropna()
            
            if len(column_data) == 0:
                raw_entropy = 0.0
            else:
                probabilities = column_data.value_counts(normalize=True)
                raw_entropy = entropy(pk=probabilities, base=2)

            fvi_val = np.clip(raw_entropy / h_max, 0.0, 1.0)

            if fvi_val <= tau_static:
                typology = "Static (Invariant)"
            elif fvi_val <= tau_status:
                typology = "Status/Control"
            else:
                typology = "Dynamic (Payload/Address)"

            fvi_data.append({
                "offset": int(idx),
                "raw_entropy": raw_entropy,
                "fvi": fvi_val,
                "typology": typology
            })

        # Explicitly pass columns parameter to guarantee column headers
        return pd.DataFrame(fvi_data, columns=["offset", "raw_entropy", "fvi", "typology"])

        return pd.DataFrame(fvi_data)

    def detect_boundaries_bcs(self, fvi_df: pd.DataFrame, alpha: float = 0.5) -> List[int]:
        """
        Notebook 05: Fuses continuous FVI gradients with discrete typology changes
        to compute Boundary Candidate Scores (BCS) and extract candidate indices.
        """
        fvi_vector = fvi_df['fvi'].values
        typology_labels = fvi_df['typology'].values
        num_bytes = len(fvi_vector)

        # Handle empty DataFrame edge case safely
        if num_bytes == 0:
            fvi_df['bcs'] = []
            return []

        bcs_vector = np.zeros(num_bytes)

        label_map = {
            "Static (Invariant)": 0,
            "Status/Control": 1,
            "Dynamic (Payload/Address)": 2
        }
        type_ids = np.array([label_map.get(lbl, 0) for lbl in typology_labels])

        for j in range(1, num_bytes):
            delta_fvi = abs(fvi_vector[j] - fvi_vector[j - 1])
            indicator_t = 1.0 if type_ids[j] != type_ids[j - 1] else 0.0
            bcs_vector[j] = (alpha * delta_fvi) + ((1.0 - alpha) * indicator_t)

        bcs_vector[0] = 0.0
        fvi_df['bcs'] = bcs_vector

        candidate_indices = np.where(bcs_vector >= self.bcs_threshold)[0].tolist()
        return candidate_indices

    def _compute_symmetric_kl(self, candidate_idx: int, window_size: int = 4, epsilon: float = 1e-10) -> float:
        """
        Helper method (Notebook 06): Calculates Symmetric KL Divergence across horizontal windows.
        """
        N, M = self.matrix.shape
        if candidate_idx <= 0 or candidate_idx >= M:
            return 0.0

        left_bound = max(0, candidate_idx - window_size)
        right_bound = min(M, candidate_idx + window_size)

        left_window = self.matrix[:, left_bound:candidate_idx].flatten()
        right_window = self.matrix[:, candidate_idx:right_bound].flatten()

        left_counts = np.bincount(left_window, minlength=256)
        right_counts = np.bincount(right_window, minlength=256)

        P = (left_counts + epsilon) / (len(left_window) + 256 * epsilon)
        Q = (right_counts + epsilon) / (len(right_window) + 256 * epsilon)

        kl_p_q = np.sum(P * np.log2(P / Q))
        kl_q_p = np.sum(Q * np.log2(Q / P))

        return (kl_p_q + kl_q_p) / 2.0

    def refine_with_kl_divergence(self, candidates: List[int], window_size: int = 4) -> List[int]:
        """
        Notebook 06: Filters candidate boundaries using Symmetric KL Divergence.
        """
        verified_boundaries = []
        for candidate in candidates:
            kl_score = self._compute_symmetric_kl(candidate_idx=candidate, window_size=window_size)
            if kl_score >= self.kl_threshold:
                verified_boundaries.append(candidate)
        return verified_boundaries

    def consolidate_macro_boundaries(self, verified_boundaries: List[int], fvi_df: pd.DataFrame) -> List[int]:
        """
        Notebook 07: Recursively merges adjacent byte segments sharing the same 
        dominant typology and compatible FVI variance.
        """
        fvi_profile = fvi_df['fvi'].values
        fvi_categories = fvi_df['typology'].values
        max_len = self.num_bytes

        current_bounds = sorted(list(set([0] + list(verified_boundaries) + [max_len])))
        cat_array = np.array(fvi_categories)
        converged = False

        while not converged:
            converged = True
            i = 0
            while i < len(current_bounds) - 2:
                b_start = current_bounds[i]
                b_mid = current_bounds[i+1]
                b_end = current_bounds[i+2]

                fvi_seg1 = fvi_profile[b_start:b_mid]
                fvi_seg2 = fvi_profile[b_mid:b_end]

                cat_seg1 = cat_array[b_start:b_mid]
                cat_seg2 = cat_array[b_mid:b_end]

                vals1, counts1 = np.unique(cat_seg1, return_counts=True)
                mode_cat1 = vals1[np.argmax(counts1)]

                vals2, counts2 = np.unique(cat_seg2, return_counts=True)
                mode_cat2 = vals2[np.argmax(counts2)]

                mean_fvi1 = np.mean(fvi_seg1)
                mean_fvi2 = np.mean(fvi_seg2)
                delta_fvi = abs(mean_fvi1 - mean_fvi2)

                if (mode_cat1 == mode_cat2) and (delta_fvi <= self.merge_threshold):
                    current_bounds.pop(i+1)
                    converged = False
                    break

                i += 1

        return [b for b in current_bounds if b not in [0, max_len]]

    def reconstruct_schema(self, final_boundaries: List[int], fvi_df: pd.DataFrame) -> pd.DataFrame:
        """
        Notebook 08: Compiles macro-boundaries into a structured protocol schema.
        """
        fvi_profile = fvi_df['fvi'].values
        full_bounds = sorted(list(set([0] + final_boundaries + [self.num_bytes])))
        schema_records = []

        for i in range(len(full_bounds) - 1):
            start_idx = full_bounds[i]
            end_idx = full_bounds[i+1]
            length = end_idx - start_idx

            mean_fvi = np.mean(fvi_profile[start_idx:end_idx])

            if mean_fvi <= 0.05:
                typology = "Static Header / Padding"
            elif mean_fvi <= 0.30:
                typology = "Status / Opcode"
            else:
                typology = "Dynamic Address / Payload"

            schema_records.append({
                "Field ID": f"Field_{i:02d}",
                "Offset Range": f"[{start_idx} : {end_idx - 1}]",
                "Length (Bytes)": length,
                "Mean FVI": round(mean_fvi, 4),
                "Semantic Typology": typology
            })

        self.schema = pd.DataFrame(schema_records)
        return self.schema

    def run_pipeline(self) -> pd.DataFrame:
        """
        Executes all notebooks sequentially (05 -> 06 -> 07 -> 08).
        """
        fvi_df = self.calculate_fvi_and_typology()
        candidates = self.detect_boundaries_bcs(fvi_df)
        verified = self.refine_with_kl_divergence(candidates)
        final_boundaries = self.consolidate_macro_boundaries(verified, fvi_df)
        return self.reconstruct_schema(final_boundaries, fvi_df)