import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from typing import List, Tuple

class ProtocolSegmentationEngine:
    """
    Multi-Stage Information-Theoretic Protocol Segmentation Engine.
    
    Now incorporates N-Gram Tokenization to handle static invariant headers 
    alongside Mutual Information for dynamic address fields.
    """

    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        bcs_threshold: float = 0.3, 
        kl_threshold: float = 1.0, 
        merge_threshold: float = 0.15,
        mi_threshold: float = 0.3,
        ngram_size: int = 2,
        static_threshold: float = 0.05
    ):
        """
        Initializes the Multi-Stage Protocol Segmentation Pipeline.
        
        Args:
            dataframe (pd.DataFrame): Raw byte matrix (Packets x Byte Offsets).
            bcs_threshold (float): Sensitivity cutoff for candidate boundary detection.
            kl_threshold (float): Minimum relative entropy (bits) required to verify boundaries.
            merge_threshold (float): Maximum FVI delta allowed when merging adjacent fields.
            mi_threshold (float): Maximum Mutual Information threshold allowed across a boundary.
            ngram_size (int): Standard protocol word size for static tokenization (default: 2 for 16-bit).
            static_threshold (float): Maximum FVI for a block to be considered 'Static'.
        """
        self.df = dataframe.copy()
        self.bcs_threshold = bcs_threshold
        self.kl_threshold = kl_threshold
        self.merge_threshold = merge_threshold
        self.mi_threshold = mi_threshold
        self.ngram_size = ngram_size
        self.static_threshold = static_threshold
        
        self.num_bytes = len(self.df.columns)
        
        # Safe casting: replace NaNs with 0 and cast matrix to unsigned 8-bit integers (0-255)
        clean_df = self.df.fillna(0)
        self.matrix = clean_df.values.astype(np.uint8)
        self.schema = None

    def calculate_fvi_and_typology(
        self, 
        tau_static: float = 0.05, 
        tau_status: float = 0.40
    ) -> pd.DataFrame:
        """
        Calculates the Field Variability Index (FVI) per byte offset using normalized Shannon Entropy.
        """
        fvi_data = []
        h_max = np.log2(256) 

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

        return pd.DataFrame(fvi_data, columns=["offset", "raw_entropy", "fvi", "typology"])

    def detect_boundaries_bcs(self, fvi_df: pd.DataFrame, alpha: float = 0.5) -> List[int]:
        """
        Fuses continuous FVI gradients with discrete typology transitions.
        """
        fvi_vector = fvi_df['fvi'].values
        typology_labels = fvi_df['typology'].values
        num_bytes = len(fvi_vector)

        if num_bytes == 0:
            fvi_df['bcs'] = []
            return []

        bcs_vector = np.zeros(num_bytes)
        label_map = {"Static (Invariant)": 0, "Status/Control": 1, "Dynamic (Payload/Address)": 2}
        type_ids = np.array([label_map.get(lbl, 0) for lbl in typology_labels])

        for j in range(1, num_bytes):
            delta_fvi = abs(fvi_vector[j] - fvi_vector[j - 1])
            indicator_t = 1.0 if type_ids[j] != type_ids[j - 1] else 0.0
            bcs_vector[j] = (alpha * delta_fvi) + ((1.0 - alpha) * indicator_t)

        bcs_vector[0] = 0.0
        fvi_df['bcs'] = bcs_vector
        return np.where(bcs_vector >= self.bcs_threshold)[0].tolist()

    def _compute_symmetric_kl(self, candidate_idx: int, window_size: int = 4, epsilon: float = 1e-10) -> float:
        """Calculates Symmetric Kullback-Leibler (KL) Divergence across sliding horizontal windows."""
        N, M = self.matrix.shape
        if candidate_idx <= 0 or candidate_idx >= M: return 0.0

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
        """Filters out statistical noise using Symmetric KL Divergence."""
        return [c for c in candidates if self._compute_symmetric_kl(c, window_size) >= self.kl_threshold]

    def prune_boundaries_with_mi(self, candidates: List[int]) -> List[int]:
        """Uses Mutual Information (MI) to prune false boundaries in multi-byte address structures."""
        pruned_boundaries = []
        for b in candidates:
            if b <= 0 or b >= self.num_bytes: continue
            col_left, col_right = self.matrix[:, b - 1], self.matrix[:, b]
            mi = mutual_info_score(col_left, col_right)
            if mi < self.mi_threshold:
                pruned_boundaries.append(b)
        return pruned_boundaries

    def consolidate_macro_boundaries(self, verified_boundaries: List[int], fvi_df: pd.DataFrame) -> List[int]:
        """Iteratively merges adjacent field units based on distribution similarity."""
        max_len = self.num_bytes
        current_bounds = sorted(list(set([0] + list(verified_boundaries) + [max_len])))
        
        while len(current_bounds) > 2:
            min_score, merge_index = float('inf'), -1
            
            for i in range(1, len(current_bounds) - 1):
                b_start, b_mid, b_end = current_bounds[i - 1], current_bounds[i], current_bounds[i + 1]
                field_left = self.matrix[:, b_start:b_mid].flatten()
                field_right = self.matrix[:, b_mid:b_end].flatten()
                
                epsilon = 1e-10
                P = (np.bincount(field_left, minlength=256) + epsilon) / (len(field_left) + 256 * epsilon)
                Q = (np.bincount(field_right, minlength=256) + epsilon) / (len(field_right) + 256 * epsilon)
                
                symmetric_kl = (np.sum(P * np.log2(P / Q)) + np.sum(Q * np.log2(Q / P))) / 2.0
                
                if symmetric_kl < min_score:
                    min_score, merge_index = symmetric_kl, i
            
            if merge_index != -1 and min_score <= self.merge_threshold:
                if (current_bounds[merge_index + 1] - current_bounds[merge_index - 1]) <= 8:
                    current_bounds.pop(merge_index)
                    continue
            break
            
        return [b for b in current_bounds if b not in [0, max_len]]

    def tokenize_static_regions(self, current_boundaries: List[int], fvi_df: pd.DataFrame) -> List[int]:
        """
        N-Gram Tokenization: Solves zero-variance under-segmentation.
        
        Scans for large contiguous blocks of static bytes (where FVI ≈ 0). Since purely 
        statistical methods fail on constants, this forces structural boundaries based on 
        standard protocol word alignments (e.g., 16-bit / 2-byte tokenization).
        """
        fvi_profile = fvi_df['fvi'].values
        full_bounds = sorted(list(set([0] + current_boundaries + [self.num_bytes])))
        final_bounds_set = set(current_boundaries)

        for i in range(len(full_bounds) - 1):
            start_idx = full_bounds[i]
            end_idx = full_bounds[i+1]
            length = end_idx - start_idx

            # Target blocks larger than the selected N-Gram size
            if length > self.ngram_size:
                mean_fvi = np.mean(fvi_profile[start_idx:end_idx])
                
                # If the block is deeply static (e.g., constant protocol identifiers)
                if mean_fvi <= self.static_threshold:
                    # Tokenize the static block by injecting boundaries at N-gram intervals
                    for split_point in range(start_idx + self.ngram_size, end_idx, self.ngram_size):
                        final_bounds_set.add(split_point)

        return sorted(list(final_bounds_set))

    def reconstruct_schema(self, final_boundaries: List[int], fvi_df: pd.DataFrame) -> pd.DataFrame:
        """Compiles verified macro-boundaries into a structured protocol schema table."""
        fvi_profile = fvi_df['fvi'].values
        full_bounds = sorted(list(set([0] + final_boundaries + [self.num_bytes])))
        schema_records = []

        for i in range(len(full_bounds) - 1):
            start_idx = full_bounds[i]
            end_idx = full_bounds[i+1]
            length = end_idx - start_idx
            mean_fvi = np.mean(fvi_profile[start_idx:end_idx])

            if mean_fvi <= self.static_threshold:
                typology = "Static Header / Padding"
            elif mean_fvi <= 0.30:
                typology = "Status / Opcode"
            elif length in [4, 6] and mean_fvi > 0.30:
                typology = "Dynamic Address (IP/MAC)"
            else:
                typology = "Dynamic Payload"

            schema_records.append({
                "Field ID": f"Field_{i:02d}",
                "Offset Range": f"[{start_idx} : {end_idx - 1}]",
                "Length (Bytes)": length,
                "Mean FVI": round(float(mean_fvi), 4),
                "Semantic Typology": typology
            })

        self.schema = pd.DataFrame(schema_records)
        return self.schema

    def run_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], List[int], List[int]]:
        """Executes the complete multi-stage pipeline."""
        fvi_df = self.calculate_fvi_and_typology()
        candidates = self.detect_boundaries_bcs(fvi_df)
        kl_verified = self.refine_with_kl_divergence(candidates)
        
        # 1. Baseline Literature Schema 
        baseline_bounds = self.consolidate_macro_boundaries(kl_verified, fvi_df)
        schema_baseline = self.reconstruct_schema(baseline_bounds, fvi_df)
        
        # 2. AI Upgrade Schema (Mutual Information + N-Gram Tokenization)
        mi_verified = self.prune_boundaries_with_mi(kl_verified)
        consolidated_mi_bounds = self.consolidate_macro_boundaries(mi_verified, fvi_df)
        
        # Apply Improvement 2: Fracture zero-entropy blocks using N-gram alignment
        final_tokenized_bounds = self.tokenize_static_regions(consolidated_mi_bounds, fvi_df)
        schema_mi = self.reconstruct_schema(final_tokenized_bounds, fvi_df)
        
        return schema_baseline, schema_mi, candidates, kl_verified, final_tokenized_bounds