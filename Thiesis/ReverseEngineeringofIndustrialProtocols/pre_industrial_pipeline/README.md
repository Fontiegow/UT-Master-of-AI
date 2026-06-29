# Unsupervised Protocol Reverse Engineering (PRE) Pipeline via Information-Theoretic Boundary Segmentation

An open-source implementation of an unsupervised, trace-based framework designed to reverse-engineer unknown binary network protocols from raw, unlabeled packet captures. Leveraging information theory and statistical feature extraction, this architecture transitions from raw byte matrices to a deterministic structural protocol schema—requiring **zero prior knowledge, predefined specifications, or binary instrumentation**.

---

## 🚀 Research Achievements & Architectural Milestones
This project implements the complete data-extraction and modeling pipeline for **Pillar 1: Byte-Pattern-Based Field Segmentation**. 

* **High-Dimensional Tensor Mapping (`01_...`):** Formulated an operational vectorization function $f: P_i \longrightarrow \mathbf{x}_i \in \mathbb{Z}_{256}^{M}$ to project raw hexadecimal string arrays into optimized, fixed-boundary uint8 tensors.
* **Sequential N-Gram Tokenization (`02_...`):** Developed an overlapping sliding-window tokenization framework ($n=2, n=4$) to model structural byte co-occurrence without manual alignment constraints.
* **Global Frequency Spectrum Profiling (`03_...`):** Developed empirical probability distributions $P(T)$ to separate structural invariants from high-variance transactional layers.
* **Information Entropy Mapping (`04_...`):** Calculated vertical, column-wise Positional Shannon Entropy $H(M_j)$ and first-order boundary gradients $\Delta H(M_j)$ to locate statistical shifts.
* **Boundary Candidate Selection & Assembly (`05_...`):** Synthesized a normalized Field Variability Index (FVI) and consolidated boundary triggers into a unified Boundary Candidate Score (BCS).
* **Statistical Refinement via Symmetric KL Divergence (`06_...`):** Validated point-wise boundary candidates using sliding-window Kullback-Leibler Divergence to eliminate false positives.
* **Agglomerative Boundary Merging (`07_...`):** Executed a recursive consolidation pass across the validated boundaries using string-compatible modal category checks and FVI variance criteria ($\tau_{merge}$).
* **Deterministic Schema Reconstruction (`08_...`):** Translated mathematical split points into a high-precision, protocol-agnostic specification DataFrame, explicitly mapping ranges, lengths, and semantic typologies.

---

## 🗺️ Architectural Roadmap

### Pillar 1: Byte-Pattern-Based Field Segmentation (Completed)
- [x] 01. High-Dimensional Dataset Vectorization
- [x] 02. Sequential N-Gram Tokenization
- [x] 03. Global Token Frequency Profiling
- [x] 04. Positional Shannon Entropy Mapping
- [x] 05. Boundary Candidate Fusion & Assembly
- [x] 06. Statistical Refinement via Symmetric KL Divergence
- [x] 07. Recursive Agglomerative Boundary Merging
- [x] 08. Tabular Specification Schema Reconstruction

### Pillar 2: Semantic Inference & Key Field Identification (Next Steps)
- [ ] 09. Length Indicator Detection & Association Mapping
- [ ] 10. Sequence Number & Transaction ID Inference
- [ ] 11. Error-Detection Validation (Checksum/CRC Localizations)
- [ ] 12. Protocol State Machine Inference & FSM Synthesis

---

## 🔬 Methodological Validation & Academic Rigor

### Theoretical Acceptability
This pipeline’s roadmap is strictly grounded in the established literature of Automatic Protocol Reverse Engineering (APRE). By utilizing an information-theoretic approach (Shannon Entropy, FVI, and Symmetric KL Divergence), the pipeline models network data as an unsupervised statistical signal. 

According to peer-reviewed literature, trace-based statistical modeling provides major advantages over dynamic binary analysis:
1. **Black-Box Adaptability:** It operates entirely on network traces, completely bypassing the need for binary execution, firmware reverse engineering, or source code instrumentation.
2. **Industrial Scalability:** It scales natively to closed-source Industrial Control Protocols (ICPs) and highly dense transmission environments where proprietary systems prevent runtime debugging.
3. **Information Stability:** Information theory consistently minimizes over-segmentation risks by verifying that inferred boundaries map directly to underlying distribution anomalies rather than local sampling noise.

### Alternative Methodologies Mentioned in Literature
To maintain objectivity and contextualize our design choices, this research acknowledges alternative segmentation techniques detailed in the literature:
* **Sequence Alignment Methods (e.g., Needleman-Wunsch / UPGMA):** Frequently used in frameworks like *Netzob* or *Discoverer*. They align pairs of messages to calculate global similarity scores. While highly effective for text-heavy or variable-length fields, they suffer from high computational complexity ($O(N^2)$) when processing large-scale binary industrial corpora.
* **Hidden Semi-Markov Models (HSMM):** Utilized to infer sequential structures by estimating hidden state parameters. While powerful for understanding temporal message transitions, they demand substantial training corpora and are sensitive to hyperparameter initialization.
* **Pattern-Mining (e.g., Voting Experts / ProWord):** Uses text-mining heuristics like boundary entropy and frequency to isolate protocol "keywords". They are effective for extracting semantic commands but can struggle with continuous, unaligned numeric variables (such as timestamps or cryptographic nonces).
* **Deep Learning Segmentations (e.g., Convolutional U-Net Networks):** Modern approaches project packet byte values into 2D imagery spaces and use computer vision encoders to isolate boundaries. These provide impressive non-linear accuracy but introduce significant black-box transparency challenges and heavy hardware overhead compared to pure information metrics.

---

## 📈 Next Research Phase: Semantic Inference

With Pillar 1 fully established, the next stage of this research transitions from **structural parsing** to **semantic decoding**:

1. **Key Field Identification:** We will transition from generic labels (e.g., `Field_03`) to operational protocol meanings. By applying Pearson Correlation Matrices and Apriori Association Rules, the pipeline will calculate structural dependencies across fields. This step mathematically identifies which blocks serve as length indicators, which act as transactional counters, and which are structural checksums.
2. **State Machine Inference (FSM):** By modeling the sequential ordering of message blocks across a timeline, we will track the protocol's state transitions, mapping initialization, transaction execution, and connection teardown loops to generate formal finite state automata.

---

## 🛠️ Repository Structure
```text
├── data/
│   ├── raw/         # Uncompressed network PCAP/HEX trace streams
│   └── processed/   # Serialized NumPy binaries (.npy) and Schema DataFrames (.csv)
├── notebooks/
│   ├── 01_dataset_vectorization.ipynb
│   ├── 02_ngram_tokenization_theory.ipynb
│   ├── 03_token_frequency_analysis.ipynb
│   ├── 04_entropy_segmentation.ipynb
│   ├── 05_field_assembly.ipynb
│   ├── 06_boundary_refinement.ipynb
│   ├── 07_boundary_merging.ipynb
│   └── 08_field_reconstruction.ipynb
└── README.md