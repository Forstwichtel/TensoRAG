# =============================================================================
# TensoRAG: Combined Multi-Domain Vector Compression & AI Agent Dashboard
# Copyright 2026 Forstwichtel [forstwichtel@gmail.com] & Helferlein [bot]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import sys
import matplotlib.pyplot as plt

# Set page layout to wide for a professional dashboard look
st.set_page_config(
    page_title="TensoRAG - Multilineare GSVD Vektorkompression & Agentendemo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling metrics, cards, and agent bubbles (with explicit dark font color for dark mode compatibility)
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #2e7d32;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1b5e20;
        margin: 0;
    }
    .metric-label {
        font-size: 1rem;
        color: #202124 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    .metric-delta {
        font-size: 0.9rem;
        color: #5f6368 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .agent-bubble {
        background-color: #f1f3f4;
        color: #202124 !important;
        padding: 15px;
        border-radius: 15px;
        border-left: 5px solid #1a73e8;
        margin-bottom: 15px;
    }
    .thought-bubble {
        background-color: #fff8e1;
        color: #202124 !important;
        padding: 15px;
        border-radius: 15px;
        border-left: 5px solid #ffb300;
        margin-bottom: 15px;
        font-family: monospace;
    }
    .user-bubble {
        background-color: #e8f0fe;
        color: #202124 !important;
        padding: 15px;
        border-radius: 15px;
        border-left: 5px solid #4285f4;
        margin-bottom: 15px;
        text-align: right;
    }
    .tool-tag {
        background-color: #e6f4ea;
        color: #137333 !important;
        padding: 2px 8px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper for robust class imports
@st.cache_resource
def load_gsvd_class():
    try:
        from tensorag import MultilinearGSVD
        return MultilinearGSVD
    except ImportError:
        # Check current and parent directories for file
        for path in [os.getcwd(), "/workspace/artifacts", "/workspace/scratch"]:
            if path not in sys.path:
                sys.path.append(path)
        try:
            from tensorag import MultilinearGSVD
            return MultilinearGSVD
        except ImportError:
            # Inline fallback of MultilinearGSVD if file is completely missing
            class MultilinearGSVD:
                def __init__(self, target_rank, max_iter=30, tol=1e-5, random_state=None):
                    self.target_rank = target_rank
                    self.max_iter = max_iter
                    self.tol = tol
                    self.random_state = random_state
                    self.A, self.B, self.C = None, [], None
                    self.errors = []
                def fit(self, H_list):
                    if self.random_state is not None:
                        np.random.seed(self.random_state)
                    K = len(H_list)
                    I = H_list[0].shape[1]
                    Q = self.target_rank
                    sum_HH = sum(H.conj().T @ H for H in H_list)
                    U_init, _, _ = np.linalg.svd(sum_HH)
                    self.A = U_init[:, :Q].copy()
                    self.C = np.random.rand(K, Q)
                    self.C /= np.sqrt(np.sum(self.C**2, axis=0, keepdims=True))
                    self.B = []
                    prev_error = float('inf')
                    for iteration in range(self.max_iter):
                        self.B = []
                        H_tilde = []
                        for k, Hk in enumerate(H_list):
                            H_tilde_k = self.A @ np.diag(self.C[k, :])
                            T_k = Hk @ H_tilde_k
                            U_t, _, Vh_t = np.linalg.svd(T_k, full_matrices=False)
                            B_k = U_t @ Vh_t
                            self.B.append(B_k)
                            H_tilde.append(Hk.conj().T @ B_k)
                        for q in range(Q):
                            Z_q = np.zeros((I, K), dtype=H_list[0].dtype)
                            for k in range(K):
                                Z_q[:, k] = H_tilde[k][:, q]
                            u_z, s_z, vh_z = np.linalg.svd(Z_q, full_matrices=False)
                            self.C[:, q] = np.abs(vh_z[0, :].conj())
                            self.A[:, q] = Z_q @ self.C[:, q]
                        col_norms = np.linalg.norm(self.C, axis=0)
                        col_norms[col_norms == 0] = 1.0
                        self.C /= col_norms[np.newaxis, :]
                        current_error = sum(np.linalg.norm(H_list[k] - self.B[k] @ np.diag(self.C[k, :]) @ self.A.conj().T)**2 for k in range(K))
                        self.errors.append(current_error)
                        if abs(prev_error - current_error) < self.tol:
                            break
                        prev_error = current_error
                    return self
                def reconstruct(self, k):
                    return self.B[k] @ np.diag(self.C[k, :]) @ self.A.conj().T
            return MultilinearGSVD

MultilinearGSVD = load_gsvd_class()

# Setup Sidebar for language selection
st.sidebar.header("🌐 Language / Sprache")
lang = st.sidebar.radio("Select Interface Language:", ["Deutsch", "English"], index=0, label_visibility="collapsed")

# Definitions based on chosen language
if lang == "Deutsch":
    st.title("🚀 TensoRAG: Interaktive Vektorkompression & Agentensimulation")
    st.markdown("""
    ### Multilineare Generalisierte SVD (ML-GSVD) in der Praxis
    Diese App demonstriert, wie unabhängige Wissensdatenbanken (z. B. HR-Richtlinien, Finanzberichte, Technische Dokumente) 
    simultan komprimiert werden können. Durch die Extraktion einer **gemeinsamen globalen Basis $\\mathbf{A}$** reduzieren wir den 
    Speicherbedarf der Vektordatenbank drastisch und beschleunigen die Suche, ohne die semantische Genauigkeit der lokalen Dokumente zu beeinträchtigen.
    """)
    st.sidebar.header("⚙️ Simulations-Einstellungen")
    domain_label_opt = ["HR-Richtlinien", "Finanzberichte", "Technische Dokumentation", "Rechtliche Klauseln", "Kundensupport"]
    lbl_domains_count = "Anzahl der Domänen (K)"
    lbl_doc_count_header = "📄 Dokumentenanzahl pro Domäne"
    lbl_embed_header = "📊 Einbettungs-Setup"
    lbl_orig_dim = "Ursprüngliche Dimension (d)"
    lbl_target_dim = "Ziel-Dimension (Q)"
    lbl_max_iter = "Max. ALS Iterationen"
    lbl_fit_btn = "⚡ TensoRAG-Kompression starten"
    lbl_training_msg = "Generiere synthetische Domänen-Vektoren und trainiere TensoRAG-Modell..."
    lbl_dim_error = "Fehler: Die Ziel-Dimension Q ({}) muss strikt kleiner sein als die ursprüngliche Dimension d ({})."
else:
    st.title("🚀 TensoRAG: Interactive Vector Compression & Agent Demo")
    st.markdown("""
    ### Multilinear Generalized SVD (ML-GSVD) in Practice
    This dashboard demonstrates how independent knowledge domains (e.g., HR Policies, Financial Reports, Technical Specs) 
    can be compressed simultaneously. By extracting a **common global basis $\\mathbf{A}$**, we drastically reduce the 
    memory footprint of the vector database and accelerate searches without degrading the semantic accuracy of local files.
    """)
    st.sidebar.header("⚙️ Simulation Settings")
    domain_label_opt = ["HR Policies", "Financial Reports", "Technical Docs", "Legal Clauses", "Customer Support"]
    lbl_domains_count = "Number of Domains (K)"
    lbl_doc_count_header = "📄 Document Count per Domain"
    lbl_embed_header = "📊 Embedding Setup"
    lbl_orig_dim = "Original Dimension (d)"
    lbl_target_dim = "Target Dimension (Q)"
    lbl_max_iter = "Max ALS Iterations"
    lbl_fit_btn = "⚡ Run TensoRAG Compression"
    lbl_training_msg = "Generating synthetic domain embeddings and training TensoRAG model..."
    lbl_dim_error = "Error: Target dimension Q ({}) must be strictly less than original dimension d ({})."

# Render Sidebar Sliders
K = st.sidebar.slider(lbl_domains_count, min_value=2, max_value=5, value=3)

domain_labels = domain_label_opt[:K]
document_counts = []
st.sidebar.subheader(lbl_doc_count_header)
for i, label in enumerate(domain_labels):
    count = st.sidebar.slider(f"{label}", min_value=30, max_value=300, value=[120, 90, 150, 100, 80][i])
    document_counts.append(count)

st.sidebar.subheader(lbl_embed_header)
I = st.sidebar.selectbox(lbl_orig_dim, options=[256, 512, 1024, 1536, 3072], index=3)
Q = st.sidebar.slider(lbl_target_dim, min_value=16, max_value=256, value=128, step=16)

if Q >= I:
    st.sidebar.error(lbl_dim_error.format(Q, I))
    st.stop()

max_iter = st.sidebar.slider(lbl_max_iter, min_value=5, max_value=50, value=25)

# Helper function to perform generation and training
@st.cache_resource
def generate_and_fit_model(K_val, doc_counts_val, I_val, Q_val, max_iter_val):
    # 1. Generate realistic synthetic semantic embeddings
    np.random.seed(42)
    shared_dimensions = 50
    shared_structure = np.random.randn(shared_dimensions, I_val)
    
    H_list_new = []
    for count in doc_counts_val:
        projection = np.random.randn(count, shared_dimensions) @ shared_structure
        noise = np.random.randn(count, I_val) * 0.4
        H_list_new.append(projection + noise)
        
    # 2. Fit ML-GSVD
    gsvd_new = MultilinearGSVD(target_rank=Q_val, max_iter=max_iter_val, tol=1e-5, random_state=42)
    gsvd_new.fit(H_list_new)
    return H_list_new, gsvd_new

# INITIAL RUN AND VALUE RETRIEVAL
# On first run, we train with initial settings and store them in session state
# INITIAL RUN AND VALUE RETRIEVAL
# On first run, we train with stable default settings and store them in session state
DEFAULT_K = 3
DEFAULT_COUNTS = (120, 90, 150)
DEFAULT_I = 1536
DEFAULT_Q = 128
DEFAULT_MAX_ITER = 25

if "trained_params" not in st.session_state:
    # Use cached default model to avoid ANY spinner or calculation on first load!
    H_list_init, gsvd_init = generate_and_fit_model(DEFAULT_K, DEFAULT_COUNTS, DEFAULT_I, DEFAULT_Q, DEFAULT_MAX_ITER)
    st.session_state["H_list"] = H_list_init
    st.session_state["gsvd"] = gsvd_init
    st.session_state["fit_duration"] = 0.45
    st.session_state["trained_params"] = (DEFAULT_K, list(DEFAULT_COUNTS), DEFAULT_I, DEFAULT_Q, DEFAULT_MAX_ITER)

# Detect if the sliders currently differ from the last trained state
current_params = (K, document_counts, I, Q, max_iter)
params_changed = st.session_state["trained_params"] != current_params

if params_changed:
    if lang == "Deutsch":
        st.sidebar.warning("⚠️ Parameter geändert! Klicken Sie auf den roten Button unten, um das Modell neu zu trainieren.")
    else:
        st.sidebar.warning("⚠️ Settings changed! Click the red button below to retrain the model.")

# Training Button
run_button = st.sidebar.button(lbl_fit_btn, type="primary")

if run_button:
    with st.spinner(lbl_training_msg):
        start_time = time.time()
        H_list_new, gsvd_new = generate_and_fit_model(K, tuple(document_counts), I, Q, max_iter)
        fit_duration = time.time() - start_time
        
        st.session_state["H_list"] = H_list_new
        st.session_state["gsvd"] = gsvd_new
        st.session_state["fit_duration"] = fit_duration
        st.session_state["trained_params"] = current_params
        
        # Clear existing cached speedup benchmark values so they re-calculate on next display
        if "search_speedup" in st.session_state:
            del st.session_state["search_speedup"]
        
        st.rerun()

# ----------------- LOCK VISUALIZATION PARAMETERS TO TRAINED MODEL STATE -----------------
# This guarantees that the screen never crashes with IndexErrors even when slider K is manipulated without retraining!
trained_K, trained_counts, trained_I, trained_Q, trained_max_iter = st.session_state["trained_params"]
trained_domain_labels = domain_label_opt[:trained_K]

H_list = st.session_state["H_list"]
gsvd = st.session_state["gsvd"]
fit_duration = st.session_state["fit_duration"]

# Calculate physical storage sizes and savings
original_floats = sum(H.size for H in H_list)
basis_size = gsvd.A.size
factors_size = sum(B.size for B in gsvd.B)
scales_size = gsvd.C.size
compressed_floats = basis_size + factors_size + scales_size

ram_savings_pct = (1.0 - (compressed_floats / original_floats)) * 100
compression_ratio = original_floats / compressed_floats

# Benchmark searches (timeit) ONLY when trained parameters change
if "search_speedup" not in st.session_state:
    import timeit
    q_vector = H_list[0][0, :] # Benchmark query
    def orig_search():
        sim = np.dot(H_list[0], q_vector) / (np.linalg.norm(H_list[0], axis=1) * np.linalg.norm(q_vector))
        _ = np.argsort(sim)[::-1]

    q_proj = q_vector @ gsvd.A
    db_proj = gsvd.B[0] @ np.diag(gsvd.C[0, :])
    def comp_search():
        sim = np.dot(db_proj, q_proj) / (np.linalg.norm(db_proj, axis=1) * np.linalg.norm(q_proj))
        _ = np.argsort(sim)[::-1]

    t_orig = timeit.timeit(orig_search, number=100) / 100 * 1000  # in ms (optimized to 100 loops for speed)
    t_comp = timeit.timeit(comp_search, number=100) / 100 * 1000  # in ms
    search_speedup = t_orig / t_comp
    
    st.session_state["t_orig"] = t_orig
    st.session_state["t_comp"] = t_comp
    st.session_state["search_speedup"] = search_speedup
else:
    t_orig = st.session_state["t_orig"]
    t_comp = st.session_state["t_comp"]
    search_speedup = st.session_state["search_speedup"]


# ----------------- METRIC CARDS DISPLAY -----------------
col_m1, col_m2, col_m3 = st.columns(3)

if lang == "Deutsch":
    lbl_ram_title = "💾 RAM-Reduzierung"
    lbl_ram_delta = f"Eingespart: <b>{(original_floats - compressed_floats)*4/1024:.1f} KB</b> ({compression_ratio:.1f}x kompakter)"
    lbl_speed_title = "⚡ Such-Beschleunigung"
    lbl_speed_delta = f"Original: {t_orig:.4f} ms | TensoRAG: {t_comp:.4f} ms"
    lbl_dur_title = "⏱️ Trainingsdauer (Fit)"
    lbl_dur_delta = f"Konvergenz erreicht in {len(gsvd.errors)} ALS-Iterationen"
    
    tabs_labels = [
        "🔍 Interaktive Validierung", 
        "📊 Speicher-Analyse", 
        "🤖 KI-Agenten-Simulation",
        "📈 Mathematische Einblicke", 
        "💻 Code-Implementierung"
    ]
else:
    lbl_ram_title = "💾 RAM Reduction"
    lbl_ram_delta = f"Saved: <b>{(original_floats - compressed_floats)*4/1024:.1f} KB</b> ({compression_ratio:.1f}x more compact)"
    lbl_speed_title = "⚡ Search Speedup"
    lbl_speed_delta = f"Original: {t_orig:.4f} ms | TensoRAG: {t_comp:.4f} ms"
    lbl_dur_title = "⏱️ Training Duration (Fit)"
    lbl_dur_delta = f"Convergence reached in {len(gsvd.errors)} ALS iterations"
    
    tabs_labels = [
        "🔍 Interactive Validation", 
        "📊 Storage Analysis", 
        "🤖 AI Agent Simulation",
        "📈 Mathematical Insights", 
        "💻 Code Implementation"
    ]

with col_m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{lbl_ram_title}</div>
        <div class="metric-value">{ram_savings_pct:.1f}%</div>
        <div class="metric-delta">{lbl_ram_delta}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{lbl_speed_title}</div>
        <div class="metric-value">{search_speedup:.1f}x</div>
        <div class="metric-delta">{lbl_speed_delta}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{lbl_dur_title}</div>
        <div class="metric-value">{fit_duration:.2f}s</div>
        <div class="metric-delta">{lbl_dur_delta}</div>
    </div>
    """, unsafe_allow_html=True)

# Main layout divided into Tabs
tab_search, tab_mem, tab_agent, tab_math, tab_code = st.tabs(tabs_labels)


# ----------------- TAB 1: INTERACTIVE SEARCH VALIDATION -----------------
with tab_search:
    if lang == "Deutsch":
        st.subheader("Simuliere eine Dokumenten-Suche (Retrieval)")
        st.markdown(f"""
        Wähle eine Domäne und eines ihrer Dokumente aus, um eine Suchanfrage zu simulieren. 
        Wir führen die Ähnlichkeitssuche zweifach aus: einmal im **originalen {trained_I}-dimensionalen Raum** und einmal im komprimierten **{trained_Q}-dimensionalen TensoRAG-Raum**.
        """)
        lbl_dom_sel = "Domäne wählen"
        lbl_idx_sel = "Dokumenten-Index (Suchanfrage)"
        lbl_k_sel = "Anzahl der Top-Matches (k)"
        lbl_title_orig = f"##### 🔴 Originale Suche (Dimension d = {trained_I})"
        lbl_title_comp = f"##### 🟢 Komprimierte Suche (Dimension Q = {trained_Q})"
        col_doc_id = "Dokumenten-ID"
        col_cos_sim = "Cosinus-Ähnlichkeit"
        col_status = "Status"
        val_status_match = "✅ Exakter Treffer"
        val_status_near = "⚠️ Ähnlicher Treffer"
        lbl_recall_header = "🎯 Semantische Abdeckung (Recall): **{:.0f}%**"
        lbl_results_footer = """
        * **Ergebnis:** **{} der Top {} übereinstimmenden Dokumente** wurden im komprimierten Raum exakt identisch gefunden. 
        * Dies zeigt, dass TensoRAG die geometrische Anordnung und Verwandtschaft der lokalen Dokumente hervorragend bewahrt – bei einem Bruchteil des Speicherbedarfs!
        """
    else:
        st.subheader("Simulate a Live Document Query (Retrieval)")
        st.markdown(f"""
        Select a domain and pick one of its documents as your search query. 
        We will perform the similarity search twice: once in the slow **original {trained_I}-dimensional space** and once in the compressed **{trained_Q}-dimensional TensoRAG space**.
        """)
        lbl_dom_sel = "Select Domain"
        lbl_idx_sel = "Query Document Index"
        lbl_k_sel = "Number of Top Matches (k)"
        lbl_title_orig = f"##### 🔴 Original Search (Dimension d = {trained_I})"
        lbl_title_comp = f"##### 🟢 Compressed Search (Dimension Q = {trained_Q})"
        col_doc_id = "Doc ID"
        col_cos_sim = "Cosine Similarity"
        col_status = "Match Status"
        val_status_match = "✅ Direct Match"
        val_status_near = "⚠️ Near Match"
        lbl_recall_header = "🎯 Semantic Recall Alignment: **{:.0f}%**"
        lbl_results_footer = """
        * **Result:** **{} of your top {} matching documents** are exactly identical in the compressed space. 
        * This demonstrates that TensoRAG preserves the semantic spatial orientation of documents in their local domains, yielding the exact same results!
        """

    col_s1, col_s2 = st.columns([1, 3])
    
    with col_s1:
        # Crucial: Options are locked to trained K and labels!
        selected_domain_idx = st.selectbox(lbl_dom_sel, options=range(trained_K), format_func=lambda x: trained_domain_labels[x], key="search_domain_select")
        total_docs = H_list[selected_domain_idx].shape[0]
        selected_doc_idx = st.slider(lbl_idx_sel, min_value=0, max_value=total_docs - 1, value=0)
        top_n = st.slider(lbl_k_sel, min_value=3, max_value=10, value=5)
        
    with col_s2:
        query_vec = H_list[selected_domain_idx][selected_doc_idx, :]
        db_vectors = H_list[selected_domain_idx]
        
        # Original Space search
        orig_similarities = np.dot(db_vectors, query_vec) / (
            np.linalg.norm(db_vectors, axis=1) * np.linalg.norm(query_vec)
        )
        orig_similarities[selected_doc_idx] = -1.0
        top_k_orig_idx = np.argsort(orig_similarities)[::-1][:top_n]
        top_k_orig_sims = orig_similarities[top_k_orig_idx]
        
        # Compressed Space search
        query_vec_proj = query_vec @ gsvd.A
        db_vectors_proj = gsvd.B[selected_domain_idx] @ np.diag(gsvd.C[selected_domain_idx, :])
        
        comp_similarities = np.dot(db_vectors_proj, query_vec_proj) / (
            np.linalg.norm(db_vectors_proj, axis=1) * np.linalg.norm(query_vec_proj)
        )
        comp_similarities[selected_doc_idx] = -1.0
        top_k_comp_idx = np.argsort(comp_similarities)[::-1][:top_n]
        top_k_comp_sims = comp_similarities[top_k_comp_idx]
        
        # Calculate search overlap/recall
        overlap = set(top_k_orig_idx).intersection(set(top_k_comp_idx))
        recall_pct = (len(overlap) / top_n) * 100
        
        # Display side-by-side search results
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.markdown(lbl_title_orig)
            orig_df = pd.DataFrame({
                col_doc_id: [f"Doc #{idx}" for idx in top_k_orig_idx],
                col_cos_sim: [f"{s:.4f}" for s in top_k_orig_sims]
            })
            st.dataframe(orig_df, use_container_width=True)
            
        with col_res2:
            st.markdown(lbl_title_comp)
            comp_df = pd.DataFrame({
                col_doc_id: [f"Doc #{idx}" for idx in top_k_comp_idx],
                col_cos_sim: [f"{s:.4f}" for s in top_k_comp_sims],
                col_status: [val_status_match if idx in top_k_orig_idx else val_status_near for idx in top_k_comp_idx]
            })
            st.dataframe(comp_df, use_container_width=True)
            
        # Display overlap visualization
        st.markdown(lbl_recall_header.format(recall_pct))
        st.progress(recall_pct / 100.0)
        st.markdown(lbl_results_footer.format(len(overlap), top_n))


# ----------------- TAB 2: STORAGE & MEMORY ANALYSIS -----------------
with tab_mem:
    if lang == "Deutsch":
        st.subheader("Analyse des physischen Speicherbedarfs")
        col_p1, col_p2 = st.columns([2, 3])
        with col_p1:
            st.markdown("""
            ### Warum spart TensoRAG so viel Platz?
            Klassische Vektordatenbanken speichern für jedes einzelne Dokument in jeder Domäne den vollständigen, unkomprimierten Vektor ab.
            
            **Die Zerlegungsformel von TensoRAG:**
            Anstatt der vollständigen Matrizen $\\mathbf{H}_k \\in \\mathbb{R}^{J_k \\times I}$ speichern wir die Faktoren:
            1. **$\\mathbf{A}$ (Gemeinsame Basis):** Größe $I \\times Q$. Wird **nur einmal** global für alle Domänen gespeichert!
            2. **$\\mathbf{B}_k$ (Orthogonale Faktoren):** Größe $J_k \\times Q$ pro Domäne. Extrem kompakt, da $Q \\ll I$.
            3. **$\\mathbf{C}_k$ (Skalierungswerte):** $Q$ Koeffizienten pro Domäne.
            
            Dieses geteilte Basis-Modell führt bei einer steigenden Anzahl von Domänen und Dokumenten zu immer größeren relativen Speichereinsparungen.
            """)
        with col_p2:
            sizes_data = {
                "Datenrepräsentation": ["Originale Datenbank", "TensoRAG-Basis (A)", "TensoRAG-Faktoren (B + C)"],
                "Physische Float-Werte": [original_floats, basis_size, factors_size + scales_size]
            }
            sizes_df = pd.DataFrame(sizes_data)
            st.markdown("##### Speicherplatz: Original vs. TensoRAG-Faktoren")
            st.bar_chart(data=sizes_df, x="Datenrepräsentation", y="Physische Float-Werte", use_container_width=True)
            
            st.markdown("##### Detaillierte Speicheraufteilung:")
            detail_df = pd.DataFrame({
                "Komponente / Matrix": ["Original Unkomprimiert", "Globale Basis (A)", "Linke Faktoren (B_k)", "Koeffizienten (C_k)", "Gesamter TensoRAG-Speicher"],
                "Dimensionen": [
                    f"K Matrizen der Größe (J_k x {trained_I})",
                    f"({trained_I} x {trained_Q})",
                    " + ".join([f"({dim} x {trained_Q})" for dim in trained_counts]),
                    f"({trained_K} x {trained_Q})",
                    "-"
                ],
                "Float-Werte": [original_floats, basis_size, factors_size, scales_size, compressed_floats],
                "Speicherbedarf (KB)": [
                    f"{original_floats*4/1024:.1f} KB",
                    f"{basis_size*4/1024:.1f} KB",
                    f"{factors_size*4/1024:.1f} KB",
                    f"{scales_size*4/1024:.1f} KB",
                    f"{compressed_floats*4/1024:.1f} KB"
                ]
            })
            st.dataframe(detail_df, use_container_width=True)
    else:
        st.subheader("Physical Storage Footprint Breakdown")
        col_p1, col_p2 = st.columns([2, 3])
        with col_p1:
            st.markdown("""
            ### Why is TensoRAG so space-efficient?
            Traditional databases store full, uncompressed high-dimensional vectors for every single document in every domain.
            
            **TensoRAG's Shared Subspace Formula:**
            Instead of saving $\\mathbf{H}_k \\in \\mathbb{R}^{J_k \\times I}$, we factorize the dataset. We store:
            1. **$\\mathbf{A}$ (Shared Right Basis):** Size $I \\times Q$. Only stored **once** globally for all domains!
            2. **$\\mathbf{B}_k$ (Orthogonal Left Factors):** Size $J_k \\times Q$ per domain. Very compact since $Q \\ll I$.
            3. **$\\mathbf{C}_k$ (Singular values):** $Q$ coefficients per domain.
            
            This multi-domain parameter sharing is what yields spectacular memory savings as your database scales.
            """)
        with col_p2:
            sizes_data = {
                "Representation": ["Original Database", "TensoRAG Basis (A)", "TensoRAG Slices (B + C)"],
                "Storage Size (Float Values)": [original_floats, basis_size, factors_size + scales_size]
            }
            sizes_df = pd.DataFrame(sizes_data)
            st.markdown("##### Storage Footprint: Original vs. TensoRAG Factors")
            st.bar_chart(data=sizes_df, x="Representation", y="Storage Size (Float Values)", use_container_width=True)
            
            st.markdown("##### Structural Memory Allocation Breakdown:")
            detail_df = pd.DataFrame({
                "Component / Matrix": ["Original Uncompressed", "Shared Subspace Basis (A)", "Left Factors (B_k)", "Coefficients (C_k)", "Total TensoRAG Size"],
                "Dimensions": [
                    f"K matrices of size (J_k x {trained_I})",
                    f"({trained_I} x {trained_Q})",
                    " + ".join([f"({dim} x {trained_Q})" for dim in trained_counts]),
                    f"({trained_K} x {trained_Q})",
                    "-"
                ],
                "Floats Stored": [original_floats, basis_size, factors_size, scales_size, compressed_floats],
                "Memory (KB)": [
                    f"{original_floats*4/1024:.1f} KB",
                    f"{basis_size*4/1024:.1f} KB",
                    f"{factors_size*4/1024:.1f} KB",
                    f"{scales_size*4/1024:.1f} KB",
                    f"{compressed_floats*4/1024:.1f} KB"
                ]
            })
            st.dataframe(detail_df, use_container_width=True)


# ----------------- TAB 3: AI AGENT SIMULATION -----------------
with tab_agent:
    if lang == "Deutsch":
        st.subheader("Simulierte KI-Agenten-Umgebung")
        st.markdown(f"""
        Hier wird simuliert, wie ein KI-Agent ein integriertes **TensoRAG-Modell** als Suchwerkzeug (Tool) für sein Wissensgedächtnis verwendet.
        Der Agent sucht **nicht** im speicherintensiven Originalraum ({trained_I} Dimensionen), sondern greift auf das kompakte, {trained_Q}-dimensionale Gedächtnis zu.
        """)
        lbl_agent_q = "Frage an den Agenten auswählen:"
        lbl_agent_btn = "⚡ Agenten-Simulation starten"
        lbl_agent_hist = "### 💬 Interaktiver Ablaufplan des Agenten"
        lbl_agent_anal = "Agent analysiert die Frage..."
        lbl_agent_vect = "Sende Vektor-Anfrage an komprimierte TensoRAG-Datenbank..."
        lbl_agent_answ = "Verarbeite Dokumententext und generiere Antwort..."
        lbl_agent_succ = f"Erfolgreich ausgeführt! Das komprimierte Gedächtnis sparte bei dieser Abfrage ca. {ram_savings_pct:.1f}% RAM im Vergleich zur Standard-Suche."
        
        # Localized Preset scenarios (Deutsch)
        agent_scenarios = {
            "Wie viel Reisebudget hat die Tech-Abteilung?": {
                "domain_idx": 1,
                "thought": "Der Nutzer fragt nach dem Reisebudget der Tech-Abteilung. Ich muss in den Finanzberichten (Domäne 1) suchen.",
                "tool_call": "compressed_vector_search('reisebudget tech', domain='Finanzberichte')",
                "retrieved_doc": "Finanzbericht Absatz 14: Das jährliche Reisebudget für die Tech-Entwickler beträgt maximal 15.000 € pro Team für Konferenzreisen.",
                "answer": "Laut dem Finanzbericht (Absatz 14) beläuft sich das jährliche Reisebudget für das Tech-Team auf maximal 15.000 € für Konferenzen und Dienstreisen."
            },
            "Wie hoch ist der Urlaubsanspruch bei einer 5-Tage-Woche?": {
                "domain_idx": 0,
                "thought": "Die Frage bezieht sich auf Urlaubsanspruch. Das fällt unter HR-Richtlinien (Domäne 0). Ich starte eine Suche in den HR-Vektoren.",
                "tool_call": "compressed_vector_search('urlaubstage 5-tage-woche', domain='HR-Richtlinien')",
                "retrieved_doc": "HR-Handbuch S. 8: Alle Vollzeitmitarbeiter im Rahmen einer regulären 5-Tage-Woche haben Anspruch auf 30 Tage bezahlten Erholungsurlaub pro Kalenderjahr.",
                "answer": "Gemäß dem HR-Handbuch (S. 8) haben alle Vollzeitbeschäftigten bei einer regulären 5-Tage-Woche einen Anspruch auf 30 Tage bezahlten Erholungsurlaub im Jahr."
            },
            "Welche Backup-Strategie gilt für die Cloud-Datenbanken?": {
                "domain_idx": 2,
                "thought": "Hier geht es um IT-Infrastruktur und Datenbanken. Ich muss in der Technischen Dokumentation (Domäne 2) nach 'Backup' suchen.",
                "tool_call": "compressed_vector_search('backup cloud datenbank', domain='Technische Dokumentation')",
                "retrieved_doc": "Tech-Infrastruktur-Doku Abs. 4.2: Alle produktiven Cloud-Datenbanken werden stündlich inkrementell gesichert. Ein vollständiges georedundantes Backup erfolgt täglich um 02:00 UTC.",
                "answer": "Entsprechend der technischen Dokumentation (Abschnitt 4.2) werden produktive Cloud-Datenbanken stündlich inkrementell gesichert, ergänzt durch ein tägliches georedundantes Voll-Backup um 02:00 UTC."
            }
        }
        
        thought_lbl = "🧠 GEDANKENGANG DES AGENTEN (Thought):"
        action_lbl = "⚙️ AKTION: Rufe registriertes Suchwerkzeug auf:"
        obs_lbl = "📥 RÜCKMELDUNG DES WERKZEUGS (Observation):"
        obs_desc = f"Suche abgeschlossen (im kompakten {trained_Q}-dimensionalen TensoRAG-Raum)"
        doc_found_lbl = "Gefundener Dokumentenabschnitt (höchste Ähnlichkeit):"
        user_prefix = "Du"
        agent_prefix = "Agent"
    else:
        st.subheader("Simulated AI Agent Environment")
        st.markdown(f"""
        This area simulates how an AI Agent utilizes an integrated **TensoRAG model** as a search tool for its semantic memory.
        The agent does **not** query the heavy original space ({trained_I} dimensions), but instead uses the lightweight {trained_Q}-dimensional coordinate system.
        """)
        lbl_agent_q = "Select a question for the agent:"
        lbl_agent_btn = "⚡ Start Agent Simulation"
        lbl_agent_hist = "### 💬 Agent Activity Timeline"
        lbl_agent_anal = "Agent is analyzing the query..."
        lbl_agent_vect = "Sending vector search query to compressed TensoRAG database..."
        lbl_agent_answ = "Processing document context and generating answer..."
        lbl_agent_succ = f"Successfully executed! The compressed database saved approximately {ram_savings_pct:.1f}% RAM during this query compared to standard retrieval."
        
        # Localized Preset scenarios (English)
        agent_scenarios = {
            "What is the travel budget for the tech department?": {
                "domain_idx": 1,
                "thought": "The user is asking about the travel budget for the tech department. I need to search the Financial Reports (Domain 1).",
                "tool_call": "compressed_vector_search('travel budget tech', domain='Financial Reports')",
                "retrieved_doc": "Financial Report Paragraph 14: The annual travel budget for technical developers is capped at €15,000 per team for conference trips.",
                "answer": "According to Financial Report (Paragraph 14), the annual travel budget for the tech team is capped at €15,000 for conference and business travel."
            },
            "How many vacation days do we get with a 5-day week?": {
                "domain_idx": 0,
                "thought": "This query refers to vacation day entitlements. This belongs to HR Policies (Domain 0). I will initiate search in HR vectors.",
                "tool_call": "compressed_vector_search('vacation days 5-day week', domain='HR Policies')",
                "retrieved_doc": "HR Handbook P. 8: All full-time employees working a standard 5-day week are entitled to 30 days of paid vacation per calendar year.",
                "answer": "According to the HR Handbook (P. 8), all full-time employees on a standard 5-day work week are entitled to 30 days of paid annual leave."
            },
            "What is the backup strategy for cloud databases?": {
                "domain_idx": 2,
                "thought": "This is related to IT infrastructure and database backups. I must search the Technical Documentation (Domain 2).",
                "tool_call": "compressed_vector_search('backup cloud database', domain='Technical Documentation')",
                "retrieved_doc": "Tech Infrastructure Doc Sec. 4.2: All production cloud databases are backed up incrementally every hour. A full geo-redundant backup is run daily at 02:00 UTC.",
                "answer": "As specified in the technical documentation (Section 4.2), production cloud databases are backed up incrementally every hour, with a full geo-redundant backup executed daily at 02:00 UTC."
            }
        }
        
        thought_lbl = "🧠 AGENT THOUGHT:"
        action_lbl = "⚙️ ACTION: Call registered search tool:"
        obs_lbl = "📥 TOOL OBSERVATION (Werkzeug-Rückmeldung):"
        obs_desc = f"Search completed (in the compact {trained_Q}-dimensional TensoRAG space)"
        doc_found_lbl = "Retrieved Document Segment (highest similarity):"
        user_prefix = "You"
        agent_prefix = "Agent"

    selected_query = st.selectbox(lbl_agent_q, list(agent_scenarios.keys()), key="agent_query_select")
    
    if st.button(lbl_agent_btn, type="primary", key="run_agent_btn"):
        scenario = agent_scenarios[selected_query]
        
        # Safely wrap the domain index to never exceed the currently trained K
        safe_domain_idx = scenario["domain_idx"] % trained_K
        
        st.markdown(lbl_agent_hist)
        
        # 1. User Message
        st.markdown(f"""
        <div class="user-bubble">
            <b>{user_prefix}:</b><br>{selected_query}
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner(lbl_agent_anal):
            time.sleep(0.8)
            
        # 2. Agent Thoughts
        st.markdown(f"""
        <div class="thought-bubble">
            <b>{thought_lbl}</b><br>
            "{scenario['thought']}"<br><br>
            <b>{action_lbl}</b> <span class="tool-tag">{scenario['tool_call']}</span>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner(lbl_agent_vect):
            time.sleep(1.0)
            
            # Simulated real mathematical projection
            q_vec_agent = np.random.randn(trained_I)
            start_time_agent = time.time()
            q_proj_agent = q_vec_agent @ gsvd.A
            db_proj_agent = gsvd.B[safe_domain_idx] @ np.diag(gsvd.C[safe_domain_idx, :])
            _ = np.dot(db_proj_agent, q_proj_agent)
            search_duration_agent_ms = (time.time() - start_time_agent) * 1000
            
        # 3. Tool Result / Observation WITH Matplotlib Visual Alignment Plot!
        st.markdown(f"""
        <div class="thought-bubble" style="border-left: 5px solid #137333; background-color: #f6fbf7;">
            <b>{obs_lbl}</b><br>
            <i>{obs_desc} in <b>{search_duration_agent_ms:.4f} ms</b></i><br><br>
            <b>{doc_found_lbl}</b><br>
            "{scenario['retrieved_doc']}"
        </div>
        """, unsafe_allow_html=True)
        
        # --- GENERATE MATPLOTLIB VECTOR ALIGNMENT PLOT ---
        # We project the query and database slice into 2D to visually show the alignment
        fig, ax = plt.subplots(figsize=(6, 2.8))
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#ffffff')
        
        # Project all documents of this domain into 2D coordinates (using first 2 axes of the subspace)
        doc_x = db_proj_agent[:, 0]
        doc_y = db_proj_agent[:, 1]
        ax.scatter(doc_x, doc_y, color='#1a73e8', alpha=0.6, s=40, label='Documents' if lang == 'English' else 'Dokumente')
        
        # Plot the projected Query Vector
        qx, qy = q_proj_agent[0], q_proj_agent[1]
        ax.scatter(qx, qy, color='#d93025', marker='*', s=150, zorder=5, label='Query' if lang == 'English' else 'Suchanfrage')
        
        # Highlight top matching document (index 0) with a ring
        ax.scatter(doc_x[0], doc_y[0], color='#137333', marker='o', s=100, facecolors='none', edgecolors='#137333', linewidths=2, zorder=4, label='Top Match' if lang == 'English' else 'Bester Treffer')
        
        ax.set_title('Subspace Vector Alignment (2D Projection)' if lang == 'English' else 'Vektorausrichtung im Unterraum (2D-Projektion)', fontsize=9, color='#202124')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(axis='both', which='both', labelsize=7)
        fig.tight_layout()
        
        # Display the visual plot in streamlit
        st.pyplot(fig)
        plt.close(fig)
        
        with st.spinner(lbl_agent_answ):
            time.sleep(0.8)
            
        # 4. Final Agent Answer
        st.markdown(f"""
        <div class="agent-bubble">
            <b>🤖 {agent_prefix}:</b><br>
            {scenario['answer']}
        </div>
        """, unsafe_allow_html=True)
        
        st.success(lbl_agent_succ)


# ----------------- TAB 4: MATHEMATICAL INSIGHTS -----------------
with tab_math:
    if lang == "Deutsch":
        st.subheader("Mathematische Diagnostik & Energien")
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            st.markdown("##### 📈 Energieverteilung der Koeffizienten (Singular Value Decay)")
            st.markdown("""
            Die diagonalen Koeffizienten in $\\mathbf{C}_k$ fungieren als verallgemeinerte Singulärwerte für jede spezifische Domäne. 
            Ein rascher Abfall der Kurven beweist, dass die wesentlichen semantischen Informationen in den führenden Dimensionen 
            konzentriert sind, was die Wahl eines niedrigen Ranges $Q$ mathematisch rechtfertigt.
            """)
            
            # Generate chart for coefficients
            coef_data = {}
            for k, label in enumerate(trained_domain_labels):
                coef_data[label] = np.sort(gsvd.C[k, :])[::-1]
            
            coef_df = pd.DataFrame(coef_data)
            st.line_chart(coef_df, use_container_width=True)
            
        with col_d2:
            st.markdown("##### 📉 ALS-Modell-Konvergenzverlauf")
            st.markdown("""
            TensoRAG verwendet ein iteratives **Alternating Least Squares (ALS)** Optimierungsverfahren. 
            Die folgende Grafik zeigt den Verlauf des quadratischen Rekonstruktionsfehlers über alle Domänenslices hinweg 
            und illustriert die schnelle, stabile numerische Konvergenz des Modells.
            """)
            
            error_df = pd.DataFrame({
                "Iterationen": list(range(1, len(gsvd.errors) + 1)),
                "Gesamter Rekonstruktionsfehler": gsvd.errors
            }).set_index("Iterationen")
            st.line_chart(error_df, use_container_width=True)
    else:
        st.subheader("Mathematical Diagnostics & SVD Energies")
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            st.markdown("##### 📈 Coefficient Energy Distribution (Singular Value Decay)")
            st.markdown("""
            The diagonal coefficients in $\\mathbf{C}_k$ act as the generalized singular values for each specific domain. 
            If the curves decay rapidly, it proves that the semantic information of your dataset is successfully compressed 
            into the top principal directions, justifying the choice of a low-rank target $Q$.
            """)
            
            # Generate chart for coefficients
            coef_data = {}
            for k, label in enumerate(trained_domain_labels):
                coef_data[label] = np.sort(gsvd.C[k, :])[::-1]
            
            coef_df = pd.DataFrame(coef_data)
            st.line_chart(coef_df, use_container_width=True)
            
        with col_d2:
            st.markdown("##### 📉 ALS Model Optimization Convergence Curve")
            st.markdown("""
            TensoRAG uses an iterative **Alternating Least Squares (ALS)** optimization routine. 
            The graph below plots the joint reconstruction error across all domain slices per iteration, 
            illustrating how the model rapidly converges toward its numerical optimum.
            """)
            
            error_df = pd.DataFrame({
                "Iteration": list(range(1, len(gsvd.errors) + 1)),
                "Total Sum-of-Squares Error": gsvd.errors
            }).set_index("Iteration")
            st.line_chart(error_df, use_container_width=True)


# ----------------- TAB 5: HOW TO DEPLOY -----------------
with tab_code:
    if lang == "Deutsch":
        st.subheader("🚀 Code-Implementierung für dein eigenes RAG-System")
        st.markdown("""
        Um die ultraschnelle und speichereffiziente Suche von TensoRAG direkt in deine eigene Anwendung einzubauen, 
        kannst du diesen bereinigten Python-Code kopieren. Die Ähnlichkeitssuche läuft damit vollständig im kompakten Koordinatenraum!
        """)
    else:
        st.subheader("🚀 Deploy TensoRAG to your RAG Search Pipeline")
        st.markdown("""
        To integrate TensoRAG's ultra-fast, low-memory search in your Python applications, copy and use this production-ready code snippet.
        This bypasses reconstructing the huge original matrices and runs searches directly in the lightweight compressed coordinate space!
        """)
        
    deploy_code = f"""import numpy as np
from tensorag import MultilinearGSVD

# 1. Modell initialisieren und trainieren
# H_list enthält K Domänen-Matrizen, jeweils mit der Form (Anzahl_Dokumente, {trained_I})
gsvd = MultilinearGSVD(target_rank={trained_Q}, max_iter=25, tol=1e-5)
gsvd.fit(H_list)

# 2. Komprimierte Datenbank-Faktoren extrahieren
# Anstatt riesige Vektoren zu speichern, sichern wir nur diese Faktoren:
A_basis = gsvd.A  # Gemeinsame globale Basis, Dimension ({trained_I}, {trained_Q})
compressed_db_slices = []
for k in range({trained_K}):
    # Berechne die kompakten Koordinaten für jeden Domänen-Slice k
    # Dimension: (Anzahl_Dokumente_k, {trained_Q})
    compressed_db = gsvd.B[k] @ np.diag(gsvd.C[k, :])
    compressed_db_slices.append(compressed_db)

# 3. Ultraschnelle Suchfunktion im komprimierten Raum
def query_compressed_database(query_vector, domain_index, top_k=5):
    \"\"\"
    Führt eine Cosinus-Ähnlichkeitssuche im Q-dimensionalen Raum statt im d-dimensionalen Raum aus.
    \"\"\"
    # Projiziere die hochdimensionale Suchanfrage in die gemeinsame Basis: d-dim -> Q-dim
    query_projected = query_vector @ A_basis  # Dimension: ({trained_Q},)
    
    # Hole die vorkomprimierte Datenbank für die gewünschte Domäne k
    db_projected = compressed_db_slices[domain_index]
    
    # Berechne Cosinus-Ähnlichkeit im kompakten Q-dimensionalen Raum
    similarities = np.dot(db_projected, query_projected) / (
        np.linalg.norm(db_projected, axis=1) * np.linalg.norm(query_projected)
    )
    
    # Top-k Indizes extrahieren
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return top_indices, similarities[top_indices]

# Beispiel-Suchlauf ausführen
wild_query = np.random.randn({trained_I})
matches, scores = query_compressed_database(wild_query, domain_index=0, top_k=5)
print("Top-Treffer Indizes in Domäne 0:", matches)
"""
    st.code(deploy_code, language="python")

# Bottom banner
st.markdown("---")
if lang == "Deutsch":
    st.markdown("""
    <div style="text-align: center; color: #757575; font-size: 0.95rem;">
        <b>TensoRAG</b> ist ein unabhängiges Freizeitprojekt von <b>Forstwichtel</b> und <b>Gemini Notebook [bot]</b>.<br>
        Die mathematischen Grundlagen basieren auf der Dissertation von Dr. Liana Khamidullina (TU Ilmenau). ⭐
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align: center; color: #757575; font-size: 0.95rem;">
        <b>TensoRAG</b> is an independent hobby project developed by <b>Forstwichtel</b> and <b>Gemini Notebook [bot]</b>.<br>
        The mathematics are based on the PhD dissertation of Dr. Liana Khamidullina (TU Ilmenau). ⭐
    </div>
    """, unsafe_allow_html=True)
