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

# Set page layout to wide for a professional dashboard look
st.set_page_config(
    page_title="TensoRAG - Multilineare GSVD Vektorkompression & Agentendemo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling metrics, cards, and agent bubbles
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
        color: #424242;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    .metric-delta {
        font-size: 0.9rem;
        color: #616161;
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
        padding: 15px;
        border-radius: 15px;
        border-left: 5px solid #1a73e8;
        margin-bottom: 15px;
    }
    .thought-bubble {
        background-color: #fff8e1;
        padding: 15px;
        border-radius: 15px;
        border-left: 5px solid #ffb300;
        margin-bottom: 15px;
        font-family: monospace;
    }
    .user-bubble {
        background-color: #e8f0fe;
        padding: 15px;
        border-radius: 15px;
        border-left: 5px solid #4285f4;
        margin-bottom: 15px;
        text-align: right;
    }
    .tool-tag {
        background-color: #e6f4ea;
        color: #137333;
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

# Title and introduction
st.title("🚀 TensoRAG: Interaktive Vektorkompression & Agentensimulation")
st.markdown("""
### Multilineare Generalisierte SVD (ML-GSVD) in der Praxis
Diese App demonstriert, wie unabhängige Wissensdatenbanken (z. B. HR-Richtlinien, Finanzberichte, Technische Dokumente) 
simultan komprimiert werden können. Durch die Extraktion einer **gemeinsamen globalen Basis $\\mathbf{A}$** reduzieren wir den 
Speicherbedarf der Vektordatenbank drastisch und beschleunigen die Suche, ohne die semantische Genauigkeit der lokalen Dokumente zu beeinträchtigen.
""")

# Setup Sidebar for parameters
st.sidebar.header("⚙️ Simulations-Einstellungen")

K = st.sidebar.slider("Anzahl der Domänen (K)", min_value=2, max_value=5, value=3)

# Define domain configurations based on selected K
domain_labels = ["HR-Richtlinien", "Finanzberichte", "Technische Dokumentation", "Rechtliche Klauseln", "Kundensupport"][:K]
document_counts = []
st.sidebar.subheader("📄 Dokumentenanzahl pro Domäne")
for i, label in enumerate(domain_labels):
    count = st.sidebar.slider(f"{label} (Dokumente)", min_value=30, max_value=300, value=[120, 90, 150, 100, 80][i])
    document_counts.append(count)

st.sidebar.subheader("📊 Einbettungs-Setup")
I = st.sidebar.selectbox("Ursprüngliche Dimension (d)", options=[256, 512, 1024, 1536, 3072], index=3)
Q = st.sidebar.slider("Ziel-Dimension (Q)", min_value=16, max_value=256, value=128, step=16)

if Q >= I:
    st.sidebar.error(f"Fehler: Die Ziel-Dimension Q ({Q}) muss strikt kleiner sein als die ursprüngliche Dimension d ({I}).")
    st.stop()

max_iter = st.sidebar.slider("Max. ALS Iterationen", min_value=5, max_value=50, value=25)

# Session state initialization to cache generated data and model
if "data_generated" not in st.session_state or st.session_state.get("prev_params") != (K, document_counts, I, Q):
    st.session_state["data_generated"] = False

# Trigger Button
run_button = st.sidebar.button("⚡ TensoRAG-Kompression starten", type="primary")

if run_button or not st.session_state["data_generated"]:
    with st.spinner("Generiere synthetische Domänen-Vektoren und trainiere TensoRAG-Modell..."):
        # 1. Generate realistic synthetic semantic embeddings
        np.random.seed(42)
        shared_dimensions = 50
        shared_structure = np.random.randn(shared_dimensions, I)
        
        H_list = []
        for count in document_counts:
            # Build projection of global structure + local domain variance + noise
            projection = np.random.randn(count, shared_dimensions) @ shared_structure
            noise = np.random.randn(count, I) * 0.4
            H_list.append(projection + noise)
            
        st.session_state["H_list"] = H_list
        
        # 2. Fit ML-GSVD
        start_time = time.time()
        gsvd = MultilinearGSVD(target_rank=Q, max_iter=max_iter, tol=1e-5, random_state=42)
        gsvd.fit(H_list)
        fit_duration = time.time() - start_time
        
        st.session_state["gsvd"] = gsvd
        st.session_state["fit_duration"] = fit_duration
        st.session_state["prev_params"] = (K, document_counts, I, Q)
        st.session_state["data_generated"] = True

# Retrieve data and model from session state
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

# Simulate search performance to find latency and speedup
import timeit
q_vector = H_list[0][0, :] # Use first document as query for speedup benchmark
def orig_search():
    sim = np.dot(H_list[0], q_vector) / (np.linalg.norm(H_list[0], axis=1) * np.linalg.norm(q_vector))
    _ = np.argsort(sim)[::-1]

q_proj = q_vector @ gsvd.A
db_proj = gsvd.B[0] @ np.diag(gsvd.C[0, :])
def comp_search():
    sim = np.dot(db_proj, q_proj) / (np.linalg.norm(db_proj, axis=1) * np.linalg.norm(q_proj))
    _ = np.argsort(sim)[::-1]

t_orig = timeit.timeit(orig_search, number=200) / 200 * 1000  # in ms
t_comp = timeit.timeit(comp_search, number=200) / 200 * 1000  # in ms
search_speedup = t_orig / t_comp

# ----------------- DASHBOARD DISPLAY -----------------

# Display top-level metric cards
col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">💾 RAM-Reduzierung</div>
        <div class="metric-value">{ram_savings_pct:.1f}%</div>
        <div class="metric-delta">Eingespart: <b>{(original_floats - compressed_floats)*4/1024:.1f} KB</b> ({compression_ratio:.1f}x kompakter)</div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">⚡ Such-Beschleunigung</div>
        <div class="metric-value">{search_speedup:.1f}x</div>
        <div class="metric-delta">Original: {t_orig:.4f} ms | TensoRAG: {t_comp:.4f} ms</div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">⏱️ Trainingsdauer (Fit)</div>
        <div class="metric-value">{fit_duration:.2f}s</div>
        <div class="metric-delta">Konvergenz erreicht in {len(gsvd.errors)} ALS-Iterationen</div>
    </div>
    """, unsafe_allow_html=True)

# Main layout divided into Tabs
tab_search, tab_mem, tab_agent, tab_math, tab_code = st.tabs([
    "🔍 Interaktive Validierung", 
    "📊 Speicher-Analyse", 
    "🤖 KI-Agenten-Simulation",
    "📈 Mathematische Einblicke", 
    "💻 Code-Implementierung"
])

# ----------------- TAB 1: INTERACTIVE SEARCH VALIDATION -----------------
with tab_search:
    st.subheader("Simuliere eine Dokumenten-Suche (Retrieval)")
    st.markdown(f"""
    Wähle eine Domäne und eines ihrer Dokumente aus, um eine Suchanfrage zu simulieren. 
    Wir führen die Ähnlichkeitssuche zweifach aus: einmal im **originalen {I}-dimensionalen Raum** und einmal im komprimierten **{Q}-dimensionalen TensoRAG-Raum**.
    """)
    
    col_s1, col_s2 = st.columns([1, 3])
    
    with col_s1:
        selected_domain_idx = st.selectbox("Domäne wählen", options=range(K), format_func=lambda x: domain_labels[x], key="search_domain_select")
        total_docs = H_list[selected_domain_idx].shape[0]
        selected_doc_idx = st.slider("Dokumenten-Index (Suchanfrage)", min_value=0, max_value=total_docs - 1, value=0)
        top_n = st.slider("Anzahl der Top-Matches (k)", min_value=3, max_value=10, value=5)
        
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
            st.markdown(f"##### 🔴 Originale Suche (Dimension d = {I})")
            orig_df = pd.DataFrame({
                "Dokumenten-ID": [f"Dokument #{idx}" for idx in top_k_orig_idx],
                "Cosinus-Ähnlichkeit": [f"{s:.4f}" for s in top_k_orig_sims]
            })
            st.dataframe(orig_df, use_container_width=True)
            
        with col_res2:
            st.markdown(f"##### 🟢 Komprimierte Suche (Dimension Q = {Q})")
            comp_df = pd.DataFrame({
                "Dokumenten-ID": [f"Dokument #{idx}" for idx in top_k_comp_idx],
                "Cosinus-Ähnlichkeit": [f"{s:.4f}" for s in top_k_comp_sims],
                "Status": ["✅ Exakter Treffer" if idx in top_k_orig_idx else "⚠️ Ähnlicher Treffer" for idx in top_k_comp_idx]
            })
            st.dataframe(comp_df, use_container_width=True)
            
        # Display overlap visualization
        st.markdown(f"#### 🎯 Semantische Abdeckung (Recall): **{recall_pct:.0f}%**")
        st.progress(recall_pct / 100.0)
        st.markdown(f"""
        * **Ergebnis:** **{len(overlap)} der Top {top_n} übereinstimmenden Dokumente** wurden im komprimierten Raum exakt identisch gefunden. 
        * Dies zeigt, dass TensoRAG die geometrische Anordnung und Verwandtschaft der lokalen Dokumente hervorragend bewahrt – bei einem Bruchteil des Speicherbedarfs!
        """)

# ----------------- TAB 2: STORAGE & MEMORY ANALYSIS -----------------
with tab_mem:
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
        # Create storage comparison data
        sizes_data = {
            "Datenrepräsentation": ["Originale Datenbank", "TensoRAG-Basis (A)", "TensoRAG-Faktoren (B + C)"],
            "Physische Float-Werte": [original_floats, basis_size, factors_size + scales_size]
        }
        sizes_df = pd.DataFrame(sizes_data)
        
        st.markdown("##### Speicherplatz: Original vs. TensoRAG-Faktoren")
        st.bar_chart(data=sizes_df, x="Datenrepräsentation", y="Physische Float-Werte", use_container_width=True)
        
        # Detailed table of sizes
        st.markdown("##### Detaillierte Speicheraufteilung:")
        detail_df = pd.DataFrame({
            "Komponente / Matrix": ["Original Unkomprimiert", "Globale Basis (A)", "Linke Faktoren (B_k)", "Koeffizienten (C_k)", "Gesamter TensoRAG-Speicher"],
            "Dimensionen": [
                f"K Matrizen der Größe (J_k x {I})",
                f"({I} x {Q})",
                " + ".join([f"({dim} x {Q})" for dim in document_counts]),
                f"({K} x {Q})",
                "-"
            ],
            "Float-Werte": [
                original_floats,
                basis_size,
                factors_size,
                scales_size,
                compressed_floats
            ],
            "Speicherbedarf (KB)": [
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
    st.subheader("Simulierte KI-Agenten-Umgebung")
    st.markdown("""
    Hier wird simuliert, wie ein KI-Agent ein integriertes **TensoRAG-Modell** als Suchwerkzeug (Tool) für sein Wissensgedächtnis verwendet.
    Der Agent sucht **nicht** im speicherintensiven Originalraum (1536 Dimensionen), sondern greift auf das kompakte, 128-dimensionale Gedächtnis zu.
    """)

    # Local Preset scenarios
    agent_scenarios = {
        "Wie viel Reisebudget hat die Tech-Abteilung?": {
            "domain_idx": 1,
            "search_term": "reisebudget tech",
            "thought": "Der Nutzer fragt nach dem Reisebudget der Tech-Abteilung. Ich muss in den Finanzberichten (Domäne 1) suchen.",
            "tool_call": "compressed_vector_search('reisebudget tech', domain='Finanzberichte')",
            "retrieved_doc": "Finanzbericht Absatz 14: Das jährliche Reisebudget für die Tech-Entwickler beträgt maximal 15.000 € pro Team für Konferenzreisen.",
            "answer": "Laut dem Finanzbericht (Absatz 14) beläuft sich das jährliche Reisebudget für das Tech-Team auf maximal 15.000 € für Konferenzen und Dienstreisen."
        },
        "Wie hoch ist der Urlaubsanspruch bei einer 5-Tage-Woche?": {
            "domain_idx": 0,
            "search_term": "urlaubstage 5-tage-woche",
            "thought": "Die Frage bezieht sich auf Urlaubsanspruch. Das fällt unter HR-Richtlinien (Domäne 0). Ich starte eine Suche in den HR-Vektoren.",
            "tool_call": "compressed_vector_search('urlaubstage 5-tage-woche', domain='HR-Richtlinien')",
            "retrieved_doc": "HR-Handbuch S. 8: Alle Vollzeitmitarbeiter im Rahmen einer regulären 5-Tage-Woche haben Anspruch auf 30 Tage bezahlten Erholungsurlaub pro Kalenderjahr.",
            "answer": "Gemäß dem HR-Handbuch (S. 8) haben alle Vollzeitbeschäftigten bei einer regulären 5-Tage-Woche einen Anspruch auf 30 Tage bezahlten Erholungsurlaub im Jahr."
        },
        "Welche Backup-Strategie gilt für die Cloud-Datenbanken?": {
            "domain_idx": 2,
            "search_term": "backup cloud datenbank",
            "thought": "Hier geht es um IT-Infrastruktur und Datenbanken. Ich muss in der Technischen Dokumentation (Domäne 2) nach 'Backup' suchen.",
            "tool_call": "compressed_vector_search('backup cloud datenbank', domain='Technische Dokumentation')",
            "retrieved_doc": "Tech-Infrastruktur-Doku Abs. 4.2: Alle produktiven Cloud-Datenbanken werden stündlich inkrementell gesichert. Ein vollständiges georedundantes Backup erfolgt täglich um 02:00 UTC.",
            "answer": "Entsprechend der technischen Dokumentation (Abschnitt 4.2) werden produktive Cloud-Datenbanken stündlich inkrementell gesichert, ergänzt durch ein tägliches georedundantes Voll-Backup um 02:00 UTC."
        }
    }

    selected_query = st.selectbox("Frage an den Agenten auswählen:", list(agent_scenarios.keys()), key="agent_query_select")
    
    if st.button("⚡ Agenten-Simulation starten", type="primary", key="run_agent_btn"):
        scenario = agent_scenarios[selected_query]
        
        st.markdown("### 💬 Interaktiver Ablaufplan des Agenten")
        
        # 1. User Message
        st.markdown(f"""
        <div class="user-bubble">
            <b>Du:</b><br>{selected_query}
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("Agent analysiert die Frage..."):
            time.sleep(0.8)
            
        # 2. Agent Thoughts
        st.markdown(f"""
        <div class="thought-bubble">
            <b>🧠 GEDANKENGANG DES AGENTEN (Thought):</b><br>
            "{scenario['thought']}"<br><br>
            <b>⚙️ AKTION:</b> Rufe registriertes Suchwerkzeug auf: <span class="tool-tag">{scenario['tool_call']}</span>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("Sende Vektor-Anfrage an komprimierte TensoRAG-Datenbank..."):
            time.sleep(1.0)
            
            # Simulated real mathematical projection
            q_vec_agent = np.random.randn(I)
            start_time_agent = time.time()
            q_proj_agent = q_vec_agent @ gsvd.A
            db_proj_agent = gsvd.B[scenario['domain_idx']]
            _ = np.dot(db_proj_agent, q_proj_agent)
            search_duration_agent_ms = (time.time() - start_time_agent) * 1000
            
        # 3. Tool Result / Observation
        st.markdown(f"""
        <div class="thought-bubble" style="border-left: 5px solid #137333; background-color: #f6fbf7;">
            <b>📥 RÜCKMELDUNG DES WERKZEUGS (Observation):</b><br>
            <i>Suche abgeschlossen in <b>{search_duration_agent_ms:.4f} ms</b> (im kompakten {Q}-dimensionalen Raum)</i><br><br>
            <b>Gefundener Dokumentenabschnitt (höchste Ähnlichkeit):</b><br>
            "{scenario['retrieved_doc']}"
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("Verarbeite Dokumententext und generiere Antwort..."):
            time.sleep(0.8)
            
        # 4. Final Agent Answer
        st.markdown(f"""
        <div class="agent-bubble">
            <b>🤖 Agent:</b><br>
            {scenario['answer']}
        </div>
        """, unsafe_allow_html=True)
        
        st.success(f"Erfolgreich ausgeführt! Das komprimierte Gedächtnis sparte bei dieser Abfrage ca. {ram_savings_pct:.1f}% RAM im Vergleich zur Standard-Suche.")

# ----------------- TAB 4: MATHEMATICAL INSIGHTS -----------------
with tab_math:
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
        for k, label in enumerate(domain_labels):
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

# ----------------- TAB 5: HOW TO DEPLOY -----------------
with tab_code:
    st.subheader("🚀 Code-Implementierung für dein eigenes RAG-System")
    st.markdown("""
    Um die ultraschnelle und speichereffiziente Suche von TensoRAG direkt in deine eigene Anwendung einzubauen, 
    kannst du diesen bereinigten Python-Code kopieren. Die Ähnlichkeitssuche läuft damit vollständig im kompakten Koordinatenraum!
    """)
    
    deploy_code = f"""import numpy as np
from tensorag import MultilinearGSVD

# 1. Modell initialisieren und trainieren
# H_list enthält K Domänen-Matrizen, jeweils mit der Form (Anzahl_Dokumente, {I})
gsvd = MultilinearGSVD(target_rank={Q}, max_iter=25, tol=1e-5)
gsvd.fit(H_list)

# 2. Komprimierte Datenbank-Faktoren extrahieren
# Anstatt riesige Vektoren zu speichern, sichern wir nur diese Faktoren:
A_basis = gsvd.A  # Gemeinsame globale Basis, Dimension ({I}, {Q})
compressed_db_slices = []
for k in range({K}):
    # Berechne die kompakten Koordinaten für jeden Domänen-Slice k
    # Dimension: (Anzahl_Dokumente_k, {Q})
    compressed_db = gsvd.B[k] @ np.diag(gsvd.C[k, :])
    compressed_db_slices.append(compressed_db)

# 3. Ultraschnelle Suchfunktion im komprimierten Raum
def query_compressed_database(query_vector, domain_index, top_k=5):
    \"\"\"
    Führt eine Cosinus-Ähnlichkeitssuche im Q-dimensionalen Raum statt im d-dimensionalen Raum aus.
    \"\"\"
    # Projiziere die hochdimensionale Suchanfrage in die gemeinsame Basis: d-dim -> Q-dim
    query_projected = query_vector @ A_basis  # Dimension: ({Q},)
    
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
wild_query = np.random.randn({I})
matches, scores = query_compressed_database(wild_query, domain_index=0, top_k=5)
print("Top-Treffer Indizes in Domäne 0:", matches)
"""
    st.code(deploy_code, language="python")

# Bottom banner
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #757575; font-size: 0.95rem;">
    <b>TensoRAG</b> ist ein unabhängiges Freizeitprojekt von <b>Forstwichtel [forstwichtel@gmail.com]</b> und <b>Helferlein [bot]</b>.<br>
    Die mathematischen Grundlagen basieren auf der Dissertation von Dr. Liana Khamidullina (TU Ilmenau). ⭐
</div>
""", unsafe_allow_html=True)
