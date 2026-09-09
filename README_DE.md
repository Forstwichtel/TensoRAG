# TensoRAG: Multilineare Verallgemeinerte Singulärwertzerlegung (ML-GSVD)

<p align="center">
  <img src="tensorag_github_logo.png" alt="TensoRAG Logo" width="800">
</p>

Eine hochperformante Python/NumPy-Implementierung der **Multilinearen Verallgemeinerten Singulärwertzerlegung (ML-GSVD)**, basierend auf der bahnbrechenden mathematischen Forschung von **Dr. Liana Khamidullina** und **Prof. Dr. Martin Haardt** (Technische Universität Ilmenau).

[![Lizenz: PolyForm NonCommercial 1.0.0](https://img.shields.io/badge/Lizenz-PolyForm_NonCommercial_1.0.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Interaktive Demo](https://img.shields.io/badge/Streamlit-Interaktive_Demo-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://tensorag.streamlit.app/)

> ⚠️ **Haftungsausschluss (Disclaimer):** Dies ist ein unabhängiges, privates Hobbyprojekt, das von @Forstwichtel entwickelt wurde. Es steht in keiner offiziellen Verbindung zu den Autoren der wissenschaftlichen Arbeit oder der Technischen Universität Ilmenau, wurde von diesen nicht geprüft und wird nicht offiziell unterstützt. Dieses Repository dient ausschließlich als unabhängige Implementierung der veröffentlichten akademischen Ergebnisse.

---

🌐 *Read this documentation in [English](README.md).*

---

## 👥 Autoren & Mitwirkende

*   **Hauptautor:** Forstwichtel (Pseudonym)
*   **Ko-Autor:** Gemini Notebook [bot]

---

## 📌 Einführung & Mathematischer Hintergrund

**TensoRAG** implementiert die **ML-GSVD**, einen bedeutenden mathematischen Durchbruch, der die klassische verallgemeinerte SVD (GSVD) für zwei Matrizen auf eine beliebige Anzahl von $K \ge 2$ Matrizen erweitert (die als Schichten eines 3D-Tensors $\mathcal{H}$ betrachtet werden) und eine gemeinsame Spaltendimension teilt.

Im Gegensatz zu früheren Ansätzen (wie der HO-GSVD) bewahrt die ML-GSVD die Kernkonzepte der klassischen GSVD, insbesondere die **exakte Spaltenorthogonalität** der linken Faktormatrizen $\mathbf{B}_k$.

Mathematisch gesehen zerlegt das Verfahren ein Set von $K$ Matrizen $\mathbf{H}_k \in \mathbb{C}^{J_k \times I}$ gleichzeitig in:
$$\mathbf{H}_k \approx \mathbf{B}_k \cdot \mathbf{C}_k \cdot \mathbf{A}^H \quad \text{für } k = 1, \dots, K$$

*   **$\mathbf{A}^H \in \mathbb{C}^{Q \times I}$**: Die **gemeinsame Repräsentationsbasis** (die "Kernmatrix", die den globalen gemeinsamen Unterraum darstellt).
*   **$\mathbf{B}_k \in \mathbb{C}^{J_k \times Q}$**: Die **linken orthogonalen Faktoren** ($\mathbf{B}_k^H \mathbf{B}_k = \mathbf{I}$), welche die geometrische Struktur der einzelnen Datenquellen exakt bewahren.
*   **$\mathbf{C}_k \in \mathbb{R}^{Q \times Q}$**: Die **Koeffizientenmatrizen** (reelle, nicht-negative Diagonalmatrizen, welche die verallgemeinerten Singulärwerte für jede spezifische Schicht enthalten).

---

## 🚀 Hauptmerkmale

*   **Gleichzeitige Multi-Matrix-Faktorisierung:** Zerlegt $K \ge 2$ Matrizen mit unterschiedlichen Zeilendimensionen, aber einer gemeinsamen Spaltendimension.
*   **Echter Orthogonal-Procrustes-Solver:** Berechnet die orthogonalen Faktor-Updates über eine stabile Polardekomposition (mittels Standard-SVD), um eine strikte Spaltenorthogonalität bis auf Maschinengenauigkeit ($\sim 10^{-15}$ bei Standard-float64) zu garantieren.
*   **Alternating Least Squares (ALS):** Nutzt den robusten, iterativen *Direct Fitting* Optimierungsalgorithmus zur schnellen Konvergenz.
*   **Two-Stage Retrieval Engine:** Integrierte zweistufige Suche (Grobfilterung im $Q$-dimensionalen Raum via `np.argpartition` gefolgt von exaktem Rescoring auf den Originaldaten).
*   **Unterstützung für komplexe & reelle Zahlen:** Vollständig kompatibel mit reellwertigen Daten (z. B. Gewichten neuronaler Netze) und komplexwertigen Daten (z. B. Kanalmatrizen in der drahtlosen Signalverarbeitung / MIMO-Systemen).
*   **Radikale Low-Rank-Kompression:** Perfekt geeignet zur Reduzierung des Speicherbedarfs im Deep Learning (z. B. Kompression von Attention-Layern) und zur drastischen Optimierung großer Vektordatenbanken (RAG).

---

## 📂 Repository-Struktur

```text
├── LICENSE                 # PolyForm NonCommercial 1.0.0 Lizenztext
├── NOTICE                  # Urheberrechts- und akademische Autorenhinweise
├── README.md               # Englische Hauptdokumentation
├── README_DE.md            # Deutsche Dokumentation (diese Datei)
├── tensorag.py             # Produktionsbereite ML-GSVD-Klasse mit Two-Stage-Suche
├── tensorag_demo.py        # Vollständige RAG-Simulation zur Demonstration der Vektorkompression
├── tensorag_benchmark.py   # Geschwindigkeit- und Latenz-Vergleichstest
└── tensorag_streamlit_demo.py # Code für das interaktive Streamlit Web-Dashboard
```

---

## 💻 Schnellstart & Anwendung

Diese Bibliothek ist als einzelnes, schlankes Python-Modul konzipiert und benötigt außer **NumPy** keine weiteren externen Abhängigkeiten.

```python
import numpy as np
from tensorag import MultilinearGSVD

# 1. Generierung strukturierter Testdaten (z. B. 4 Matrizen mit gemeinsamer Spaltendimension von 100)
K, I = 4, 100
row_dimensions = [64, 48, 80, 120]  # Die Zeilenanzahl darf sich je Matrix unterscheiden!
H_list = [np.random.randn(dim, I) for dim in row_dimensions]

# 2. ML-GSVD initialisieren und auf Ziel-Rang Q = 16 komprimieren
Q = 16
gsvd = MultilinearGSVD(target_rank=Q, max_iter=50, tol=1e-6)
gsvd.fit(H_list)

# 3. Zugriff auf die berechneten Faktoren
print("Gemeinsame Basis A Form:", gsvd.A.shape)  # Erwartet: (100, 16)
for k in range(K):
    print(f"Schicht {k} - Orthogonaler Faktor B_k Form:", gsvd.B[k].shape)  # z.B. (64, 16)
    print(f"Schicht {k} - Koeffizienten C_k Diagonale:", gsvd.C[k, :])

# 4. Rekonstruktion und Fehlerberechnung
H_0_rec = gsvd.reconstruct(0)
reconstruction_error = np.linalg.norm(H_list[0] - H_0_rec)
print(f"Rekonstruktionsfehler für Schicht 0: {reconstruction_error:.4f}")
```

---

## 🎯 Zweistufige Retrieval-Pipeline (Coarse-to-Fine Rescoring)

Um kompressionsbedingte Recall-Verluste bei **bis zu 91 % RAM-Ersparnis im Hauptindex** vollständig auszugleichen, bietet TensoRAG eine eingebaute **Two-Stage Search Pipeline**:

1. **Stufe 1 (Grobfilter im Subspace):** Projiziert die Suchanfrage in den $Q$-dimensionalen Raum ($\mathbf{q}_{comp} = \mathbf{q} \cdot \mathbf{A}^*$) und wählt in $O(N)$ Linearzeit via `np.argpartition` die Top $N$ Kandidaten (z. B. $N=30$) aus.
2. **Stufe 2 (Exaktes Rescoring auf Originaldaten):** Berechnet die exakte Kosinus-Ähnlichkeit nur für diese 30 vorausgewählten Kandidaten auf den unkomprimierten Originalvektoren.

```python
import numpy as np
from tensorag import MultilinearGSVD

# 1. ML-GSVD Modell anpassen
gsvd = MultilinearGSVD(target_rank=128).fit(H_list)

# 2. Zweistufige Suche für einen 1536-dimensionalen Query-Vektor ausführen
query_vec = np.random.randn(1536)

final_doc_ids, final_scores = gsvd.two_stage_search(
    k=0,                        # Index der Ziel-Kollektion
    query_vector=query_vec,     # Unkomprimierter Original-Suchvektor
    raw_vectors_k=H_list[0],   # Unkomprimierte Datenvektoren für Stufe 2 Rescoring
    top_k=5,                    # Gewünschte Anzahl finaler Ergebnisse
    top_n_candidates=30         # Anzahl der vorausgewählten Kandidaten in Stufe 1
)

print("Top-5 Dokument-IDs:", final_doc_ids)
print("Top-5 Exakte Kosinus-Scores:", final_scores)
```

---

## 📜 Akademische Zitation & Quellen

Wenn Sie diesen Code oder den Algorithmus in Ihrer Forschung oder Ihren Anwendungen nutzen, zitieren Sie bitte die zugrundeliegenden akademischen Originalarbeiten:

### Primäre Dissertation
> **Khamidullina, Liana (2024).**  
> *Tensor decompositions and algorithms for efficient multidimensional signal processing.*  
> Dissertation, Technische Universität Ilmenau.  
> URN: [urn:nbn:de:gbv:ilm1-2024000104](https://nbn-resolving.org/urn:nbn:de:gbv:ilm1-2024000104)  
> DOI: [10.22032/dbt.59389](https://doi.org/10.22032/dbt.59389)

### Wissenschaftliche Hauptpublikation
> **L. Khamidullina, A. L. F. de Almeida, and M. Haardt,**  
> "Multilinear Generalized Singular Value Decomposition (ML-GSVD) and Its Application to Multiuser MIMO Systems,"  
> *IEEE Transactions on Signal Processing*, vol. 70, pp. 2783-2797, 2022.  
> DOI: [10.1109/TSP.2022.3178902](https://doi.org/10.1109/TSP.2022.3178902)

---

## ⚖️ Lizenz & Kommerzielle Lizensierung

### Nicht-Kommerzielle & Akademische Nutzung
TensoRAG ist dual lizenziert. Dieses Repository ist für **akademische Forschung, Bildungszwecke, universitäre Projekte, gemeinnützige wissenschaftliche Evaluierung und private nicht-kommerzielle Tests** unter der [PolyForm NonCommercial License 1.0.0](LICENSE) vollkommen kostenfrei.

### Kommerzielle Lizensierung & Enterprise-Nutzung
Jede kommerzielle Nutzung – einschließlich der Integration in kommerzielle Produkte, SaaS-Plattformen, interne Produktionssysteme in gewinnorientierten Unternehmen oder kostenpflichtige Beratungsdienstleistungen – erfordert eine separate **Kommerzielle Lizenz**.

Für Anfragen zu kommerziellen Lizenzen, individuellen SLAs oder Enterprise-Support kontaktieren Sie bitte:
* **Projektleitung:** Forstwichtel
* **E-Mail:** [forstwichtel@gmail.com](mailto:forstwichtel@gmail.com)

### Datenschutz / DSGVO-Konformität
* **Keine Telemetrie:** Dieser Code arbeitet zu 100 % offline und lokal. Es werden keinerlei Nutzungsdaten, Systemmetriken oder persönliche Identifikatoren erfasst oder übertragen.
* **Privacy-First:** Alle verarbeiteten Vektordaten verbleiben lokal in Ihrem System.
