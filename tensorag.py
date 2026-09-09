# Copyright 2026 Forstwichtel & Gemini Notebook [bot]
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
#
# Project: TensoRAG
# Authors: Forstwichtel, Gemini Notebook [bot]

import numpy as np

class MultilinearGSVD:
    """
    Implementation of the Multilinear Generalized Singular Value Decomposition (ML-GSVD)
    as part of the TensoRAG project, based on the dissertation of Liana Khamidullina (TU Ilmenau, 2022).
    
    The ML-GSVD decomposes a set of K matrices H_k (size J_k x I) with a common column dimension I:
        H_k ≈ B_k * C_k * A^H
        
    Where:
        - A (size I x Q) is the common, left-invertible right basis matrix.
        - B_k (size J_k x Q) are individual, left factor matrices with orthogonal columns.
        - C_k (size Q x Q) are diagonal matrices containing the generalized singular values,
          satisfying the normalization condition: sum_{k=1}^K C_k^2 = I_Q.
    """
    def __init__(self, target_rank, max_iter=100, tol=1e-5, random_state=None):
        self.target_rank = target_rank
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        self.A = None      # Common right basis matrix (I x Q)
        self.B = None      # List of left orthogonal factor matrices [J_k x Q]
        self.C = None      # Coefficient matrix (K x Q), diag(C_k) corresponds to C[k, :]
        self.errors = []   # Reconstruction errors per iteration
        
    def fit(self, H_list):
        """
        Fits the ML-GSVD model to a list of K matrices H_k.
        
        Parameters:
            H_list (list of np.ndarray): List of K matrices of shape (J_k, I).
                                         Can be real or complex.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        K = len(H_list)
        I = H_list[0].shape[1]
        Q = self.target_rank
        
        # Verify dimensions
        for k, Hk in enumerate(H_list):
            if Hk.shape[1] != I:
                raise ValueError(f"Matrix at index {k} has column dimension {Hk.shape[1]}, expected {I}.")
        
        # Detect if input is complex-valued
        is_complex = any(np.iscomplexobj(Hk) for Hk in H_list)
        dtype = np.complex128 if is_complex else np.float64
        
        # 1. Initialize A using SVD of the sum of Gramians sum_k (H_k^H * H_k)
        sum_HH = np.zeros((I, I), dtype=dtype)
        for Hk in H_list:
            sum_HH += Hk.conj().T @ Hk
            
        U_init, _, _ = np.linalg.svd(sum_HH)
        self.A = U_init[:, :Q].copy() # Shape: I x Q
        
        # Initialize C column-normalized
        self.C = np.random.rand(K, Q)
        self.C /= np.sqrt(np.sum(self.C**2, axis=0, keepdims=True))
        
        self.B = [None] * K
        H_tilde = [None] * K
        
        prev_error = float('inf')
        self.errors = []
        
        # 2. Alternating Least Squares (ALS) Loop
        for iteration in range(self.max_iter):
            # Step A: Update individual left factor matrices B_k (Orthogonal Procrustes)
            for k, Hk in enumerate(H_list):
                # H_tilde_k = A * diag(C(k, :))
                H_tilde_k = self.A @ np.diag(self.C[k, :]) # Shape: I x Q
                T_k = Hk @ H_tilde_k                      # Shape: J_k x Q
                
                # Polar decomposition via SVD
                U_t, _, Vh_t = np.linalg.svd(T_k, full_matrices=False)
                self.B[k] = U_t @ Vh_t                    # Shape: J_k x Q
                
                # Update H_tilde_k = H_k^H * B_k for step B
                H_tilde[k] = Hk.conj().T @ self.B[k]      # Shape: I x Q
                
            # Step B: Jointly update A and C column-wise (Least Squares & Normalization)
            for q in range(Q):
                # Form Z_q by gathering the q-th columns of H_tilde_k across all K slices
                Z_q = np.zeros((I, K), dtype=dtype)
                for k in range(K):
                    Z_q[:, k] = H_tilde[k][:, q]
                    
                # Standard SVD of Z_q (size I x K)
                u_z, s_z, vh_z = np.linalg.svd(Z_q, full_matrices=False)
                
                # Extract first right singular vector
                v_1 = vh_z[0, :].conj()
                
                # Set C[:, q] as the absolute value (non-negative, real-valued constraint)
                self.C[:, q] = np.abs(v_1)
                
                # Set A[:, q] to the optimal least-squares direction
                self.A[:, q] = Z_q @ self.C[:, q]
                
            # Step C: Normalize column metrics of C to preserve sum_k C(k, q)^2 = 1
            col_norms = np.linalg.norm(self.C, axis=0)
            # Avoid division by zero
            col_norms[col_norms == 0] = 1.0
            self.C /= col_norms[np.newaxis, :]
            
            # Recalculate Reconstruction Error
            current_error = 0.0
            for k, Hk in enumerate(H_list):
                recon = self.B[k] @ np.diag(self.C[k, :]) @ self.A.conj().T
                current_error += np.linalg.norm(Hk - recon)**2
                
            self.errors.append(current_error)
            
            # Check convergence
            if abs(prev_error - current_error) < self.tol:
                break
            prev_error = current_error
            
        return self
        
    def reconstruct(self, k):
        """
        Reconstructs the k-th slice matrix.
        """
        if self.A is None or self.B is None or self.C is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        return self.B[k] @ np.diag(self.C[k, :]) @ self.A.conj().T

    def transform_query(self, query_vector):
        """
        Projects an uncompressed high-dimensional query vector (I,) into 
        the shared compressed subspace (Q,).
        
        q_compressed = q @ A*
        """
        if self.A is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        
        q = np.asarray(query_vector).squeeze()
        if q.ndim != 1 or q.shape[0] != self.A.shape[0]:
            raise ValueError(f"Query vector must have dimension {self.A.shape[0]}, got {q.shape}.")
            
        return q @ self.A.conj()

    def search_compressed(self, k, query_compressed, top_k=5):
        """
        Executes fast cosine similarity search directly in the compressed Q-dimensional subspace.
        
        Parameters:
            k (int): Index of the domain slice (0 <= k < K).
            query_compressed (np.ndarray): Compressed query vector of shape (Q,).
            top_k (int): Number of top documents to retrieve.
            
        Returns:
            indices (np.ndarray): Indices of top_k documents.
            scores (np.ndarray): Compressed similarity scores.
        """
        if self.B is None or self.C is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
            
        B_k_scaled = self.B[k] * self.C[k, :] # Weight factor matrix by singular values
        
        norm_q = np.linalg.norm(query_compressed)
        norm_B = np.linalg.norm(B_k_scaled, axis=1)
        
        similarities = (B_k_scaled @ query_compressed) / (norm_B * norm_q + 1e-10)
        
        if len(similarities) > top_k:
            part_ids = np.argpartition(similarities, -top_k)[-top_k:]
            sorted_top = part_ids[np.argsort(similarities[part_ids])[::-1]]
        else:
            sorted_top = np.argsort(similarities)[::-1]
            
        return sorted_top, similarities[sorted_top]

    def two_stage_search(self, k, query_vector, raw_vectors_k, top_k=5, top_n_candidates=30):
        """
        Executes Two-Stage Retrieval (Coarse-to-Fine):
          - Stage 1: Fast candidate selection in compressed Q-dim space using np.argpartition.
          - Stage 2: Exact rescoring on uncompressed raw_vectors_k for candidates only.
          
        Parameters:
            k (int): Domain slice index.
            query_vector (np.ndarray): Original uncompressed query vector (I,).
            raw_vectors_k (np.ndarray): Original uncompressed dataset slice matrix (J_k, I).
            top_k (int): Number of final top documents to return.
            top_n_candidates (int): Number of candidates to over-fetch in Stage 1.
            
        Returns:
            final_indices (np.ndarray): Indices of top_k documents.
            final_scores (np.ndarray): Exact cosine similarity scores.
        """
        # Ensure candidate pool is at least top_k
        top_n_candidates = max(top_k, top_n_candidates)
        
        # 1. Project query into compressed subspace
        q_comp = self.transform_query(query_vector)
        
        # 2. Stage 1: Retrieve candidate IDs in compressed space
        candidate_ids, _ = self.search_compressed(k, q_comp, top_k=top_n_candidates)
        
        # 3. Stage 2: Exact rescoring on raw vectors for candidates
        candidate_vectors = raw_vectors_k[candidate_ids]
        
        norm_q = np.linalg.norm(query_vector)
        norm_candidates = np.linalg.norm(candidate_vectors, axis=1)
        
        exact_similarities = (candidate_vectors @ query_vector) / (norm_candidates * norm_q + 1e-10)
        
        if len(exact_similarities) > top_k:
            part_ids = np.argpartition(exact_similarities, -top_k)[-top_k:]
            sorted_in_candidates = part_ids[np.argsort(exact_similarities[part_ids])[::-1]]
        else:
            sorted_in_candidates = np.argsort(exact_similarities)[::-1]
            
        final_indices = candidate_ids[sorted_in_candidates]
        final_scores = exact_similarities[sorted_in_candidates]
        
        return final_indices, final_scores

# Quick validation run
if __name__ == "__main__":
    print("Verifying TensoRAG ML-GSVD Python Class implementation with Two-Stage Retrieval...")
    # 4 slices, common column dimension 64, varying row dimensions J_k
    I = 64
    H_test = [
        np.random.randn(200, I),
        np.random.randn(150, I),
        np.random.randn(100, I),
        np.random.randn(180, I)
    ]
    
    # Run with target rank 16
    gsvd = MultilinearGSVD(target_rank=16, max_iter=50, tol=1e-6, random_state=42)
    gsvd.fit(H_test)
    
    print(f"Convergence reached after {len(gsvd.errors)} iterations.")
    print(f"Initial Error: {gsvd.errors[0]:.6f} -> Final Error: {gsvd.errors[-1]:.6f}")
    
    # Generate random test query
    q_test = np.random.randn(I)
    
    # Test transform_query
    q_comp = gsvd.transform_query(q_test)
    print("Compressed query shape:", q_comp.shape)
    
    # Test Stage 1 search
    cand_ids, cand_scores = gsvd.search_compressed(k=0, query_compressed=q_comp, top_k=10)
    print("Stage 1 top 5 candidate IDs:", cand_ids[:5])
    
    # Test Two-Stage search
    final_ids, final_scores = gsvd.two_stage_search(
        k=0, 
        query_vector=q_test, 
        raw_vectors_k=H_test[0], 
        top_k=5, 
        top_n_candidates=30
    )
    print("Two-stage final top 5 IDs:", final_ids)
    print("Two-stage final top 5 scores:", final_scores)
    
    # Compare with exact brute-force search on full raw matrix
    norm_q = np.linalg.norm(q_test)
    norm_H0 = np.linalg.norm(H_test[0], axis=1)
    bf_sims = (H_test[0] @ q_test) / (norm_H0 * norm_q + 1e-10)
    bf_top5 = np.argsort(bf_sims)[-5:][::-1]
    print("Brute-force exact top 5 IDs:", bf_top5)
    
    overlap = len(set(final_ids).intersection(set(bf_top5)))
    print(f"Two-stage recall overlap with brute-force: {overlap}/5")
