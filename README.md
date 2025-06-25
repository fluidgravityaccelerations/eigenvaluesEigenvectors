# eigenvaluesEigenvectors

## PyTorch Batched SVD with Generic Data Types

Batched singular value decompositions (SVD) on GPU using PyTorch, with support for both real and complex matrix types:

- **Underlying Algorithm**  
  - On **CPU**, `torch.linalg.svd(..., full_matrices=False)` dispatches to LAPACK’s divide‐and‐conquer driver `*gesdd` (falling back to `*gesvd` if needed).  
  - On **CUDA**, it calls MAGMA’s `gesvd` implementation, which likewise performs bidiagonalization followed by a divide‐and‐conquer solve of the bidiagonal SVD.

## PyTorch Batched Singular-Values-Only SVD with Generic Data Types

Batched computation of singular values only on GPU using PyTorch, with support for both real and complex matrices.

- **Underlying Algorithm**
  CPU: Dispatches to LAPACK’s divide-and-conquer driver *gesdd (with fallback to *gesvd).
  CUDA: Calls MAGMA’s gesvd implementation, which performs bidiagonalization followed by a divide-and-conquer solve, while only materializing the singular values.

## JAX Batched Singular-Values-Only SVD with Generic Data Types

Batched computation of singular values only on CPU and GPU using JAX, with support for both real and complex matrices.

- **Underlying Algorithm**  
  - **CPU**: Uses XLA’s SVD high-level operation, which on CPU dispatches (via Eigen’s LAPACK bindings) to the divide-and-conquer driver `*gesdd` (with fallback to `*gesvd`) to compute singular values only when `compute_uv=False`. :contentReference[oaicite:0]{index=0}  
  - **CUDA**: The same XLA SVD HLO is lowered to call cuSolver’s `gesvd` (divide-and-conquer) backend, materializing only the singular values when `compute_uv=False`. :contentReference[oaicite:1]{index=1}
