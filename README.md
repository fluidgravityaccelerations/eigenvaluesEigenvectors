# GPU computation of eigenvalues and eigenvectors

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

## JAX Batched SVD with Generic Data Types

Batched singular value decompositions (SVD) on GPU (and CPU) using JAX, with support for both real and complex matrix types:

- **Underlying Algorithm**  
  - On **CPU**, JAX’s `jax.numpy.linalg.svd` is implemented in XLA using Eigen’s divide-and-conquer bidiagonalization driver (analogous to LAPACK’s `*gesdd`, with a fallback to `*gesvd`).  
  - On **CUDA**, it issues an XLA custom-call to NVIDIA cuSolver’s `gesvd` divide-and-conquer solver (and, for smaller batched workloads, can fall back to the `gesvdj` Jacobi-based implementation).

## JAX Batched Singular-Values-Only SVD with Generic Data Types (`JAXBatchedSVsonlyGenericType.ipynb`)

Batched computation of singular values only of complex matrices on GPU using JAX. With minor modifications, the code can run on CPU. `compute_uv=False` forces the computation of the singular values only.

- **Underlying Algorithm**  
  - **CPU**: JAX uses LAPACK (via SciPy's interface), typically the `?gesdd/?gesvd` routines (divide-and-conquer or QR-based methods).
  - **GPU**: On NVIDIA GPUs, JAX uses `cuSOLVER`'s `cusolverDnXgesvd` for smaller matrices or `cusolverDnXgesvdp` for larger ones, depending on performance.
    
## CuPy Batched SVD with Generic Data Types (`CuPyBatchedSVDGenericType.ipynb`)

Batched computation of SVD of complex matrices on GPU using CuPy.

- **Underlying Algorithm**
  - **CPU** (with minor modifications, the algorithm can run on CPU): Dispatches to SciPy/LAPACK’s divide-and-conquer driver `gesdd` (with fallback to `gesvd`).
  - **CUDA (CuPy)**: Calls cuSOLVER’s `gesvdaStridedBatched` driver, which performs bidiagonalization followed by a divide-and-conquer solve—and when `compute_uv=False` only materializes the singular values.

## CuPy Batched Singular-Values-Only SVD with Generic Data Types

Batched computation of singular values only on GPU using cuPy, with support for both real and complex matrices.

- **Underlying Algorithm**
  - **CPU:** Dispatches to LAPACK’s divide-and-conquer driver `*gesdd` (falling back to `*gesvd` when necessary).
  - **CUDA:** Calls CUDA Toolkit’s cuSOLVER `gesvd` (divide-and-conquer) implementation for real and complex types, with a fallback to the Jacobi-based `gesvdj` solver when higher accuracy or robustness is required.

## CuPy Batched Full SVD with Generic Data Types

Batched computation of full singular‐value decomposition on GPU using CuPy, with support for both real and complex matrices.

- **Underlying Algorithm**
  - **CPU**: Dispatches to LAPACK’s divide-and-conquer driver *gesdd (with fallback to *gesvd).
  - **CUDA**: Calls NVIDIA cuSOLVER’s gesvd (divide-and-conquer) implementation, which performs bidiagonalization followed by a divide-and-conquer solve to produce the full U, Σ and Vᵀ factors.

## Batched Singular-Value Computation on a Single GPU

Compute only the singular values of a large batch of small matrices on one GPU, using PyCUDA kernels.

---

### 🔧 Dependencies
- Python 3.x  
- [PyCUDA](https://documen.tician.de/pycuda/)  
- NumPy  

### ⚙️ Usage Notes
- Assumes **number of rows ≤ number of columns**.  
- Accuracy parameters (e.g. `1e-7` for single, `1e-13` for double) can be adjusted in the root solver.

## Batched Singular-Value Computation on Multiple GPUs

Distribute a large batch of small matrices across one or more GPUs and compute **only** their singular values in parallel using PyCUDA.

---

### 🔧 Dependencies
- Python 3.x  
- [PyCUDA](https://documen.tician.de/pycuda/)  
- NumPy  

---

### ⚙️ Usage Notes
- **Multi‑GPU support:** automatically splits the batch evenly across available devices (fallback to one GPU if only one is present).  
- Assumes **number of rows ≤ number of columns**.  
- Uses a pipelined sequence of custom kernels (rearrange → bidiagonalize → extract → tridiagonalize → pivots → intervals → Sturm bisection) on each GPU.  
- Accuracy tolerance adjustable via `sturmTol` (`1e-7` for `float32`, `1e-13` for `float64`).  
- No intermediate copies—everything runs asynchronously on per‑GPU CUDA streams for maximal throughput.  
---
