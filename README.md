# eigenvaluesEigenvectors

## PyTorch Batched SVD with Generic Data Types

This notebook demonstrates how to perform high-performance, batched singular value decompositions (SVD) on GPU using PyTorch, with support for both real and complex matrix types:

- **Generic Type Handling**  
  Automatically generates batches of real (`float32`/`float64`) or complex (`complex64`/`complex128`) matrices, driven by a single `dtype` flag.

- **Batched SVD on GPU**  
  Uses `torch.linalg.svd` (or `torch.linalg.svdvals`) to compute either full SVD or singular-values-only for each matrix in the batch, leveraging CUDA for massive parallelism.

- **Underlying Algorithm**  
  - On **CPU**, `torch.linalg.svd(..., full_matrices=False)` dispatches to LAPACK’s divide‐and‐conquer driver `*gesdd` (falling back to `*gesvd` if needed).  
  - On **CUDA**, it calls MAGMA’s `gesvd` implementation, which likewise performs bidiagonalization followed by a divide‐and‐conquer solve of the bidiagonal SVD.

- **Timing & Warm-up**  
  Includes warm-up steps to initialize the CUDA context, and measures elapsed GPU time with `torch.cuda.Event` for accurate benchmarking.

- **Accuracy Validation**  
  Transfers results back to CPU and compares against NumPy’s `np.linalg.svd` (or `np.linalg.svd(..., compute_uv=False)`) by computing relative reconstruction or singular-value errors.

- **Batch Size Scaling**  
  Iterates over a range of power-of-two batch sizes to illustrate performance scaling and accuracy trade-offs across different workloads.

---

> **Usage:**  
> 1. Set `device` (e.g. `"cuda"`), `dtype`, and the list of `batch_sizes`.  
> 2. Run the notebook cells in order to generate data, perform batched SVD, and collect timing & error metrics.  
> 3. Inspect the printed output or log the results for analysis of speed vs. precision.

## PyTorch Batched Singular-Values-Only SVD with Generic Data Types

This notebook demonstrates how to perform high-performance, **batched computation of singular values only** on GPU using PyTorch, with support for both real and complex matrices.

### Key Features

- **Generic Type Handling**  
  Automatically generates batches of real (`float32`/`float64`) or complex (`complex64`/`complex128`) matrices using a single `dtype` flag.

- **Singular-Values-Only SVD**  
  Computes only the singular values via:  
  ```python
  S = torch.linalg.svdvals(X)

- **Underlying Algorithm**
  CPU: Dispatches to LAPACK’s divide-and-conquer driver *gesdd (with fallback to *gesvd).
  CUDA: Calls MAGMA’s gesvd implementation, which performs bidiagonalization followed by a divide-and-conquer solve, while only materializing the singular values.

- **Timing & Warm-up**  
  Properly initializes the CUDA context for both real and complex dtypes.

- **Measures GPU execution time using torch.cuda.Event for accurate benchmarking**
  Accuracy Validation
  Retrieves S back to CPU and validates against NumPy’s

  ```python
  S_ref = np.linalg.svd(A, compute_uv=False)

  using a relative-error metric.

- **Batch Size Scaling**
  Iterates over a range of power-of-two batch sizes to illustrate performance scaling and precision trade-offs.

> **Usage:**  
> 1. Set your device (e.g. "cuda"), desired dtype, and list of batch_sizes.
> 2. Run the notebook cells in sequence:
>    Data generation
>    Singular-values-only SVD on GPU
>    Timing & error computation
> 3. Inspect the printed output or logs to analyze speed vs. accuracy.
