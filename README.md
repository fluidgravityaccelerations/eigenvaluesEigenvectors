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

## JAX Batched SVD or Singular-Values-Only Computation with Generic Data Types (`JAXBatchedSVDGenericType.ipynb` and `JAXBatchedSVsonlyGenericType.ipynb`)

Batched computation of singular values only of real or complex matrices on GPU using JAX. With minor modifications, the code can run on CPU. `compute_uv=False` forces the computation of the singular values only.

- **Underlying Algorithm**  
  - **CPU**: JAX uses LAPACK's `?gesdd/?gesvd` routines (1. bidiagonalization plus 2. divide-and-conquer or 2. QR-based methods, respectively).
  - **GPU**: On NVIDIA GPUs, JAX uses `cuSOLVER`'s `?gesvd` for smaller matrices or `?gesvdp` for larger ones, depending on performance. They use 1. bidiagonalization plus 2. divide-and-conquer or 2. QR-based methods, respectively.

## CuPy Batched SVD or Singular-Values-Only Computation with Generic Data Types (`CuPyBatchedSVDGenericType.ipynb` and `CuPyBatchedSVsonlyGenericType.ipynb`)

Batched computation of singular values only of real or complex matrices on GPU using JAX. With minor modifications, the code can run on CPU. `compute_uv=False` forces the computation of the singular values only.

- **Underlying Algorithm**
  - **CPU**: same as JAX.
  - **GPU**: On NVIDIA GPUs, CuPy uses `cuSOLVER`'s `?gesvd`.

## Batched Singular-Value Computation on a Single GPU (`batchedSingularValuesSingleGPU.ipynb`)

Batched computation of singular values only on GPU using cuPy, with support for real matrices only (single or double precision). Algorithm developed for small matrices.

- **Main dependencies**
  - [PyCUDA](https://documen.tician.de/pycuda/)  

- **Usage Notes**
  - Assumes **number of rows ≤ number of columns**.  
  - Accuracy parameters (e.g. `1e-7` for single, `1e-13` for double) can be adjusted in the root solver.

## Batched Singular-Value Computation on Multiple GPUs ('batchedSingularValuesMultiGPU.ipynb')

Same as `batchedSingularValuesSingleGPU.ipynb`, but distributes the batch of small matrices across one or more GPUs.
