# GPU computation of eigenvalues and eigenvectors

## JAX Batched SVD or Singular-Values-Only Computation with Generic Data Types (`JAXBatchedSVDGenericType.ipynb` and `JAXBatchedSVsonlyGenericType.ipynb`)

Batched computation of singular values only of real or complex matrices on GPU using JAX. With minor modifications, the code can run on CPU. `compute_uv=False` forces the computation of the singular values only.

- **Underlying Algorithm**  
  - **CPU**: JAX uses LAPACK's `?gesdd/?gesvd` routines (1. bidiagonalization plus 2. divide-and-conquer or 2. QR-based methods, respectively).
  - **GPU**: On NVIDIA GPUs, JAX uses `cuSOLVER`'s `?gesvd` for smaller matrices or `?gesvdp` for larger ones, depending on performance. They use 1. bidiagonalization plus 2. divide-and-conquer or 2. QR-based methods, respectively.

## JAX Batched Singular-Values-Only Computation with Generic Data Types (`JAXmultiGPU.ipynb`)

Multi-GPU version of the previous algorithm. With minor modifications, the code can run on CPU. `compute_uv=False` forces the computation of the singular values only. Change this assignment to enable to full SVD computation.

## CuPy Batched SVD or Singular-Values-Only Computation with Generic Data Types (`CuPyBatchedSVDGenericType.ipynb` and `CuPyBatchedSVsonlyGenericType.ipynb`)

Batched computation of singular values only of real or complex matrices on GPU using JAX. With minor modifications, the code can run on CPU. `compute_uv=False` forces the computation of the singular values only.

- **Underlying Algorithm**
  - **CPU**: same as JAX.
  - **GPU**: On NVIDIA GPUs, CuPy uses `cuSOLVER`'s `?gesvd`.

## PyTorch Batched SVD or Singular-Values-Only Computation with Generic Data Types (`PyTorchBatchedSVDGenericType.ipynb` and `PyTorchBatchedSVsonlyGenericType.ipynb`)

Batched computation of singular values only of real or complex matrices on GPU using JAX. With minor modifications, the code can run on CPU. `compute_uv=False` forces the computation of the singular values only.

- **Underlying Algorithm**  
  - **CPU**: same as JAX.  
  - **GPU**: same as CuPy.

## Batched Singular-Value Computation on a Single GPU (`batchedSingularValuesSingleGPU.ipynb`)

Batched computation of singular values only on GPU using cuPy, with support for real matrices only (single or double precision). Algorithm developed for small matrices.

- **Main dependencies**
  - [PyCUDA](https://documen.tician.de/pycuda/)  

- **Usage Notes**
  - Assumes **number of rows ≤ number of columns**.  
  - Accuracy parameters (e.g. `1e-7` for single, `1e-13` for double) can be adjusted in the root solver.

## Batched Singular-Value Computation on Multiple GPUs ('batchedSingularValuesMultiGPU.ipynb')

Same as `batchedSingularValuesSingleGPU.ipynb`, but distributes the batch of small matrices across one or more GPUs.
