<h1 align="center">
<img src="./docs/_static/depth-logo.jpg" width="200">
</h1><br>

Data depth package
=================
___
Following the seminal idea of Tukey (1975), data depth is a function that measures how close an arbitrary point of the space is located to an implicitly defined center of a data cloud. Having undergone theoretical and computational developments, it is now employed in numerous applications. The **data-depth** library is a software directed to fuse experience of the applicant with recent achievements in the area of data depth. This library provides an implementation for exact and approximate computation of most reasonable and widely applied notions of **data-depth**.

- **Website:** https://data-depth.github.io
- **Source code:** https://github.com/data-depth/library
- **Contributing:** https://data-depth.github.io/multivariate/credits.html#contributors

___
Instalation:
---

data-depth can be directly installed using **pip**:

    pip install data-depth

Running GPU based depth requires CUDA availability, it can be installed using pytorch:
    
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    pip install data-depth

When using **conda**, run before:

    conda install conda-forge::pytorch
    pip install data-depth
    
Or for GPU usage:

    conda install conda-forge::pytorch-gpu 
    pip install data-depth

For more information about CUDA version, see https://pytorch.org/get-started/locally/
