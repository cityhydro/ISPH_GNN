# ISPH_GNN
The implementation of a hybrid method (isph_gnn) for predicting the results presented in paper
"A hybrid method coupling ISPH and graph neural network for simulating free surface flows"

Description of the code structure:

  The ISPH code is run on Fortran and the GNN code is run on Python. C++ (f_c_p.so) including pybind11 is used as the interface to transfer the input data from ISPH 
  to the GNN program, and then return the predicted pressure from GNN to the ISPH program


Requirement:

  PyTorch (>1.7.1), DGL (>0.7.0), PyTorch Geometric (>1.7.0), CuPy (latest version), Numpy, Scipy, Scikit-learn & Numba;
  Pandas, Partio (https://github.com/wdas/partio) for I/O;
  FRNN from: https://github.com/lxxue/FRNN;
  pybind11(https://github.com/pybind/pybind11) for I/O between C++ (ISPH data information) and python (GNN data information);
  The code is tested under Linux Ubuntu 18.04 with CUDA 11.7;

To run the simulation of ISPH_GNN for producing the numerical results including the particle distribution and pressure contour (reslt_animate), 
the surface evelations at time instant t1(result_surfacet1) or the surface evelations at position x1(result_surfacex1),etc. 

  LD_LIBRARY_PATH=. ./isphgnntest


More information about the original GNN model used in this paper can be found in https://github.com/BaratiLab/FGN.
