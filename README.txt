The file Functions.py is a script containing all of the field and potential functions. 
The functions within this class use the numba package. Details can be found here: https://numba.readthedocs.io/en/stable/user/jit.html.
To install numba: $ conda install numba
This package is a compiler for Python that works best on code that uses NumPy arrays and functions, and loops. It effectively compiles code in C where possible.

The file CahnHilliard.py contains the update class, as well as the visualizer and data collection components. This program uses prompts to initialize the system. The default parameters are assumed to be N=100, M=0.1, a=0.1, k=0.1, dx=1 and dt=1. The phi parameter is left up to the user. Note: if Spyder is used, in order to run the animation the plotting backend should be changed from inline to automatic.

The file BVPSimulation.py contains the class that is used to model the boundary value problems. The user is again prompted from the terminal upon execution of the program. Selections can be made as to whether a magnetic field or electric field is modelled, with either the Jacobi or Gauss-Seidl algorithm. If an electric field is modelled, the decision can be made to model either a random or point charge distribution, or optimize the over-relaxation. For all plotting, a tolerance of 0.001 and grid size of N=100 was used to achieve convergence.

The .csv files that are included for the electric field used the Gaussian algorithm and a point charge. The files for the magnetic field used the Gaussian algorithm (and the wires). Relevant graphs are included for the electric field/potential using the Gaussian algorithm and random/point charge distributions, as well as for the magnetic field/potential using the Gaussian algorithm. A graph of the over-relaxation optimization is also attached. For this, the optimal value of omega was determined to be omega=1.94. This was the point when the number of iterations to converge was minimized.
