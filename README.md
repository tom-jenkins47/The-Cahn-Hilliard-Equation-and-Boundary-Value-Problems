## The Cahn-Hilliard Equation ##

The Cahn-Hilliard equation is a non-linear partial differential equation used to describe phase separation in systems such as water-oil emulsions. Unlike the linear diffusion equation, the Cahn-Hilliard equation cannot be solved analytically but can be approached numerically using finite difference methods.

The Cahn-Hilliard equation is given by:

$\frac{\partial \phi(x, t)}{\partial t} = M \nabla^2 \mu(x, t)$

where the chemical potential $(\mu(x, t))$ is:

$\mu(x, t) = -a\phi + b\phi^3 - \kappa \nabla^2 \phi$

Here $\phi(x, t)$ is the compositional order parameter $a$ and $b$ are constants, $\kappa$ relates to the surface tension and $M$ is the mobility constant. The Laplacian and the time derivative are approximated to simulate phase separation dynamics.

#### Run the code CahnHilliardSimulation.py and follow the prompts to either simulate a phase-separated fluid on a defined grid size, or collect data relating to the evolution of the system over time. If using the Spyder IDE, ensure the plotting backend is set to automatic. ####

## Boundary Value Problems ##

In the case of boundary value problems, the solution is not time-dependent and is instead specified on the boundaries of the simulation domain. The Poisson equation is a typical example. This equation can be used to describe the behaviour of electric and magnetic fields in space and their evolution over time.

#### Run the code BVPSimulation.py and follow the prompts to simulate an electric or magnetic field distribution using either the Jacobi or Gauss algorithms on a defined grid size. If using the Spyder IDE, ensure the plotting backend is set to automatic. ####








