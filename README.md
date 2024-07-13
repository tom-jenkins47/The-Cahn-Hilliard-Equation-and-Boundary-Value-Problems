## The Cahn-Hilliard Equation ##

The Cahn-Hilliard equation is a non-linear partial differential equation used to describe phase separation in systems such as water-oil emulsions. Unlike the linear diffusion equation, the Cahn-Hilliard equation cannot be solved analytically but can be approached numerically using finite difference methods.

The Cahn-Hilliard equation is given by:

<div align="center">
<img src="https://latex.codecogs.com/svg.latex?\frac{\partial \phi(x, t)}{\partial t} = M \nabla^2 \mu(x, t)" alt="\frac{\partial \phi(x, t)}{\partial t} = M \nabla^2 \mu(x, t)">
</div>

where the chemical potential \(\mu(x, t)\) is:

<div align="center">
<img src="https://latex.codecogs.com/svg.latex?\mu(x, t) = -a\phi + b\phi^3 - \kappa \nabla^2 \phi" alt="\mu(x, t) = -a\phi + b\phi^3 - \kappa \nabla^2 \phi">
</div>
