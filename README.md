# Smagorinsky_LES_FEM
This code was created by Nate Lin and Sumedh Soman as a final project for the MCEN 5231 (Computational Fluid Dynamics) course at University of Colorado Boulder, taught in Spring 2023 by Dr. Debanjan Mukherjee (Assistant Professor, Paul M. Rady Mechanical Engineering). The code implements a standard Smagorinsky LES model in an incompressible stabilized Petrov-Galerkin Finite Element code. The code was based on homework assignment codes for the course, and utilizes FEniCS. 
The Smagorinsky terms are implemented as forcing terms in the incompressible Navier Stokes equations. P1P1 elements are used with stabilization to reduce computational cost. The 3D taylor green vortex is used as a test case.

