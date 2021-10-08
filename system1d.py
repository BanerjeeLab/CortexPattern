#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:03:27 2020

@author: michaelstaddon
"""

from fipy import (CellVariable, PeriodicGrid1D, Viewer, TransientTerm, DiffusionTerm,
                  UniformNoiseVariable, LinearLUSolver,
                  ImplicitSourceTerm, ExponentialConvectionTerm)

from fipy.tools import numerix
import matplotlib.pyplot as plt
import numpy as np
import time
from plotting import rho_cmap, myo_cmap


def run_sim(Br=6.8744e-4,
            a=0.1609,
            A=0.3833,
            g=0.1787,
            G=0.01,
            Bm=0.0076,
            ka=0.1408,
            kd=0.0828,
            Dr=0.1,
            Dm=0.01,
            gamma=1,
            tau=5,
            lam=14.3,
            sigma0=24.9 * 2,
            m0=1,
            phi=1,
            duration=600,
            dt=1,
            nx=250,
            L=140):
    
    # Define mesh size and number of points

    dx = L / nx
    
    mesh = PeriodicGrid1D(dx, nx)
    
    
    # Variables to use
    r = CellVariable(name='r', mesh=mesh, hasOld=1)
    m = CellVariable(name='m', mesh=mesh, hasOld=1)
    v = CellVariable(name='v', mesh=mesh, hasOld=1)
    
    # Add some random noise
    r.setValue(UniformNoiseVariable(mesh=mesh, minimum=0, maximum=0.01))
    m.setValue(UniformNoiseVariable(mesh=mesh, minimum=0, maximum=0.01))    
    
                     
    # Define the equations using their special classes
    # TransientTerm = d var / dt
    # Convection Term = u dot grad(v) - here I do it by x and y components
    # Source term for reactions - use this instead of equations for the solver to work better
    # DiffusionTerm for diffusion...
    eq_r = (TransientTerm(var=r) ==
            Br
            - ExponentialConvectionTerm(coeff=[[1.0]] * v * phi, var=r)
            + ImplicitSourceTerm(coeff=(a / (A+r) - g * m / (G+r)), var=r)
            + DiffusionTerm(var=r, coeff=Dr))
            
    eq_m = (TransientTerm(var=m) ==
            Bm
            - ExponentialConvectionTerm(coeff=[[1.0]] * v, var=m)
            + ImplicitSourceTerm(coeff=ka * r, var=r)
            - ImplicitSourceTerm(coeff=kd, var=m)
            + DiffusionTerm(var=m, coeff=Dm))
    
    
    eq_v = (gamma * tau * TransientTerm(var=v) ==
            DiffusionTerm(coeff=lam * lam, var=v) 
            - ImplicitSourceTerm(coeff=gamma, var=v)
            + sigma0 * m.grad.dot([[1.0]]) * (m0 / (m0 + m) ** 2))
    
    # Couple them into one big equation
    eq = eq_r & eq_m & eq_v
    
    elapsed = 0.0
    
    # You can use the viewer when the mesh is a weird shape eg sphere
    # viewer = Viewer(vars=(r,m,u,v), datamin=0.)

    i = 0
    
    rs = []
    ms = []
    vs = []
    while elapsed < duration:
        # Old values are used for sweeping when solving nonlinear values
        r.updateOld()
        m.updateOld()
        v.updateOld()
        
        elapsed += dt
        
        # Optional sweep instead of solve - use for nonlinear problems
        res = 1e5
        steps = 0
        while res > 1e-3 and steps < 5:
            res = eq.sweep(dt=dt)
            steps += 1
        
        # # One timestep
        # eq.solve(dt=dt)
        
        # The variable values are just numpy arrays so easy to use!
        rs.append(r.value.copy())
        ms.append(m.value.copy())
        vs.append(v.value.copy())
        
        
        i += 1
        
    return rs, ms, vs


if __name__ == '__main__':

    sigma = 80
    Br = 0.0
    
   
    # Run the simulation
    rs, ms, vs = run_sim(Br=Br, sigma0=sigma)
    
    plt.pcolor(rs)
    plt.show()