#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:03:27 2020

@author: michaelstaddon
"""

from fipy import (CellVariable, PeriodicGrid1D, Viewer, TransientTerm, DiffusionTerm,
                  UniformNoiseVariable, LinearLUSolver,
                  ImplicitSourceTerm, ExponentialConvectionTerm)


import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
from scipy.signal import correlate
from plotting import rho_cmap, myo_cmap

from scipy.stats import kurtosis

def run_simulation(sigma0, Br0, delta_Br,
                   signal_start, signal_duration, extra_time):

    # Define mesh size and number of points
    nx = 250
    ny = nx
    
    L = 150
    
    dx = L / nx
    dy = dx
    
    mesh = PeriodicGrid1D(dx, nx)
    
    
    # Variables to use
    r = CellVariable(name='r', mesh=mesh, hasOld=1)
    m = CellVariable(name='m', mesh=mesh, hasOld=1)
    v = CellVariable(name='v', mesh=mesh, hasOld=1)

    # parameters
    Br=6.8744e-4
    a=0.1609
    A=0.3833
    g=0.1787
    G=0.01
    Bm=0.0076
    ka=0.1408
    kd=0.0828
    Dr=0.1
    Dm=0.01
    gamma=1
    tau=5
    lam=14.3
    # sigma0=24.9 * 2
    m0=1

    
    Br = Br0
    
    elapsed = 0.0
    dt = 1
    
    eq_r = (TransientTerm(var=r) ==
        Br
        - ExponentialConvectionTerm(coeff=[[1.0]] * v, var=r)
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
    
    eq = eq_r & eq_m & eq_v

    i = 0
    
    rs = []
    ms = []
    vs = []
    
    while elapsed < 500:
        # Old values are used for sweeping when solving nonlinear values
        r.updateOld()
        m.updateOld()
        v.updateOld()
        
        elapsed += dt
        
        
        # One timestep
        res = 1e5
        old_res = res * 2
        step = 0
        while res > 1e-5 and step < 5 and old_res / res > 1.01:            
            old_res = res
            res = eq.sweep(dt=dt)
            step += 1
        
        if (elapsed > 500 - signal_start):
            # The variable values are just numpy arrays so easy to use!
            rs.append(r.value.copy())
            ms.append(m.value.copy())
            vs.append(v.value.copy())
        
        
        i += 1
    
    # This is where only a small region is activated!
    Br = Br0 + delta_Br * (mesh.x > 67.5) * (mesh.x < 82.5)
    
    eq_r = (TransientTerm(var=r) ==
            Br
            - ExponentialConvectionTerm(coeff=[[1.0]] * v, var=r)
            + ImplicitSourceTerm(coeff=(a / (A+r) - g * m / (G+r)), var=r)
            + DiffusionTerm(var=r, coeff=Dr))
            
    
    
    # Couple them into one big equation
    eq = eq_r & eq_m & eq_v

    elapsed = 0
    while elapsed < signal_duration:
        # Old values are used for sweeping when solving nonlinear values
        r.updateOld()
        m.updateOld()
        v.updateOld()
        
        elapsed += dt
        
        
        # One timestep
        res = 1e5
        old_res = res * 2
        step = 0
        while res > 1e-5 and step < 5 and old_res / res > 1.01:            
            old_res = res
            res = eq.sweep(dt=dt)
            step += 1
        
        # The variable values are just numpy arrays so easy to use!
        rs.append(r.value.copy())
        ms.append(m.value.copy())
        vs.append(v.value.copy())
            
    # And propagate without up regulation
    Br = Br0
    
    eq_r = (TransientTerm(var=r) ==
    Br
    - ExponentialConvectionTerm(coeff=[[1.0]] * v, var=r)
    + ImplicitSourceTerm(coeff=(a / (A+r) - g * m / (G+r)), var=r)
    + DiffusionTerm(var=r, coeff=Dr))
            
    
    
    # Couple them into one big equation
    eq = eq_r & eq_m & eq_v
    
    elapsed = 0
    while elapsed < extra_time:
        # Old values are used for sweeping when solving nonlinear values
        r.updateOld()
        m.updateOld()
        v.updateOld()
        
        elapsed += dt
        
        
        # One timestep
        res = 1e5
        old_res = res * 2
        step = 0
        while res > 1e-5 and step < 5 and old_res / res > 1.01:            
            old_res = res
            res = eq.sweep(dt=dt)
            step += 1
        
        # The variable values are just numpy arrays so easy to use!
        rs.append(r.value.copy())
        ms.append(m.value.copy())
        vs.append(v.value.copy())
        
    return rs, ms, vs

if __name__ == '__main__':

    Br = 0
    delta_Br = 0.03
    sigma0 = 80
    
    signal_start = 100
    signal_duration = 200
    extra_time = 1700
    
    signal_end = signal_start + signal_duration
    total_time = signal_start + signal_duration + extra_time
    
    
    # for sigma0 in sigmas:
    rs, ms, vs = run_simulation(sigma0, Br, delta_Br,
                                signal_start, signal_duration, extra_time)
  

