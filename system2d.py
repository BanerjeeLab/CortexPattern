#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:03:27 2020

@author: michaelstaddon
"""

from fipy import (CellVariable, PeriodicGrid2D, TransientTerm, DiffusionTerm,
                  UniformNoiseVariable,
                  ImplicitSourceTerm, ExponentialConvectionTerm)


import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 16})


import numpy as np




def run_sim(savename,
            Br=6.8744e-4,
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
            duration=600,
            dt=1):

    # Define mesh size and number of points
    nx = 100
    ny = nx

    L = 140

    # Bulk and shear moduli
    nu, nu_b = lam ** 2 / 4, 3 * lam ** 2 / 4

    dx = L / nx
    dy = dx

    mesh = PeriodicGrid2D(dx, dy, nx, ny)


    # Variables to use
    r = CellVariable(name='r', mesh=mesh, hasOld=1)
    m = CellVariable(name='m', mesh=mesh, hasOld=1)
    u = CellVariable(name='u', mesh=mesh, hasOld=1)
    v = CellVariable(name='v', mesh=mesh, hasOld=1)


    # Add some random noise
    r.setValue(UniformNoiseVariable(mesh=mesh, minimum=0, maximum=0.01))
    m.setValue(UniformNoiseVariable(mesh=mesh, minimum=0, maximum=0.01))

    # # Or us the numpy array directly
    # r.value = np.random.uniform(0, 0.01, r.value.shape)

    # # Turn it into an array - note that y is first
    # r_vals = r.value.reshape((ny, nx))


    x_hat = [1.0, 0.0]
    y_hat = [0.0, 1.0]

    # Define the equations using their special classes
    # TransientTerm = d var / dt
    # Convection Term = u dot grad(v) - here I do it by x and y components
    # Source term for reactions - use this instead of equations for the solver to work better
    # DiffusionTerm for diffusion...

    eq_r = (TransientTerm(var=r) ==
            Br + ImplicitSourceTerm(coeff=(a / (A+r) - g * m / (G+r)), var=r)
            - ExponentialConvectionTerm(coeff=x_hat * u + y_hat * v, var=r)
            + DiffusionTerm(coeff=Dr, var=r))

    eq_m = (TransientTerm(var=m) ==
            Bm
            + ImplicitSourceTerm(coeff=ka * r, var=r)
            - ImplicitSourceTerm(coeff=kd, var=m)
            - ExponentialConvectionTerm(coeff=x_hat * u + y_hat * v, var=m)
            + DiffusionTerm(coeff=Dm, var=m))

    eq_u = (gamma * tau * TransientTerm(var=u) ==
            DiffusionTerm(coeff=[[[nu + nu_b, 0.0], [0.0, nu]]], var=u)
            + DiffusionTerm(coeff=[[[0.0, nu_b / 2], [nu_b / 2, 0.0]]], var=v)
            - ImplicitSourceTerm(coeff=gamma, var=u)
            + sigma0 * m.grad.dot(x_hat * (m0 / (m0 + m) ** 2)))

    eq_v = (gamma * tau * TransientTerm(var=v) ==
            DiffusionTerm(coeff=[[[nu, 0.0], [0.0, nu + nu_b]]], var=v)
            + DiffusionTerm(coeff=[[[0.0, nu_b / 2], [nu_b / 2, 0.0]]], var=u)
            - ImplicitSourceTerm(coeff=gamma, var=v)
            + sigma0 * m.grad.dot(y_hat * (m0 / (m0 + m) ** 2)))



    # Couple them into one big equation
    eq = eq_r & eq_m & eq_u & eq_v

    elapsed = 0.0


    # You can use the viewer when the mesh is a weird shape eg sphere
    # viewer = Viewer(vars=(r,m,u,v), datamin=0.)

    y = []
    t = []

    frame = 0


    while elapsed < duration:
        print(elapsed)
        # Old values are used for sweeping when solving nonlinear values
        r.updateOld()
        m.updateOld()
        u.updateOld()
        v.updateOld()

        elapsed += dt


        # One timestep

        # Solve nonlinear problems using sweeping a few times until error is
        # small

        res = 1e5
        old_res = res * 2
        step = 0
        while res > 1e-5 and step < 5 and old_res / res > 1.01:
            old_res = res
            res = eq.sweep(dt=dt)
            step += 1


        # Show last 2 minutes!
        # if elapsed > duration - 120 and frame % 20 == 0:

        t.append(elapsed)
        y.append([r.value.reshape((ny, nx)).copy(),
                  m.value.reshape((ny, nx)).copy(),
                  u.value.reshape((ny, nx)).copy(),
                  v.value.reshape((ny, nx)).copy()])

        frame += 1


    y = np.array(y)
    t = np.array(t)


    return np.array(y)

    
if __name__ == '__main__':

    y = run_sim(Br=0, sigma0=80)
    
    plt.pcolor(y[-1, 0])
    plt.show()
