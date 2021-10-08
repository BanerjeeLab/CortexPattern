# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 09:33:22 2018

@author: Mike Staddon

Solve the rhoA RGA3/4 equation in 0D
"""

from scipy.integrate import odeint, LSODA, RK45
import scipy.optimize

import numpy as np

import plotting

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 16})

""" Wrapped functions """
def solve(f, t_eval, y0, shape, jac=None):
    solver = RK45(f, t_eval[0], y0, t_eval[-1], vectorized=True)
    
    # Code taken from solve_ivp source
    ts = []
    ys = []
    
    t_eval_i = 0
    
    status = None
    while status is None:
        solver.step()
    
        if solver.status == 'finished':
            status = 0
        elif solver.status == 'failed':
            status = -1
            break
    
        # Fix negative concentrations!
        t = solver.t
        y = solver.y
        
        y = y.reshape(shape)
        
    #            y[y[:, 0] < 0, 0] = 0
    #            y[y[:, 1] < 0, 1] = 0
        
        y[np.isnan(y)] = 0
    
        y = y.flatten()
        solver.y = y
    
        # The value in t_eval equal to t will be included.
        t_eval_i_new = np.searchsorted(t_eval, t, side='right')
        t_eval_step = t_eval[t_eval_i:t_eval_i_new]
    
        if t_eval_step.size > 0:
            sol = solver.dense_output()
            ts.append(t_eval_step)
            ys.append(sol(t_eval_step).T)
            t_eval_i = t_eval_i_new
            
    #                print(t_eval_step)
            
    #                print(t_eval_i)
    
    ts = np.hstack(ts)
    ysol = np.vstack(ys)
    
    return ts, ysol


""" ODE """

class System0D():
    """ Defines the 0D system for RhoA - myosin - contraction feedback 
    
    Parameters in units of seconds from Ed Munro
    
    Cell mechanics mu from Measure Properties of Cells - see 1D
    """
    def __init__(self,
                 n1=1,
                 n2=2,
                 Br=6.8744e-4,
                 a=0.1609,
                 A=0.3833,
                 g=0.1787,
                 G=0.01,
                 Bm=0.0076,
                 ka=0.1408,
                 kd=0.0828,
                 mu=5,
                 ku=1,
                 sigma0=0.2,
                 m0=1,
                 n3=1):
        
        self.pars = n1, n2, a, A, g, G, Bm, ka, kd, Br, mu, ku, sigma0, m0, n3

    
    def dr(self, r, m, u):
        """ dr / dt """
        n1, n2, a, A, g, G, Bm, ka, kd, Br, mu, ku, sigma0, m0, n3 = self.pars
        return Br + a * r**n1 / (A + r**n1) - g * m * r / (G + r) - r/(1+u) * self.du(r, m, u)
    
    
    def dm(self, r, m, u):
        """ dm / dt """
        n1, n2, a, A, g, G, Bm, ka, kd, Br, mu, ku, sigma0, m0, n3 = self.pars
        return Bm + ka * r**n2 - kd * m - m/(1+u) * self.du(r, m, u)
    

    def du(self, r, m, u):
        """ du / dt """
        n1, n2, a, A, g, G, Bm, ka, kd, Br, mu, ku, sigma0, m0, n3 = self.pars
        return (-ku * u - sigma0 * m**n3/(m0**n3 + m**n3))/mu
    
    
    def nullcline_r(self, r):
        """ Returns the values of m for which dr(r, m) = 0 """
        n1, n2, a, A, g, G, Bm, ka, kd, Br, mu, ku, sigma0, m0, n3 = self.pars
        
        
        m = (Br + a * r**n1 / (A + r**n1)) * (G + r) / g

        # Replace dividing by zero with "infinite"
        return np.divide(m, r, out=np.zeros_like(m)+5, where=r!=0)
    
    
    def nullcline_m(self, r):
        """ Returns the values of m for which dm(r, m) = 0 """
        n1, n2, a, A, g, G, Bm, ka, kd, Br, mu, ku, sigma0, m0, n3 = self.pars
        
        return (Bm + ka * r ** n2) / kd
    
    
    def Jac3(self, r, m, u):
        """ The Jacobian d(du/dt)/ d (r, m, u). Speeds up numerical integration
        by a lot!
        """
        n1, n2, a, A, g, G, Bm, ka, kd, Br, mu, ku, sigma0, m0, n3 = self.pars
        
        # The others depend on du/dt so calculate first
        j3 = [0,
              -sigma0 * n3 / mu * m0**n3 * m**(n3 - 1) / (m0**n3 + m**n3)**2,
              -ku / mu]
    
        # d(dr/dt)/ d (r, m, u)
        j1 = [n1 * a * A * r**(n1-1)/(A + r**n1)**2 - g * m * G / (G + r)**2 - 1/(1+u) * self.du(r, m, u),
              -g * r / (G + r) - r/(1+u) * j3[1],
              1/(1+u)**2 * self.du(r, m, u) - r/(1+u) * j3[2]]
    
        # d(dm/dt)/ d (r, m, u)
        j2 = [ka * n2 * r**(n2-1),
              -kd - 1/(1+u) * self.du(r, m, u) - m/(1+u) * j3[1],
              1/(1+u)**2 * self.du(r, m, u) - m/(1+u) * j3[2]]
    
        return np.array([j1, j2, j3])
    
    
    # dy/dt = f
    def f(self, y, t):
        """ Defines the differential equation
            y: [r, m, u]
                the state of rhoA and myosin and strain
            t: float
                time - doesn't actually change the simulation but is needed
        """
    
        r, m, u = y
    
        return [self.dr(r, m, u), self.dm(r, m, u), self.du(r, m, u)]


    def solve_ode(self, y0=None, tend=60, dt=0.1):
        """ Solve the ode and return solution
        
        Arguements:
            y0: [r0, m0, u0], optional
                initial conditions, defaults to [0, 0, 0]
            stop_time: float
                maximum time to calculate
            num_points: int
                number of timesteps between
                
        Returns:
            t: array
                time points
            ysol: array, [r, m, u]
                solution
        """

        y0 = y0 if y0 is not None else [0, 0, 0]
        t = np.arange(0, tend + dt, dt)
        ysol = odeint(self.f, y0, t)
    
        return t, ysol


    def stationary_points(self, rmin=0, rmax=10):
        """ Find the stationary points of the system for uniform concentrations.
        Same as the 0D case
        
        Arguments:
            rmin: float
                minimum rhoA value to start from
            rmax: float
                maximum rhoA value to start from
                
        Returns:
            roots: list of lists
                up to 3 lists of [r, m, u]
        """
        
        n1, n2, a, A, g, G, Bm, ka, kd, Br, mu, ku, sigma0, m0, n3 = self.pars
    
        # Uniform roots are the same as in 1D
        r = np.linspace(rmin, rmax, 10000)
    
        def y(x):
            """ This is found by solving for steady state in terms of rhoA """
            c1 = g * ka / kd
            c2 = g * A * ka / kd
            c3 = g * Bm / kd - a - Br
            c4 = - a  * G - Br * G
            c5 = g * Bm * A / kd - Br * A
            c6 = -Br * A * G
        
            return c1 * x**(n2+n1+1) + c2 * x**(n2+1) + c3 * x**(n1+1) + c4 * x**n1 + c5 * x + c6
        
        rs, ms, vs = [], [], []

        # For each axis crossing, find a root
        for i in range(len(r)-1):
            if y(r[i]) * y(r[i+1]) <= 0:
                root_r = scipy.optimize.root(y, r[i])
                
                rs += [root_r.x[0]]
                
        rs = np.array(rs)
        ms = (Bm + ka * rs**n2)/kd
        vs = 0 * rs
        
        return rs, ms, vs
    
    
    def get_stability(self, rmin=0, rmax=10):
        """ Check the linear stability of each point. Find the maximum eigenvalue
        for different wave numbers.
        
        Arguments:
            rmin: float
                minimum bound to find roots in
            rmax: float
                maximum bound to find roots in
            plot: bool
                plot eigenvalue as a function of wavenumber
            verbose: bool
                print results
            savename: str
                if not None, save plot. save with this name + modifiers
        
        Returns:
            max_eigs: list
                list of meximum (real) eigenvalue for each root
        """
        n1, n2, a, A, g, G, Bm, ka, kd, Br, mu, ku, sigma0, m0, n3 = self.pars
        
        rs, ms, vs = self.stationary_points(rmin, rmax)
    
        # Only use last steady state point
        r, m, v = rs[-1], ms[-1], vs[-1]
        

        jac = self.Jac3(r, m, v)
        return np.linalg.eig(jac)[0]
    
    
    def plot_nullclines(self, rmin=0, rmax=1, mmin=0, mmax=1, ysol=None,
                        stream=True, ls=None):
        """
        Args are limits to plot nullclines
        """
        
        # Colors
        col_r = plotting.rho_color
        col_m = plotting.myo_color
        
        n1, n2, a, A, g, G, Bm, ka, kd, Br, mu, ku, sigma0, m0, n3 = self.pars
    
        r = np.linspace(rmin, rmax, 17)
        m = np.linspace(mmin, mmax, 17)
        
        R, M = np.meshgrid(r, m)
        
        # Assume u always in equilibrium
        U = -sigma0/ku * M**n3 / (m0**n3 + M**n3)
        
        u, v = self.dr(R, M, U), self.dm(R, M, U)
        
        r = np.linspace(0, rmax, 10001)
        plt.plot(r, self.nullcline_r(r), lw=3, color=col_r, ls=ls)
        plt.plot(r, self.nullcline_m(r), lw=3, color=col_m, ls=ls)
        
        if stream:
            plt.quiver(R, M, u, v, color=(0.5, 0.5, 0.5))
            
            # Plot stationary points
            rs, ms, _ = self.stationary_points()
            
            for i in range(len(rs)):
                print(i)
                if i % 2 == 0:
                    fc = 'k'
                else:
                    fc = 'w'
                    
                plt.scatter([rs[i]], [ms[i]],
                            color='k', facecolor=fc, s=36, zorder=10, lw=1.5)
        
        if ysol is not None:
            print(ls)
            plt.plot(ysol[:,0], ysol[:,1], color='k', lw=3, zorder=12, ls=ls)
            
            for i in range(50, len(ysol), 100):
                plt.arrow(ysol[i, 0], ysol[i, 1],
                          ysol[i, 0] - ysol[i-1, 0], ysol[i, 1] - ysol[i-1, 1],
                          head_width=0.04 * rmax,
                          color='k',
                          zorder=100)
        
        # Plot settings
        plt.xticks([0.5 * i for i in range(11)])
        plt.yticks([0.5 * i for i in range(11)])
        plt.xlim(rmin - 0.01, rmax + 0.01)
        plt.ylim(mmin - 0.01, mmax + 0.01)
        
        plt.xlabel('RhoA Concentration r')
        plt.ylabel('Myosin Concentration m')
        
        plotting.set_ax_size(3.5, 3.5)
        
        
    def plot_lognullclines(self, rmin=1e-4, rmax=10, mmin=1e-2, mmax=10,
                           ysol=None, label=None,
                           stream=True, ls=None):
        """
        Args are limits to plot nullclines
        """
        
        # Colors
        col_r = plotting.rho_color
        col_m = plotting.myo_color
        
        n1, n2, a, A, g, G, Bm, ka, kd, Br, mu, ku, sigma0, m0, n3 = self.pars
    
        r = np.logspace(np.log10(rmin), np.log10(rmax), 17)
        m = np.logspace(np.log10(mmin), np.log10(mmax), 17)
        
        R, M = np.meshgrid(r, m)
        
        # Assume u always in equilibrium
        U = -sigma0/ku * M**n3 / (m0**n3 + M**n3)
        
        u, v = self.dr(R, M, U), self.dm(R, M, U)
        
        mag = np.hypot(u, v)
        
        u, v = u / mag, v / mag
        
        r = np.logspace(np.log10(rmin), np.log10(rmax), 1001)
        plt.loglog(r, self.nullcline_r(r), lw=3, color=col_r, ls=ls)
        plt.loglog(r, self.nullcline_m(r), lw=3, color=col_m, ls=ls)
        
        if stream:
            plt.quiver(R, M, u, v, color=(0.5, 0.5, 0.5))
            
            # Plot stationary points
            rs, ms, _ = self.stationary_points()

            for i in range(len(rs)):
                print(i)
                if i % 2 == 0:
                    fc = 'k'
                else:
                    fc = 'w'
                    
                plt.scatter([rs[i]], [ms[i]],
                            color='k', facecolor=fc, s=50, zorder=10, lw=3)
        
        if ysol is not None:
            plt.plot(ysol[:,0], ysol[:,1], color='k', lw=3, zorder=12, ls=ls,
                     label=label)
            
            for i in range(50, len(ysol), 200):
                
                plt.annotate('',
                             ysol[i, :2], ysol[i-1, :2],
                             arrowprops={'facecolor':'black',
                                         'headwidth': 12})
        
        # Plot settings
        plt.xlim(rmin, rmax)
        plt.ylim(mmin, mmax)
        
        plt.xlabel('RhoA Concentration r')
        plt.ylabel('Myosin Concentration m')
        
        plotting.set_ax_size(3.5, 3.5)
        
        
if __name__ == '__main__':
    ode = System0D()
    t, y = ode.solve_ode()
    
    plt.plot(t, y[:, 0])
    plt.show()
