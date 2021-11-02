#!/usr/bin/env python

import numpy as np
import sys, copy
from scipy.special import gammaincc

import matplotlib as mpl
rcparams = {}
rcparams['axes.linewidth'] = 0.5
rcparams['font.family'] = 'serif'
rcparams['font.size'] = 22
rcparams['legend.fontsize'] = 16
rcparams['mathtext.fontset'] = "stix"
mpl.rcParams.update(rcparams) # update plot parameters
import matplotlib.pyplot as plt

# Just clear format to correspond with the mathematica/Matlab calculations
def Gam(N,t):
    """ The incomplete Gamma function wrapper.
    """
    return gammaincc(N, t)

#
# Define the function that will yield the BF 
#
def BFf(S, N, D, epsilon, De=None, t=1.0, ceil=1e5, v=True):
    """ BF(S, N, D, epsilon, De=None, ceil=1e5, v=True)
    
    BF(e) : Calculation of a BF between two models M1 and M0. M1 refers  
            to a model that predicts the presence of a stochastic signal 
            (Gaussian & isotropic) in the data, while M0 corresponds to
            the noise only case. The BF(e) is calculated per frequency.
    
            This calculation assumes that the signal that is below the noise. 
            Basically the prior of the signal has been set to [-inf, t]. Depending on
            the data 'D', if the signal is stronger then the value of BF is set 
            to +inf. 
    
            For more details see: https://arxiv.org/abs/1906.09027
            
            Inputs: 
                    
                    S: PSD model of the noise (smooth curve)
                    N: Number of averages per frequency f_i of S(f_i).
                       Calculated with the LPSD algorithm.
                    D: The PSD of the data as calculated with the LPSD
                       algorithm.
                   De: The (fractional) erros of the PSD to be taken into account.
              epsilon: The epsilon is the level of confidence on the noise PSD.    
                       If len(epsilon) == 1 and (0<epsilon and epsilon<1)
                       then it is assumed that it's fractional and applies to all
                       frequencies. It can have the same shape as the data
                       for a confidence level that depends on f_i: epsilon(f_i).
                    t: Upper limit of the prior for the signal. equal to 1 by
                       default.
                 ceil: A given ceiling in order to avoid very large numbers and infinities. Any given
                       computation that yields value larger than 'ceil', is going to be replaced.
                    v: Verbose flag. Set to True by default.

            Ouputs: 
    
                 logB: The log10(BF(e)) per frequency. 
    
    NK 2019-2020
    """    
    # Init
    S  = np.array(S) 
    N  = np.array(N)
    BF = np.zeros(S.shape)
    if type(D) == list: Nc = len(D) # Get the Number of channels: if it's a list I assume each element is a channel
    elif type(D) == np.array: Nc = 1 # If it's a numpy array I assume it's a single channel
    if De is None: De = np.zeros(S.shape)
    Const = 1.0/t
    Nom   = 1.0
    Denom = 1.0

    # Print
    if v: print(' # Nc = {}, Nf = {}'.format(Nc, S.shape[0]))

    # Relative S
    if type(epsilon) == float and len([epsilon]) == 1 and (0<=epsilon and epsilon<1):
        epsilon = epsilon*S + De
    elif len(epsilon) == len(De):
        epsilon = epsilon + De
    else:
        sys.exit('\t ERROR: >>> The epsilon must be either between [0,1] or with the same shape as the data.', 
                 '\n\t        >>> Remember: It''s fractional error.')
    
    # Loop over data channels: This version is for integrating S_o in [0, t]
    for kk in range(Nc):
        Dkk   = np.array(D[kk])
        Denom = Denom * (N-2) * ( Gam( -2 + N, (Dkk*N)/(S-epsilon) ) - Gam(-2 + N, (Dkk*N)/(S+epsilon) ) )

        # As
        with np.errstate(divide='ignore', invalid='ignore'): # Avoid the divide by zero warning

            A1 = + Gam( -2 + N, (Dkk*N)/(S - epsilon) ) - Gam(-2 + N, (Dkk*N)/(t + S - epsilon) )
            A2 = - Gam( -2 + N, (Dkk*N)/(S + epsilon) ) + Gam(-2 + N, (Dkk*N)/(t + S + epsilon) )
            A  = N*Dkk*( A1 + A2 )

            # Bs
            B1 = (epsilon-S) * Gam( -2 + N, (Dkk*N)/(S - epsilon) ) + (S - epsilon + t) * Gam( -2 + N, (Dkk*N)/(t + S - epsilon) )
            B2 = (S + epsilon) * ( Gam( -2 + N, (Dkk*N)/(S + epsilon) ) - (t + S + epsilon) * Gam( -2 + N, (Dkk*N)/(t + S + epsilon) ) )
            B  = (N-2) * (B1 + B2)
            Nom = Nom * (A + B) # Get the nominator
    
    with np.errstate(divide='ignore', invalid='ignore'): # Avoid the divide by zero warning
        BF = Const * (Nom / Denom) # BF

    # Negative BFs set to zero
    zeroInd = np.argwhere(BF<0)
    if np.sum(zeroInd>0):
        BF[zeroInd] = np.absolute(BF[zeroInd])
        if v: print('\t WARNING: >>> Found negative BF values, for channel {}. Taking their ABS.'.format(kk))

    # If there are infinities, use the cap
    BF[np.isinf(BF)] = ceil

    # For better numerical results handle the NaNs
    BF[np.argwhere(np.isnan(BF))] = -np.inf

    return np.real(np.log10(BF))


#
# Define the function that will yield the BF 
#
def posterior_f(f, S, N, D, D_e=None, epsilon=.05, \
                    logylims=[-20, -11], ngrid=100, \
                        Nsigmas=1, v=True, plot=True, \
                            **contourkwargs):
    """ posterior_f()
    
    posterior(e, f) : Calculation of the posterior density of the presence of
                      a signal given a noise model and an uncertainty epsilon.
    
    
            For more details see: https://arxiv.org/abs/1906.09027
            
            Inputs: 
                    
    
            Ouputs: 
    
                 post(f): The posterior density per frequency. 
    
    NK 2019-2020
    """    
        
    gridVals = np.logspace(logylims[0], logylims[1], num=ngrid)
    if isinstance(D, np.ndarray): D = [D]
    if isinstance(S, np.ndarray): S = [S]
    Nchannls = len(D)

    # Generate lists with data for each channel, if not given properly as inputs
    if D_e is None: 
        D_e = [np.zeros((D[0].shape[0], 2)) for _ in range(len(D))]
    else:
        if isinstance(D_e, np.ndarray): D_e = [D_e]
    if len(S) == 1 and len(S) < len(D): 
        S = [S[0] for _ in range(len(D))]

    # Define the posterior lambda function
    postpdf = lambda S_o: (Gam(N-1, (N * D_hat) / ((S_hat + S_o + Nsigmas * sigma_max))) - Gam(N-1, (N * D_hat) / ((S_hat + S_o - Nsigmas * sigma_min))) ) 
    
    # Init the log-posterior grid
    logpost = np.zeros( (D[0].shape[0], gridVals.shape[0]) )

    # Loop over the data channels
    for nc in range(Nchannls):
        scale = np.max(D[nc])
        D_hat = D[nc] / scale
        sigma_min = (epsilon * S[nc] + D_e[nc][:,0]) / scale
        sigma_max = (epsilon * S[nc] + D_e[nc][:,1]) / scale
        S_hat = S[nc] / scale
        S_o_hat = gridVals / scale
        # Loop over the frequencies and fill the table with the log-posterior
        for kk in range(S_o_hat.shape[0]):
            logpost[:, kk] += np.log(postpdf(S_o_hat[kk]))

    # Calculate the evidence and normalize the posterior
    logpostn_norm = (logpost.transpose() - np.log( np.trapz( np.exp(logpost), gridVals, axis=1 ).transpose() ) ).transpose()
    # normalize all columns to the maximum of each : useful for plotting
    logpostn_norm_plot = np.nan_to_num(np.exp(logpostn_norm / np.max(logpostn_norm, axis=0))) 

    if plot: # Plot if asked for
        fig  = plt.figure(figsize=(12,8))
        ax   = fig.add_subplot(1, 1, 1)
        X, Y = np.meshgrid(f, gridVals)
        ax.loglog(f, S[0], lw=2, color='k')
        # ax.contourf(X, Y, logpostn_norm_plot.T, )
        pcm = ax.pcolor(X, Y, logpostn_norm_plot.T, **contourkwargs)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim(10**(logylims[0]), 10**(logylims[1]))
        ax.set_xlim(f[0], f[-1])
        ax.set_xlabel("Frequency [Hz]", fontsize=22)
        ax.set_ylabel("$h^2\Omega_\mathrm{GW}$ [1/Hz]", fontsize=22)

    return logpostn_norm_plot, logpost, logpostn_norm, (fig, ax)
#
# Helper function to define inputs for the DFT - Returns the Navs per frequency bin
#   
def ltf_plan(Ndata, fs=1, olap=0, Lmin=0, Jdes=500, Kdes=200, flims=None):
    """
        Helper function to define inputs for the DFT - Returns the Navs per frequency bin
        Taken from the lpsd function in the 'spectral' module in https://github.com/karnesis/spectral
        
        NK 2020
    """
    xov     = (1 - olap/100)
    fmin    = fs / Ndata 
    fmax    = fs / 2
    fresmin = fs / Ndata
    freslim = fresmin * (1+xov*(Kdes-1))
    logfact = (Ndata/2)**(1/Jdes) - 1
    fi      = fmin 
    bmin    = 1
    f, r, b, L, K = ([] for i in range(5)) # Init
    # Loop over frequencies
    while fi < fmax:
        fres = fi * logfact
        if fres <= freslim: fres = np.sqrt(fres*freslim)
        if fres < fresmin: fres = fresmin
        fbin = fi/fres
        if fbin < bmin:
            fbin = bmin
            fres = fi/fbin
        dftlen = np.round(fs / fres)
        if dftlen > Ndata: dftlen = Ndata
        if dftlen < Lmin: dftlen = Lmin
        nseg = np.round((Ndata - dftlen) / (xov*dftlen) + 1)
        if nseg == 1: dftlen = Ndata
        fres = fs / dftlen
        fbin = fi / fres
        # Store outputs
        f.append(fi)
        r.append(fres)
        b.append(fbin)
        L.append(dftlen)
        K.append(int(nseg))
        fi = fi + fres
    ind = range(len(f))
    if flims is not None: # split in frequencies
        ind = np.where(np.logical_and(np.array(f)>=flims[0], np.array(f)<=flims[1])) 
    return np.array(f)[ind], np.array(r)[ind], np.array(b)[ind], np.array(L)[ind], np.array(K)[ind]

#
# XYZ -> AET
#
def AET(xyz):
    """ Utility function to compute the AET channels
        from the XYZ TDI combinations.
    """
    X   = xyz[:,0]
    Y   = xyz[:,1]
    Z   = xyz[:,2]
    A   = (Z-X)/np.sqrt(2.0)
    E   = (X - 2.0*Y + Z)/np.sqrt(6.0)
    return A, E

#
# EVAL ON GRID 2D
#
def eval_on_grid2D(params, fun, pnames=None, plot=True, exportdata=False):
    """ A simple function to evaluate a function/ 
        criterion on a grid of parameters.

        Inputs:

            params: A list containing the arrays of parameters to compute the function on.
               fun: The function to be evaluated
            pnames: List containing the parameter names
              plot: Flag to make a plot with pre-defined style 
        exportdata: Flag to return the data if plot is needed to be done outside

        Example:

        p1 = np.linspace(-14, -12, num=20)
        p2 = np.linspace(-3, 3, num=20)
        p3 = np.linspace(0, 100, num=20)
        p0 = [-12, 0., 55.]

        # Evaluate on a grid
        paramgrid = eval_on_grid(p0, [p1, p2, p3])

    """
    p1, p2 = params[:] 
    g = np.zeros((p1.shape[0], p2.shape[0]))
    ii, jj = 0, 0
    for a in p1:
        for s in p2:
            g[ii,jj] = fun([a, s])
            jj += 1
        ii += 1
        jj = 0

    if plot:
        X,Y = np.meshgrid(p1, p2)
        fig = plt.figure(figsize=(10,7))
        ax  = fig.add_subplot(1, 1, 1)
        clr = 'lightskyblue'
        ax.contour(X, Y, g.T, 0, colors=clr, linestyles='-', alpha = 0.9,)
        cs = ax.contourf(X, Y, g.T, 1, hatches=['', '/'],colors='none',) # , cmap=cmap

        for _, collection in enumerate(cs.collections):
            collection.set_edgecolor(clr)
        for collection in cs.collections:
            collection.set_linewidth(0.)
        if pnames is None: pnames = [r'$p_1$', r'$p_2$']
        ax.set_ylabel(pnames[1], fontsize=26)
        ax.set_xlabel(pnames[0], fontsize=26)
        plt.show()

    if exportdata:
        return g
    else:
        return None

#
# EVAL ON GRID
#
def eval_on_grid(p0, grid, fun, pnames=None, plot=True, exportdata=False):
    """ A simple function to evaluate the above 
        criterion on a grid of parameters.

        Inputs:

                p0: Since we take slices, for more than 2 parameters, we have to compute the 
                  criterion given 2 parameters, but keeping the rest constant. 
              grid: a list containing the arrays of parameters to compute the criterion on.
               fun: The function to be evaluated
            pnames: List containing the parameter names
              plot: Flag to make a plot with pre-defined style 
        exportdata: Flag to return the data if plot is needed to be done outside

        Example:

        p1 = np.linspace(-14, -12, num=20)
        p2 = np.linspace(-3, 3, num=20)
        p3 = np.linspace(0, 100, num=20)
        p0 = [-12, 0., 55.]

        # Evaluate on a grid
        paramgrid = eval_on_grid(p0, [p1, p2, p3])

    """
    outgrid = []
    # Loop over the parameters
    for rr in range(0, len(grid)):
        for cc in range(rr + 1, len(grid)):

            paramgrid = np.zeros((grid[rr].shape[0], grid[cc].shape[0]))

            ii, jj = 0, 0
            for p1 in grid[rr]:
                for p2 in grid[cc]:
                    
                    params_to_eval = copy.deepcopy(p0)
                    params_to_eval[rr] = p1 # I feel like there is smarter way of doing this
                    params_to_eval[cc] = p2

                    paramgrid[ii,jj,] = fun(params_to_eval)
                    jj += 1
                ii += 1
                jj = 0
                if exportdata: outgrid.append(outgrid)

            if plot:
                fig  = plt.figure(figsize=(10,7))
                ax   = fig.add_subplot(1, 1, 1)
                clr  = 'lightskyblue'
                X, Y = np.meshgrid(grid[rr], grid[cc])
                ax.contour(X, Y, paramgrid.T, 0, colors=clr, linestyles='-', alpha = 0.9,)
                cs = ax.contourf(X, Y, paramgrid.T, 1, hatches=['', '/', '\\', '//'],colors='none',) # , cmap=cmap

                # For each level, we set the color of its hatch 
                for _, collection in enumerate(cs.collections):
                    collection.set_edgecolor(clr)
                for collection in cs.collections:
                    collection.set_linewidth(0.)
                if pnames is None: pnames = [r'$p_{}$'.format(ii+1), r'$p_{}$'.format(jj+1)]
                ax.set_ylabel(pnames[cc], fontsize=22)
                ax.set_xlabel(pnames[rr], fontsize=22)
                plt.show()
    if exportdata:
        return outgrid
    else:
        return None