#!/usr/bin/env python

import numpy as np
import sys
import LISAParameters as LP

#
# Define the signal model class
#
class powerlaw:
    """
    Class powerlaw (f, refFreq=25.0)

        A model class to calculate a power-law signal at different amplitude 
        and slope parameters. 

        Inputs:

            f: The frequency array of the analysis
        refFreq: The reference frequency of the power law (pivot frequency)
        
        Outputs: An instance of the class.

        methods: The 'eval' method evaluates the model at the given parameters

        Example: 

                    plmodel = powerlaw(fvec, refFreq=25.0)
                    Shnum   = plmodel.eval([logamplitude, slope])

        NK 2020
    """
    def __init__(self, f, refFreq=3e-3):
        self._freqs   = f
        self._refFreq = refFreq
    def eval(self, theta):
        """
        Process model function, or evaluate.

        positional arguments:

            theta: numpy vector (or list) of length of 2, of the input parameter values. 
                   theta[0] is the log-amplitude, and theta[1] is the slope. 
        """
        # Evaluate and return 
        return ( 10**theta[0] * (self._freqs/self._refFreq)**theta[1] )

class broken_powerlaw:
    """
    Class broken_powerlaw (f)
    
    A model class to calculate a broken power-law signal at different amplitude 
        and slope parameters. 

        Inputs:

            f: The frequency array of the analysis
        
        Outputs: An instance of the class.

        methods: The 'eval' method evaluates the model at the given parameters

        Example: 

                    plmodel = broken_powerlaw(fvec)
                    Shnum   = plmodel.eval([logamplitude, slope, slope])

        NK 2021
    """
    def __init__(self, freq, fstar=1e-3):

        self._f = freq
        self._fstar = fstar

    def eval(self,theta):
        """
        Process model function, or evaluate.

        positional arguments:

            theta: numpy vector (or list) of length of n, of the input parameter values. 
                   theta[0] is the log-amplitude, and theta[1:2] is the slopes. 
        """
        logOm, n1, n2 = theta[:]

        # Evaluate the first order - a single power law
        Sh = 10**logOm * ( self._f**(n1 + n2) / (self._fstar**n1 * self._f**n2 + self._fstar**n2 * self._f**n1) )
        return Sh

class broken_powerlaw_generic:
    """
    Class broken_powerlaw_generic (f, Delta=.05)
    
    A model class to calculate a broken power-law signal at different amplitude 
        and slope parameters. 

        Inputs:

            f: The frequency array of the analysis
        Delta: Smoothing parameter for the powerlaws intersection point
      refFreq: The reference frequency of the first-order powerlaw (pivot frequency)
        
        Outputs: An instance of the class.

        methods: The 'eval' method evaluates the model at the given parameters

        Example: 

                    plmodel = broken_powerlaw_generic(fvec, Delta=.8)
                    Shnum   = plmodel.eval([logamplitude, slope])

        NK 2021
    """
    def __init__(self, freq, refFreq=2e-3, Delta=0.05):

        self._freqs   = freq
        self._Delta   = Delta
        self._refFreq = refFreq

    def eval(self,theta):
        """
        Process model function, or evaluate.

        positional arguments:

            theta: numpy vector (or list) of length of n, of the input parameter values. 
                   theta[0] is the log-amplitude, and theta[1] is the slope. Then, for higher
                   orders (many broken laws) the parameters are
                   [Amplitude, slope, freq_break, slope, freq_break, slope, freq_break, ...] 
        """

        # Enumerate the parameters: [Amplitude, slope, fb, slope, fb, slope, fb, ...]
        Nparams = len(theta)

        # Evaluate the first order - a single power law
        Sh = 10**theta[0] * self._freqs ** (-theta[1]) # * (self._freqs/self._refFreq)**theta[1]

        # Generate the model depending on the demensionality
        if Nparams > 2:
            for ii in range(2, Nparams, 2):
                if ii==2:
                    Sh = Sh * (1/(theta[ii] ** (-theta[ii-1]))) * ( 0.5 * ( 1.0 + (self._freqs/theta[ii])**(1/self._Delta) )  ) ** ( (theta[ii-1] - theta[ii+1])*self._Delta )
                else:
                    Sh = Sh * ( 0.5 * ( 1.0 + (self._freqs/theta[ii])**(1/self._Delta) )  ) ** ( (theta[ii-1] - theta[ii+1])*self._Delta )
        return Sh


#
# The response
#
def response(f):
    """ Sky averaged response of the LISA constellation. 

        This is a simplified assumption that we have to make.
        For durations less than a few months the response is not
        the same for both A and E channels.
    """
    x = 2.0 * np.pi * LP.lisaLT * f 
    return np.absolute(9/20 /(1 + (3 *x/4)**2 ) * ((16 * x**2 * np.sin(x)**2)))

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