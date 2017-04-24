# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la


class Kernel(object):
    """
    Implemented following definitions at found at http://en.wikipedia.org/wiki/Support_vector_machine
    
    For these ones do I have to choose my kernel and pass arguments defined in the outer def, 
    later passing x and y args to the  nested f function? How does this work?
    """
    @staticmethod
    def linear_kernel(**kwargs):
        def f(x1, x2):
            return np.inner(x1, x2) # inner is the dot product
        return f

    @staticmethod
    def gaussian(sigma):
        def f(x, y):
            left = la.norm(x-y) ** 2
            right = (2 * sigma ** 2)
            exponent = np.sqrt(left / right) * -1
            return np.exp(exponent)
        return f

    @staticmethod
    def polynomial_kernel(exponent, coef, **kwargs):
        def f(x1, x2):
            dotProduct = np.inner(x1, x2)
            return (dotProduct + coef)**exponent
        return f

