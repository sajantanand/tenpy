"""Implementation of the FFID beyond JW  model on a chain."""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np

from .model import CouplingMPOModel
from .lattice import Chain
from ..networks.site import SpinSite

__all__ = ['FFIDChain']


class FFIDChain(CouplingMPOModel):
    r""" The Free fermion in disguise (FFID) beyond Jordan-Wigner (JW) model is introduced
    in :doi:`10.21468/SciPostPhys.16.4.102` and interpolates between the FFID model from
    :doi:`10.1088/1751-8121/ab305d` and the DFNR model from 
    :doi:`10.1088/1742-5468/2016/02/023104`.

    The Hamiltonian reads:

    .. math ::
        H =  \mathtt{b} Z_0 Z_{2} + \sum_{i=1}^{L-3} X_i X_{i+1} Z_{i+2}
            + \mathtt{b}^2 Z_{i-1} Y_i Y_{i+1} + \mathtt{b} Z_i Z_{i+2}

    """

    default_lattice = Chain
    force_default_lattice = True  # we implicitly assume a 1D chain,
    # otherwise more P's need to be added

    def init_sites(self, model_params):
        sort_charge = model_params.get('sort_charge', True, bool)
        s = SpinSite(S=0.5, conserve=None, sort_charge=sort_charge)
        return s

    def init_terms(self, model_params):
        b = model_params.get('b', 0.)
        L = model_params['L']

        for i in range(L-2):
            self.add_local_term(b*4, [('Sz', [i, 0]), ('Sz', [i+2, 0])])
        for i in range(1, L-2):
            self.add_local_term(1*8, [('Sx', [i, 0]), ('Sx', [i+1, 0]), ('Sz', [i+2, 0])])
        for i in range(0, L-3):
            self.add_local_term(b**2*8, [('Sz', [i, 0]), ('Sy', [i+1, 0]), ('Sy', [i+2, 0])])
