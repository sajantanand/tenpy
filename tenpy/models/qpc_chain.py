"""Spin 1/2 chain describing quantum point contact.

Two reservoirs of XXZ chains coupled to central spin by nearest-neighbor interactions.
"""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np

from ..networks.site import SpinHalfSite
from .model import CouplingMPOModel, NearestNeighborModel
from .lattice import Chain
from ..tools.params import asConfig

__all__ = ['QPCChain']


class QPCChain(CouplingMPOModel, NearestNeighborModel):
    r"""Spin-1/2 sites coupled by to approximate quantum point contact

    The Hamiltonian reads:

    .. math ::
        H = \sum_{\langle i,j\rangle, i < j \in L, R}
              (\mathtt{t} S^x_i S^x_j + \mathtt{t} S^y_i S^y_j + \mathtt{Delta} S^z_i S^z_j) \\
            + \sum_{i \in L} \mathtt{muL} S^z_i + \sum_{i \in R} \mathtt{muR} S^z_i \\
            + \mathtt{tL} S^{x}_{L//2-1} S^{x}_{L//2} + \mathtt{tL} S^{y}_{L//2-1} S^{y}_{L//2} \\
            + \mathtt{tR} S^{x}_{L//2} S^{x}_{L//2+1} + \mathtt{tR} S^{y}_{L//2} S^{y}_{L//2+1} \\
            + \mathtt{Vd} S^z_{L//2} \\

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs.
    Site :math:`L//2` is the center dot cite, while `L` and `R` refer to the two reservoir chains.
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`SpinModel` below.

    Options
    -------
    .. cfg:config :: QPCChain
        :include: CouplingMPOModel

        conserve : 'best' | 'Sz' | None
            What should be conserved. See :class:`~tenpy.networks.Site.SpinHalfSite`.
            For ``'best'``, preserve spin magnetizaiton.
        sort_charge : bool
            Whether to sort by charges of physical legs. `True` by default.
        t, Delta, muL, muR, tL, tR, Vd : float | array
            Coupling as defined for the Hamiltonian above.

    """
    
    default_lattice = Chain
    force_default_lattice = True
   
    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'best', str)
        if conserve == 'best':
            conserve = 'Sz'
            self.logger.info('%s: set conserve to %s', self.name, conserve)
        sort_charge = model_params.get('sort_charge', True, bool)
        site = SpinHalfSite(conserve=conserve, sort_charge=sort_charge)
        return site

    def init_terms(self, model_params):
        L = model_params['L']
        assert L % 2 == 1, "System size must be odd."

        t = model_params.get('t', -1.0, 'real_or_array')
        tL = model_params.get('tL', -1.0, 'real')
        tR = model_params.get('tR', -1.0, 'real')
        Delta = model_params.get('Delta', 0.0, 'real_or_array')
        muL = model_params.get('muL', 0.0, 'real_or_array')
        muR = model_params.get('muR', 0.0, 'real_or_array')
        Vd = model_params.get('Vd', 0.0, 'real') 

        # Left onsite
        for u in range(L//2):
            self.add_onsite_term(muL, u, 'Sz')
        # Right onsite
        for u in range(L//2+1,L):
            self.add_onsite_term(muR, u, 'Sz')
        # Dot onsite
        self.add_onsite_term(Vd, L//2, 'Sz')
        
        # Sp = Sx + i Sy, Sm = Sx - i Sy,  Sx = (Sp+Sm)/2, Sy = (Sp-Sm)/2i
        # Sx.Sx = 0.25 ( Sp.Sm + Sm.Sp + Sp.Sp + Sm.Sm )
        # Sy.Sy = 0.25 ( Sp.Sm + Sm.Sp - Sp.Sp - Sm.Sm )
        # Left coupling
        for i in range(L//2-1):
            self.add_coupling_term(t/2, i, i+1, 'Sp', 'Sm', plus_hc=True)
            self.add_coupling_term(Delta, i, i+1, 'Sz', 'Sz', plus_hc=False)
        # Right coupling
        for i in range(L//2+1,L-1):
            self.add_coupling_term(t/2, i, i+1, 'Sp', 'Sm', plus_hc=True)
            self.add_coupling_term(Delta, i, i+1, 'Sz', 'Sz', plus_hc=False)
        # Dot coupling
        self.add_coupling_term(tL/2, L//2-1, L//2, 'Sp', 'Sm', plus_hc=True)
        self.add_coupling_term(tR/2, L//2, L//2+1, 'Sp', 'Sm', plus_hc=True)
        # done
