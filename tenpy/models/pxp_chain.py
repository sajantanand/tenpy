import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy

import tenpy
from .lattice import Site, Chain
from ..linalg import np_conserved as npc
from ..networks.site import SpinHalfSite, SpinSite  # if you want to use the predefined site
from .model import CouplingModel, NearestNeighborModel, MPOModel, CouplingMPOModel
from .lattice import TrivialLattice, Chain
from ..tools.params import asConfig

#from tenpy.networks.site import BosonSite

__all__ = ['PXPChain', 'RydbergSpinHalfSite']

class PXPChain(CouplingMPOModel):
    """Implementation of spin-1/2 PXP model on a chain using spin-1/2 site

    Neither Sz nor parity can be conserved as sigma_x violates both.

    Parameters
    ----------
    model_params : dict | :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`PXPChain` below.

    Options
    -------
    .. cfg:config :: PXPChain
        :include: CouplingMPOModel

        L : int
            Length of the chain.
        Omega : float
            Strength of the drive; just sets overall scale if no perturbations added.
        bc_MPS : {'finite' | 'infinite'}
            MPS boundary conditions. Coupling boundary conditions are chosen appropriately.
        sort_charge : bool | None
            Whether to sort by charges of physical legs.
            See change comment in :class:`~tenpy.networks.site.Site`.
    """
    default_lattice = "Chain"
    force_default_lattice = True

    def init_sites(self, model_params):
        sort_charge = model_params.get('sort_charge', None)
        return SpinHalfSite(conserve='None', sort_charge=sort_charge)  # use predefined Site

    def init_terms(self, model_params):
        # read out parameters
        Omega = model_params.get('Omega', 1.)
        # add terms
        self.add_multi_coupling(Omega, [('P0', 0, 0), ('Sigmax', 1, 0), ('P0', 2, 0)], category='PXP')


class RydbergSpinHalfSite(SpinHalfSite):
    """
    We want to do two things:
        (1) Add projectors for the ground and excited state, the first of which is used for the PXP model.
        (2) Conserve Rydberg charge
    """
    def __init__(self, ind, Nmax=1, conserve='Exp_N', p=2, PBC=False, L=None):
        conserve2 = deepcopy(conserve)
        if conserve == "Exp_N":
            conserve2 = 'None'
        BosonSite.__init__(self, Nmax=Nmax, conserve=conserve2, filling=0.)
        def powm(A, k):
            M = A.copy()
            for i in range(1, k):
                # Not the most efficient way to compute the power, but that's ok.
                M = npc.tensordot(M, A, axes=(['p*', 'p']))
            return M
        for i in range(1, Nmax):
            self.add_op('Bd' + str(i+1), powm(self.Bd, i+1))
            self.add_op('B' + str(i+1), powm(self.B, i+1))
        if p is None:
            p = 1
        if conserve == 'Exp_N':
            self.conserve=conserve
            if PBC:
                mod = p**L - 1
                chinfo = npc.ChargeInfo([mod], ['Exp_N'])
                leg = npc.LegCharge.from_qflat(chinfo, [(i * p**ind) % mod for i in range(self.dim)])
            else:
                chinfo = npc.ChargeInfo([1], ['Exp_N'])
                leg = npc.LegCharge.from_qflat(chinfo, [i * p**ind for i in range(self.dim)])
            self.change_charge(leg)
            self.ind = ind
        self.p = p # Set this so that we can do exponential hoppings without charge conservation.

    def __repr__(self):
        return "ModulatedBosonSite(Nmax={Nmax!s}, {c!r}, p={p!s}, ops={ops!r})".format(Nmax=self.Nmax, c=self.conserve, p=self.p, ops=self.hc_ops)

class ExponentiallyModulatedBosonModel(CouplingMPOModel):
    r"""Spinless Bose-Hubbard model with exponential hoppings and chemical potential.

    The Hamiltonian is:

    .. math ::
        H = - gamma \sum_{\langle i, j \rangle, i < j} (b_i^{\dagger} b_j + b_j^{\dagger} b_i)
            - J \sum_{\langle i, j \rangle, i < j} (b_i^{\dagger}^LP b_j^RP + b_i^LP b_j^{\dagger}^RP)
            + V \sum_{\langle i, j \rangle, i < j} n_i n_j
            + \frac{U}{2} \sum_i n_i (n_i - 1) - \mu \sum_i n_i - \mu_E \sum_i n_i p^i

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs.
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`BoseHubbardModel` below.

    Options
    -------
    .. cfg:config :: BoseHubbardModel
        :include: CouplingMPOModel

        n_max : int
            Maximum number of bosons per site.
        conserve: {'best' | 'N' | 'parity' | 'Exp_N' | None}
            What should be conserved. See :class:`~tenpy.networks.Site.ModulatedBosonSite` (or whereever this is).
        gamma, J, U, V, mu, mu_E: float | array
            Couplings as defined in the Hamiltonian above. Note the signs!
        left_power, right_power, p: int
            powers of the exotic hopping term that gives rise to an exponentially modulated symmetry Q = sum_i p^i Q_i
        phi_ext : float
            For 2D lattices and periodic y boundary conditions only.
            External magnetic flux 'threaded' through the cylinder. Hopping amplitudes for bonds
            'across' the periodic boundary are modified such that particles hopping around the
            circumference of the cylinder acquire a phase ``2 pi phi_ext``.
    """
    def init_sites(self, model_params):
        Nmax = model_params.get('Nmax', 1)
        conserve = model_params.get('conserve', 'Exp_N')
        p = model_params.get('p', 2)
        L = model_params['L']
        PBC = model_params.get('PBC', False)

        # Since each site can have different charges (if we use 'Exp_N'), we need a different `site` per site in the chain
        # to represent the different Hilbert space.
        sites = [ModulatedBosonSite(i, Nmax=Nmax, conserve=conserve, p=p, PBC=PBC, L=L) for i in range(L)]
        return sites

    def init_lattice(self, model_params):
        """
        Build trivial lattice which has a single unit cell and the exponentially modulated sites.
        """
        sites = self.init_sites(model_params)
        NN_pairs = model_params['NN_pairs']
        NNN_pairs = model_params.get('NNN_pairs', [])
        NNNN_pairs = model_params.get('NNNN_pairs', [])
        L = model_params['L']

        pairs = {
            'nearest_neighbors': NN_pairs,
            'next_nearest_neighbors': NNN_pairs,
            'next_next_nearest_neighbors': NNNN_pairs
        }

        lat = TrivialLattice(sites, pairs=pairs)
        return lat

    def init_terms(self, model_params):
        J = model_params.get('J', 1.)
        gamma = model_params.get('gamma', 0.)
        U = model_params.get('U', 0.)
        V = model_params.get('V', 0.)
        mu = model_params.get('mu', 0)
        mu_E = model_params.get('mu_E', 0)
        phi_ext = model_params.get('phi_ext', None)
        p = model_params.get('p', 2)

        left_power = model_params.get('left_power', 2)
        right_power = model_params.get('right_power', 1)
        if model_params.get('conserve', 'Exp_N') == 'Exp_N':
            assert left_power / right_power == p
        op_left = 'B' + (str(left_power) if left_power > 1 else '')
        op_right = 'Bd' + (str(right_power) if right_power > 1 else '')

        for u in range(len(self.lat.unit_cell)):
            if p is not None:
                # Since TeNPy requires integer charges, we need left_power / right_power == p such that charge per site increases to the right
                # But if we don't use 'Exp_N', we should be able to use fractional p.
                self.add_onsite(-mu_E * p**u, u, 'N', category='N')
            self.add_onsite(-mu - U / 2., u, 'N', category='N')
            self.add_onsite(U / 2., u, 'NN', category='NN')
        # Technically adds the terms to each unit cell, but since there is only one unit cell, we add the couplings only once.
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            if phi_ext is None:
                hop_J = -J
                hop_gamma = -gamma
            else:
                raise NotImplementedError()
                hop_J = self.coupling_strength_add_ext_flux(-J, dx, [0, 2 * np.pi * phi_ext])
                hop_gamma = self.coupling_strength_add_ext_flux(-J, dx, [0, 2 * np.pi * phi_ext])
            self.add_coupling(hop_J, u1, op_left, u2, op_right, dx, plus_hc=True)
            self.add_coupling(hop_gamma, u1, 'Bd', u2, 'B', dx, plus_hc=True)
            self.add_coupling(V, u1, 'N', u2, 'N', dx)

class ExponentiallyModulatedBosonChain(ExponentiallyModulatedBosonModel, NearestNeighborModel):
    """The :class:`ExponentiallyModulatedBosonModel` on a Chain, suitable for TEBD.

    See the :class:`ExponentiallyModulatedBosonModel` for the documentation of parameters.
    """

    default_lattice = "Chain"
    force_default_lattice = True
