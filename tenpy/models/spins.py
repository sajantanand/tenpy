"""Nearest-neighbor spin-S models.

Uniform lattice of spin-S sites, coupled by nearest-neighbor interactions.
"""
# Copyright 2018-2023 TeNPy Developers, GNU GPLv3

import numpy as np

from ..networks.site import SpinSite
from .model import CouplingMPOModel, NearestNeighborModel
from .lattice import Chain, Square
from ..tools.params import asConfig

__all__ = ['SpinModel', 'SpinChain', 'ExponentiallyDecayingXXZ', 'AnisotropicSpinModel']


class SpinModel(CouplingMPOModel):
    r"""Spin-S sites coupled by nearest neighbor interactions.

    The Hamiltonian reads:

    .. math ::
        H = \sum_{\langle i,j\rangle, i < j}
              (\mathtt{Jx} S^x_i S^x_j + \mathtt{Jy} S^y_i S^y_j + \mathtt{Jz} S^z_i S^z_j
            + \mathtt{muJ} i/2 (S^{-}_i S^{+}_j - S^{+}_i S^{-}_j))  \\
            - \sum_i (\mathtt{hx} S^x_i + \mathtt{hy} S^y_i + \mathtt{hz} S^z_i) \\
            + \sum_i (\mathtt{D} (S^z_i)^2 + \mathtt{E} ((S^x_i)^2 - (S^y_i)^2))

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs.
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`SpinModel` below.

    Options
    -------
    .. cfg:config :: SpinModel
        :include: CouplingMPOModel

        S : {0.5, 1, 1.5, 2, ...}
            The 2S+1 local states range from m = -S, -S+1, ... +S.
        conserve : 'best' | 'Sz' | 'parity' | None
            What should be conserved. See :class:`~tenpy.networks.Site.SpinSite`.
            For ``'best'``, we check the parameters what can be preserved.
        sort_charge : bool | None
            Whether to sort by charges of physical legs.
            See change comment in :class:`~tenpy.networks.site.Site`.
        Jx, Jy, Jz, hx, hy, hz, muJ, D, E  : float | array
            Coupling as defined for the Hamiltonian above.

    """
    def init_sites(self, model_params):
        S = model_params.get('S', 0.5)
        conserve = model_params.get('conserve', 'best')
        if conserve == 'best':
            # check how much we can conserve
            if not model_params.any_nonzero([('Jx', 'Jy'), 'hx', 'hy', 'E'],
                                            "check Sz conservation"):
                conserve = 'Sz'
            elif not model_params.any_nonzero(['hx', 'hy'], "check parity conservation"):
                conserve = 'parity'
            else:
                conserve = None
            self.logger.info("%s: set conserve to %s", self.name, conserve)
        sort_charge = model_params.get('sort_charge', None)
        site = SpinSite(S, conserve, sort_charge)
        return site

    def init_terms(self, model_params):
        Jx = model_params.get('Jx', 1.)
        Jy = model_params.get('Jy', 1.)
        Jz = model_params.get('Jz', 1.)
        hx = model_params.get('hx', 0.)
        hy = model_params.get('hy', 0.)
        hz = model_params.get('hz', 0.)
        D = model_params.get('D', 0.)
        E = model_params.get('E', 0.)
        muJ = model_params.get('muJ', 0.)

        # (u is always 0 as we have only one site in the unit cell)
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-hx, u, 'Sx')
            self.add_onsite(-hy, u, 'Sy')
            self.add_onsite(-hz, u, 'Sz')
            self.add_onsite(D, u, 'Sz Sz')
            self.add_onsite(E * 0.5, u, 'Sp Sp')
            self.add_onsite(E * 0.5, u, 'Sm Sm')
        # Sp = Sx + i Sy, Sm = Sx - i Sy,  Sx = (Sp+Sm)/2, Sy = (Sp-Sm)/2i
        # Sx.Sx = 0.25 ( Sp.Sm + Sm.Sp + Sp.Sp + Sm.Sm )
        # Sy.Sy = 0.25 ( Sp.Sm + Sm.Sp - Sp.Sp - Sm.Sm )
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling((Jx + Jy) / 4., u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling((Jx - Jy) / 4., u1, 'Sp', u2, 'Sp', dx, plus_hc=True)
            self.add_coupling(Jz, u1, 'Sz', u2, 'Sz', dx)
            self.add_coupling(muJ * 0.5j, u1, 'Sm', u2, 'Sp', dx, plus_hc=True)
        # done


class SpinChain(SpinModel, NearestNeighborModel):
    """The :class:`SpinModel` on a Chain, suitable for TEBD.

    See the :class:`SpinModel` for the documentation of parameters.
    """
    default_lattice = Chain
    force_default_lattice = True

class AnisotropicSpinModel(SpinModel):
    r"""SpinModel on square lattice where two-spin couplings in the X and Y directions
    are allowed to be different. This allows for the creation of decoupled chains, and then
    slowly the coupling can be reintroduced.

    """
    default_lattice = Square
    force_default_lattice = True

    def init_terms(self, model_params):
        Jx = model_params.get('Jx', (1., 1.))   # X coupling, Y coupling
        Jy = model_params.get('Jy', (1., 1.))
        Jz = model_params.get('Jz', (1., 1.))
        hx = model_params.get('hx', 0.)
        hy = model_params.get('hy', 0.)
        hz = model_params.get('hz', 0.)
        D = model_params.get('D', 0.)
        E = model_params.get('E', 0.)
        muJ = model_params.get('muJ', (0., 0.))
        assert len(Jx) == len(Jy) == len(Jz) == len(muJ) == 2, "Need two parameters, one for each direction."

        # (u is always 0 as we have only one site in the unit cell)
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-hx, u, 'Sx')
            self.add_onsite(-hy, u, 'Sy')
            self.add_onsite(-hz, u, 'Sz')
            self.add_onsite(D, u, 'Sz Sz')
            self.add_onsite(E * 0.5, u, 'Sp Sp')
            self.add_onsite(E * 0.5, u, 'Sm Sm')
        # Sp = Sx + i Sy, Sm = Sx - i Sy,  Sx = (Sp+Sm)/2, Sy = (Sp-Sm)/2i
        # Sx.Sx = 0.25 ( Sp.Sm + Sm.Sp + Sp.Sp + Sm.Sm )
        # Sy.Sy = 0.25 ( Sp.Sm + Sm.Sp - Sp.Sp - Sm.Sm )
        #NN = [(0, 0, np.array([1, 0])), (0, 0, np.array([0, 1]))]
        for i, (u1, u2, dx) in enumerate(self.lat.pairs['nearest_neighbors']):
            print(i, u1, u2, dx)
            self.add_coupling((Jx[i] + Jy[i]) / 4., u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling((Jx[i] - Jy[i]) / 4., u1, 'Sp', u2, 'Sp', dx, plus_hc=True)
            self.add_coupling(Jz[i], u1, 'Sz', u2, 'Sz', dx)
            self.add_coupling(muJ[i] * 0.5j, u1, 'Sm', u2, 'Sp', dx, plus_hc=True)
        # done

class ExponentiallyDecayingXXZ(CouplingMPOModel):
    """
    f(r) * [X_i X_{i+r} + Y_i Y_{i+r} + Delta*Z_i Z_{i+r}]
    f(r) is approximated by a set of exponentials
    bond dimension is 3 * num_exponentials + 2

    One needs to approximate f(r) prior to initialization of the model.
    """

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', None)

        sort_charge = model_params.get('sort_charge', None)
        S = model_params.get('S', 0.5)

        site = SpinSite(S=S, conserve=conserve, sort_charge=sort_charge)
        return site

    def init_terms(self, model_params):
        lambdas = model_params['lambdas']
        prefactors = model_params['prefactors']
        delta = model_params.get('delta', 1)

        for lam, pre in zip(lambdas, prefactors):
            # Sp = Sx + i Sy, Sm = Sx - i Sy,  Sx = (Sp+Sm)/2, Sy = (Sp-Sm)/2i
            # Sp Sm + Sm Sp = 2 (Sx Sx + Sy Sy)
            self.add_exponentially_decaying_coupling(pre*2, lam, 'Sp', 'Sm', plus_hc=True)
            self.add_exponentially_decaying_coupling(pre*4*delta, lam, 'Sz', 'Sz')
        # done
