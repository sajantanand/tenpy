"""Nearest-neighbor spin-S models.

Uniform lattice of spin-S sites, coupled by nearest-neighbor interactions.
"""
# Copyright 2018-2023 TeNPy Developers, GNU GPLv3

import numpy as np
from scipy.special import comb

from ..networks.site import SpinSite
from .model import CouplingMPOModel, NearestNeighborModel
from .lattice import Chain, Square
from ..tools.params import asConfig

__all__ = ['SpinModel', 'SpinChain', 'XXXChain', 'ExponentiallyDecayingXXZ', 'AnisotropicSpinModel', 'AnisotropicBarberPole']


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

class XXXChain(SpinChain):
    """The :class:`SpinChain` for the XXX model with conserved MPOs

    See the :class:`SpinModel` for the documentation of parameters.
    """

    def conserved_charge_MPO(self, k, center=None, verbose=False):
        assert k >= 1, "Conserved quantities for k>1."
        if not hasattr(self, 'MPOs'):
            self.MPOs = {}
            self.Tensors = {}
        sites = self.lat.mps_sites()
        L = len(sites)

        if center != None:
            assert k > 1, "Can't make MPO of L=1."
            assert center - k // 2 >= 0
            assert center + (k // 2 - (k+1) % 2) <= L-1
        if k in self.MPOs.keys() and center == None:
            return self.MPOs[k]

        site = sites[0] # This contains the operators we use to build the sites.
        X = 2*site.get_op('Sx')
        Y = 2*site.get_op('Sy')
        Z = 2*site.get_op('Sz')
        Zero = 0 * Z
        I = site.get_op('Id')
        H_MPO = self.H_MPO

        if k in self.Tensors.keys():
            dW, dI = self.Tensors[k]
        else:
            D = 3*k - 1
            dI = np.empty((D, D), dtype=object)
            dW = np.empty((D, D), dtype=object)
            dW_str = np.empty((D, D), dtype=object)
            for i in range(D):
                for j in range(D):
                    dI[i,j] = Zero
                    dW[i,j] = Zero

            dI[0, 0] = dI[D-1, D-1] = I
            dW[0, 0] = dW[D-1, D-1] = I
            dW_str[0, 0] = dW_str[D-1, D-1] = 'I'

            if k == 1:
                dW[0, 1] = Z
                dW_str[0, 1] = 'Z'
            else:
                sigma = [X, Y, Z]
                sigma_str = ['X', 'Y', 'Z']
                M = [[Zero, -Z, Y], [Z, Zero, -X], [-Y, X, Zero]]
                M_str = [['0', '-Z', 'Y'], ['Z', '0', '-X'], ['-Y', 'X', '0']]

                def Catalan(n):
                    return comb(2*n, n) - comb(2*n, n-1)

                dW[0, 1:4] = sigma
                dW[D-4:D-1, D-1] = sigma
                dW_str[0, 1:4] = sigma_str
                dW_str[D-4:D-1, D-1] = sigma_str

                assert (D - 1 - 3 - 1) % 3 == 0
                for i in range(int((D - 1 - 3 - 1)//3)):
                    dW[(1+3*i):(1+3*(i+1)), (4+3*i):(4+3*(i+1))] = M
                    dW_str[(1+3*i):(1+3*(i+1)), (4+3*i):(4+3*(i+1))] = M_str
                for i in range(1, k-2):
                    for j in range(1, k-1):
                        if j - i > 0 and (j - i) % 2:
                            n = (j - i) // 2
                            Cn = Catalan(n)
                            Cn_str = 'C' + str(n)
                            I3 = [[Cn*I, Zero, Zero], [Zero, Cn*I, Zero], [Zero, Zero, Cn*I]]
                            I3_str = [[Cn_str, '0', '0'], ['0', Cn_str, '0'], ['0', '0', Cn_str]]
                            dW[(1+3*(i-1)):(1+3*i), (4+3*(j-1)):(4+3*j)] = I3
                            dW_str[(1+3*(i-1)):(1+3*i), (4+3*(j-1)):(4+3*j)] = I3_str

            if verbose:
                print(dW_str)
            self.Tensors[k] = (dW, dI)

        IdL = [0] * (L + 1)
        IdR = [-1] * (L + 1)
        from ..networks.mpo import MPO
        if center == None:
            CC_MPO = MPO.from_grids(sites, [dW]*L, H_MPO.bc, IdL, IdR, max_range=k, explicit_plus_hc=False)
            self.MPOs[k] = CC_MPO
        else:
            #CC_MPO = MPO.from_grids(sites, [dI]*(center - k//2) + [dW]*k + [dI]*(L-center + k//2 - k), H_MPO.bc, IdL, IdR, max_range=k, explicit_plus_hc=False)
            CC_MPO = MPO.from_grids(list(sites[:k]), [dW]*k, H_MPO.bc, IdL[:k+1], IdR[:k+1], max_range=k, explicit_plus_hc=False)
            I = I.add_trivial_leg(axis=0, label='wL', qconj=1).add_trivial_leg(axis=1, label='wR', qconj=-1)
            CC_MPO = MPO(sites, [I]*(center - k//2) + CC_MPO._W + [I]*(L-center + k//2 - k), bc=H_MPO.bc, IdL=IdL, IdR=IdR, max_range=k, explicit_plus_hc=False)

        return CC_MPO


class AnisotropicSpinModel(SpinModel):
    r"""SpinModel on 2D (or more) lattice where two-spin couplings in the Bravais lattice directions
    are allowed to be different. This allows for the creation of decoupled chains, and then
    slowly the coupling can be reintroduced.

    Two-body coupling parameters are now tuples, one entry for eavh Bravais lattice vector.
    """

    def init_terms(self, model_params):
        BVs = len(self.lat.pairs['nearest_neighbors'])
        Jx = model_params.get('Jx', (1.,)*BVs)   # X coupling, Y coupling
        Jy = model_params.get('Jy', (1.,)*BVs)
        Jz = model_params.get('Jz', (1.,)*BVs)
        hx = model_params.get('hx', 0.)
        hy = model_params.get('hy', 0.)
        hz = model_params.get('hz', 0.)
        D = model_params.get('D', 0.)
        E = model_params.get('E', 0.)
        muJ = model_params.get('muJ', (0.,)*BVs)
        assert len(Jx) == len(Jy) == len(Jz) == len(muJ) == BVs, "Need one parameter for each direction."

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
            self.add_coupling((Jx[i] + Jy[i]) / 4., u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling((Jx[i] - Jy[i]) / 4., u1, 'Sp', u2, 'Sp', dx, plus_hc=True)
            self.add_coupling(Jz[i], u1, 'Sz', u2, 'Sz', dx)
            self.add_coupling(muJ[i] * 0.5j, u1, 'Sm', u2, 'Sp', dx, plus_hc=True)
        # done

class AnisotropicBarberPole(SpinModel):
    r"""Same as anisotropic spin model but with next_nearest_neighbor couplings. This allows for a "barber pole"-like spreading of spin-spin correlator.
    """
    # default_lattice = Square
    # force_default_lattice = True
    
    def init_terms(self, model_params):
        BVs_NN = len(self.lat.pairs['nearest_neighbors'])
        BVs_NNN = len(self.lat.pairs['next_nearest_neighbors'])
        Jx = model_params.get('Jx', (1.,)*BVs_NN)
        Jy = model_params.get('Jy', (1.,)*BVs_NN)
        Jz = model_params.get('Jz', (1.,)*BVs_NN)
        Jxx = model_params.get('Jxx', (1.,)*BVs_NNN)
        Jyy = model_params.get('Jyy', (1.,)*BVs_NNN)
        Jzz = model_params.get('Jzz', (1.,)*BVs_NNN)
        hx = model_params.get('hx', 0.)
        hy = model_params.get('hy', 0.)
        hz = model_params.get('hz', 0.)
        D = model_params.get('D', 0.)
        E = model_params.get('E', 0.)
        muJ = model_params.get('muJ', (0.,)*BVs_NN)
        assert len(Jx) == len(Jy) == len(Jz) == len(muJ) == BVs_NN, "Need one parameter for each direction."
        assert len(Jxx) == len(Jyy) == len(Jzz) == BVs_NNN, "Need one parameter for each direction."

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
            self.add_coupling((Jx[i] + Jy[i]) / 4., u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling((Jx[i] - Jy[i]) / 4., u1, 'Sp', u2, 'Sp', dx, plus_hc=True)
            self.add_coupling(Jz[i], u1, 'Sz', u2, 'Sz', dx)
            self.add_coupling(muJ[i] * 0.5j, u1, 'Sm', u2, 'Sp', dx, plus_hc=True)
        # done
        
        #nNN = [(0, 0, np.array([1, 1])), (0, 0, np.array([1, -1]))]
        for i, (u1, u2, dx) in enumerate(self.lat.pairs['next_nearest_neighbors']):
            self.add_coupling((Jxx[i] + Jyy[i]) / 4., u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling((Jxx[i] - Jyy[i]) / 4., u1, 'Sp', u2, 'Sp', dx, plus_hc=True)
            self.add_coupling(Jzz[i], u1, 'Sz', u2, 'Sz', dx)
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

    def conserved_charge_MPO(self, k, center=None, verbose=False):
        print("conserved charge")
        assert k == 1, "Only conserve total Sz."
        if not hasattr(self, 'MPOs'):
            self.MPOs = {}
            self.Tensors = {}
        sites = self.lat.mps_sites()
        L = len(sites)

        if center != None:
            assert k > 1, "Can't make MPO of L=1."
            assert center - k // 2 >= 0
            assert center + (k // 2 - (k+1) % 2) <= L-1
        if k in self.MPOs.keys() and center == None:
            return self.MPOs[k]

        site = sites[0] # This contains the operators we use to build the sites.
        Z = 2*site.get_op('Sz')
        Zero = 0 * Z
        I = site.get_op('Id')
        H_MPO = self.H_MPO

        if k in self.Tensors.keys():
            dW, dI = self.Tensors[k]
        else:
            D = 3*k - 1
            dI = np.empty((D, D), dtype=object)
            dW = np.empty((D, D), dtype=object)
            dW_str = np.empty((D, D), dtype=object)
            for i in range(D):
                for j in range(D):
                    dI[i,j] = Zero
                    dW[i,j] = Zero

            dI[0, 0] = dI[D-1, D-1] = I
            dW[0, 0] = dW[D-1, D-1] = I
            dW_str[0, 0] = dW_str[D-1, D-1] = 'I'

            if k == 1:
                dW[0, 1] = Z
                dW_str[0, 1] = 'Z'
            else:
                raise NotImplementedError('Do you know the form of the CCs?')
            if verbose:
                print(dW_str)
            self.Tensors[k] = (dW, dI)

        IdL = [0] * (L + 1)
        IdR = [-1] * (L + 1)
        from ..networks.mpo import MPO
        if center == None:
            CC_MPO = MPO.from_grids(sites, [dW]*L, H_MPO.bc, IdL, IdR, max_range=k, explicit_plus_hc=False)
            self.MPOs[k] = CC_MPO
        else:
            #CC_MPO = MPO.from_grids(sites, [dI]*(center - k//2) + [dW]*k + [dI]*(L-center + k//2 - k), H_MPO.bc, IdL, IdR, max_range=k, explicit_plus_hc=False)
            CC_MPO = MPO.from_grids(list(sites[:k]), [dW]*k, H_MPO.bc, IdL[:k+1], IdR[:k+1], max_range=k, explicit_plus_hc=False)
            I = I.add_trivial_leg(axis=0, label='wL', qconj=1).add_trivial_leg(axis=1, label='wR', qconj=-1)
            CC_MPO = MPO(sites, [I]*(center - k//2) + CC_MPO._W + [I]*(L-center + k//2 - k), bc=H_MPO.bc, IdL=IdL, IdR=IdR, max_range=k, explicit_plus_hc=False)

        return CC_MPO

