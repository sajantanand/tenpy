"""Nearest-neighbor spin-S models.

Uniform lattice of spin-S sites, coupled by nearest-neighbor interactions.
"""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np
from scipy.special import comb

from ..networks.site import SpinSite
from .model import CouplingMPOModel, NearestNeighborModel
from .lattice import Chain, Square
from ..tools.params import asConfig
from ..tools.fit import sum_of_exp
from ..linalg.charges import LegCharge, ChargeInfo

__all__ = ['SpinModel', 'SpinChain', 'DipolarSpinChain',
           'XXXChain', 
           'AnisotropicSpinModel', 'AnisotropicBarberPole', 
           'FiniteRangeSpinChain', 'VDWExponentiallyDecayingXXZ', 'ExponentiallyDecayingSpinModel']


class SpinModel(CouplingMPOModel):
    r"""Spin-S sites coupled by nearest neighbor interactions.

    The Hamiltonian reads:

    .. math ::
        H = \sum_{\langle i,j\rangle, i < j}
              (\mathtt{Jx} S^x_i S^x_j + \mathtt{Jy} S^y_i S^y_j + \mathtt{Jz} S^z_i S^z_j
            + \mathtt{muJ} i/2 (S^{-}_i S^{+}_j - S^{+}_i S^{-}_j))  \\
            - \sum_i (\mathtt{hx} S^x_i + \mathtt{hy} S^y_i + \mathtt{hz} S^z_i) \\
            + \sum_i (\mathtt{D} (S^z_i)^2 + \mathtt{E} ((S^x_i)^2 - (S^y_i)^2)) \\
            + \sum_j j F S^z_j + \sum_j j^2 G S^z_j


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
        sort_charge : bool
            Whether to sort by charges of physical legs. `True` by default.
        Jx, Jy, Jz, hx, hy, hz, muJ, D, E  : float | array
            Coupling as defined for the Hamiltonian above.
            Defaults to Heisenberg ``Jx=Jy=Jz=1.`` with other couplings 0.

    """

    def init_sites(self, model_params):
        S = model_params.get('S', 0.5, 'real')
        conserve = model_params.get('conserve', 'best', str)
        if conserve == 'best':
            # check how much we can conserve
            if not model_params.any_nonzero([('Jx', 'Jy'), 'hx', 'hy', 'E'], 'check Sz conservation'):
                conserve = 'Sz'
            elif not model_params.any_nonzero(['hx', 'hy'], 'check parity conservation'):
                conserve = 'parity'
            else:
                conserve = None
            self.logger.info('%s: set conserve to %s', self.name, conserve)
        sort_charge = model_params.get('sort_charge', True, bool)
        site = SpinSite(S, conserve, sort_charge)
        return site

    def init_terms(self, model_params):
        Jx = model_params.get('Jx', 1.0, 'real_or_array')
        Jy = model_params.get('Jy', 1.0, 'real_or_array')
        Jz = model_params.get('Jz', 1.0, 'real_or_array')
        hx = model_params.get('hx', 0.0, 'real_or_array')
        hy = model_params.get('hy', 0.0, 'real_or_array')
        hz = model_params.get('hz', 0.0, 'real_or_array')
        D = model_params.get('D', 0.0, 'real_or_array')
        E = model_params.get('E', 0.0, 'real_or_array')
        muJ = model_params.get('muJ', 0.0, 'real_or_array')
        F = model_params.get('F', 0., 'real_or_array')
        G = model_params.get('G', 0., 'real_or_array')
        L = model_params['L']

        # (u is always 0 as we have only one site in the unit cell)
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-hx, u, 'Sx')
            self.add_onsite(-hy, u, 'Sy')
            self.add_onsite(-hz, u, 'Sz')
            self.add_onsite(D, u, 'Sz Sz')
            self.add_onsite(E * 0.5, u, 'Sp Sp')
            self.add_onsite(E * 0.5, u, 'Sm Sm')
        for i in range(L):
            # Linearly tilted magnetic field
            # 0 of magnetic field is at middle site
            self.add_onsite_term((i - L//2) * F, i, 'Sz')

            # Quadratically tilted magnetic field
            # 0 of magnetic field is at middle site
            self.add_onsite_term((i - L//2)**2 * G, i, 'Sz')

        # Sp = Sx + i Sy, Sm = Sx - i Sy,  Sx = (Sp+Sm)/2, Sy = (Sp-Sm)/2i
        # Sx.Sx = 0.25 ( Sp.Sm + Sm.Sp + Sp.Sp + Sm.Sm )
        # Sy.Sy = 0.25 ( Sp.Sm + Sm.Sp - Sp.Sp - Sm.Sm )
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling((Jx + Jy) / 4.0, u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling((Jx - Jy) / 4.0, u1, 'Sp', u2, 'Sp', dx, plus_hc=True)
            self.add_coupling(Jz, u1, 'Sz', u2, 'Sz', dx)
            self.add_coupling(muJ * 0.5j, u1, 'Sm', u2, 'Sp', dx, plus_hc=True)
        # done

class SpinChain(SpinModel, NearestNeighborModel):
    """The :class:`SpinModel` on a Chain, suitable for TEBD.

    See the :class:`SpinModel` for the documentation of parameters.
    """

    default_lattice = Chain
    force_default_lattice = True

class fSimChain(SpinChain):
    """Model such that TEBD with dt=1.0 performs brick-wall evolution by fSim gates.

    fSim(theta, phi) = [[1,0,0,0],
                        [0,cos(theta),-isin(theta),0],
                        [0,-isin(theta),cos(theta),0],
                        [0,0,0,e^{-i phi}]]

    We want a Hamiltonian such that brick wall TEBD generates this evolution with dt=1.0.
    The effective Hamiltonian is given by H = alpha/2 (XX + YY) + beta/4 * (1-Z)*(1-Z),
    where alpha = logm(fSim)[1,2]*i, beta=logm(fSim)[3,3]*i. We will implement this with
    onsite fields for the single body Z. We drop the overall additive shift, which just
    gives rise to a global phase.
    """
    def init_sites(self, model_params):
        site = super().init_sites(model_params)
        assert model_params.get('S', 0.5, 'real') == 0.5
        # Id/2 - Sz
        # We assume `type(site) == SpinSite`, as this is what the parent class constructs.
        site.add_op('P1', np.array([[0., 0.], [0., 1.0]]), hc='P1', permute_dense=True)

        return site

    def init_terms(self, model_params):
        theta = model_params.get('theta', 1., 'real') * np.pi
        phi = model_params.get('phi', 1., 'real') * np.pi

        fSim = fSim = np.array([[1, 0, 0, 0],
                 [0,np.cos(theta), -1.j*np.sin(theta), 0],
                 [0, -1.j*np.sin(theta), np.cos(theta), 0],
                 [0, 0, 0, np.exp(-1.j * phi)]])
        self.fSim = fSim

        from scipy.linalg import expm, logm

        Hf = 1.j * logm(fSim)
        self.Hf = Hf
        assert np.isclose(np.linalg.norm(Hf.imag), 0.0)
        alpha = Hf[1,2]
        beta = Hf[3,3]
        
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            # Want Pauli, not Sz
            # alpha / 2 for boh XX and YY
            self.add_coupling(alpha, u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            #self.add_coupling(beta, u1, 'Sz', u2, 'Sz', dx)
            self.add_coupling(beta, u1, 'P1', u2, 'P1', dx)

class XXXChain(SpinModel):
    """The :class:`SpinModel` for the XXX model with conserved MPOs.

    We allow for both U(1) charge conserving and non charge conserving MPOs.

    If we pass a "center" parameter, we only place terms centered on that site. So k tells
    us the range of the conserved charge, and we place k MPO tensors centered on site
    center.

    Note: This does NOT inherit from NearestNeighborModel, so we cannot do TEBD.

    See the :class:`SpinModel` for the documentation of parameters.
    """
    
    default_lattice = Chain
    force_default_lattice = True
    
    def conserved_charge_MPO_U1(self, k, center=None, verbose=False, staggered=False):
        assert k >= 1, "Conserved quantities for k>=1."
        if not hasattr(self, 'MPOs_U1'):
            self.MPOs_U1 = {}
            self.Tensors_U1 = {}
        sites = self.lat.mps_sites()
        L = len(sites)

        if center != None:
            assert k > 1, "Can't make MPO of L=1."
            assert center - k // 2 >= 0
            assert center + (k // 2 - (k+1) % 2) <= L-1
        if k in self.MPOs_U1.keys() and center == None and not verbose and not staggered:
            return self.MPOs_U1[k]

        site = sites[0] # This contains the operators we use to build the sites.
        Sp = site.get_op('Sp')
        Sm = site.get_op('Sm')
        Sz = site.get_op('Sz')
        I = site.get_op('Id')
        Zero = None

        H_MPO = self.H_MPO

        D = 3*k - 1

        chinfo = I.chinfo
        triv_charge = chinfo == ChargeInfo()
        wR_charges = np.zeros((D, 1)) if not triv_charge else np.zeros((D, 0))
        wR_slices = np.arange(D+1)

        if k in self.Tensors_U1.keys() and not verbose:
            dW, dI = self.Tensors_U1[k]
        else:
            dI = np.empty((D, D), dtype=object)
            dW = np.empty((D, D), dtype=object)
            dW_str = np.empty((D, D), dtype=object)

            dI[0, 0] = dI[D-1, D-1] = I
            dW[0, 0] = dW[D-1, D-1] = I
            dW_str[0, 0] = dW_str[D-1, D-1] = 'I'

            if k == 1:
                dW[0, 1] = 2*Sz
                dW_str[0, 1] = 'Z'
            else:
                sigma = [np.sqrt(2)*Sm, np.sqrt(2)*Sp, 2*Sz]
                sigma_dag = [np.sqrt(2)*Sp, np.sqrt(2)*Sm, 2*Sz]
                sigma_str = ['sqrt(2).Sm', 'sqrt(2).Sp', '2.Sz']
                sigma_dag_str = ['sqrt(2).Sp', 'sqrt(2).Sm', '2.Sz']
                M = [[-2*Sz*1.j, Zero, np.sqrt(2)*Sm*1.j], [Zero, 2*Sz*1.j, -np.sqrt(2)*Sp*1.j], [np.sqrt(2)*Sp*1.j, -np.sqrt(2)*Sm*1.j, Zero]]
                M_str = [['-2i.Sz', 'None', 'i.sqrt(2).Sm'], ['None', '2i.Sz', '-i.sqrt(2).Sp'], ['i.sqrt(2).Sp', '-i.sqrt(2).Sm', '0']]

                def Catalan(n):
                    return comb(2*n, n) - comb(2*n, n-1)

                dW[0, 1:4] = sigma_dag
                dW[D-4:D-1, D-1] = sigma
                dW_str[0, 1:4] = sigma_dag_str
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
            self.Tensors_U1[k] = (dW, dI)

        if staggered:
            dWL = dW.copy()
            dWR = dW.copy()

            for i in range(1,D):
                dWR[0,i] = -1 * dWR[0,i] if dWR[0,i] is not None else None

        if not triv_charge:
            wR_charges[:,0] = [0] + [2,-2,0] * (k-1) + [0]
        wR_leg = LegCharge(chinfo, wR_slices, wR_charges, qconj=-1)

        IdL = [0] * (L + 1)
        IdR = [-1] * (L + 1)
        from ..networks.mpo import MPO
        if center == None:
            # We need to provide the leg charges since the there are dangling terms that do not connect to IdL or IdR.
            # We could get rid of these terms by cutting of rownd and columns of the left and right matrices, but since
            # the interactions are non-local (beyond NN), we'd have to change multiple tensors. It's easier to just
            # define the charges.
            if staggered:
                CC_MPO = MPO.from_grids(sites, [dWL]*(L//2) + [dWR]*(L-L//2), H_MPO.bc, IdL, IdR, legs = [wR_leg.conj()] * (L+1), max_range=k, explicit_plus_hc=False)
            else:
                CC_MPO = MPO.from_grids(sites, [dW]*(L), H_MPO.bc, IdL, IdR, legs = [wR_leg.conj()] * (L+1), max_range=k, explicit_plus_hc=False)
                self.MPOs_U1[k] = CC_MPO
        else:
            CC_MPO = MPO.from_grids(list(sites[:k]), [dW]*k, H_MPO.bc, IdL[:k+1], IdR[:k+1], max_range=k, explicit_plus_hc=False)
            I = I.add_trivial_leg(axis=0, label='wL', qconj=1).add_trivial_leg(axis=1, label='wR', qconj=-1)
            CC_MPO = MPO(sites, [I]*(center - k//2) + CC_MPO._W + [I]*(L-center + k//2 - k), bc=H_MPO.bc, IdL=IdL, IdR=IdR, max_range=k, explicit_plus_hc=False)

        return CC_MPO

    def conserved_charge_MPO_trivial(self, k, center=None, verbose=False, staggered=False):
        assert k >= 1, "Conserved quantities for k>=1."
        if not hasattr(self, 'MPOs'):
            self.MPOs = {}
            self.Tensors = {}
        sites = self.lat.mps_sites()
        L = len(sites)

        if center != None:
            assert k > 1, "Can't make MPO of L=1."
            assert center - k // 2 >= 0
            assert center + (k // 2 - (k+1) % 2) <= L-1
        if k in self.MPOs.keys() and center == None and not verbose and not staggered:
            return self.MPOs[k]

        site = sites[0] # This contains the operators we use to build the sites.
        X = 2*site.get_op('Sx')
        Y = 2*site.get_op('Sy')
        Z = 2*site.get_op('Sz')
        I = site.get_op('Id')
        Zero = None
    
        H_MPO = self.H_MPO

        D = 3*k - 1

        chinfo = I.chinfo
        assert chinfo == ChargeInfo()
        wR_charges = np.zeros((D, 0))
        wR_slices = np.arange(D+1)

        if k in self.Tensors.keys() and not verbose:
            dW, dI = self.Tensors[k]
        else:
            dI = np.empty((D, D), dtype=object)
            dW = np.empty((D, D), dtype=object)
            dW_str = np.empty((D, D), dtype=object)

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

        if staggered:
            dWL = dW.copy()
            dWR = dW.copy()

            for i in range(1, D):
                dWR[0,i] = -1 * dWR[0,i] if dWR[0,i] is not None else None

        wR_leg = LegCharge(chinfo, wR_slices, wR_charges, qconj=-1)

        IdL = [0] * (L + 1)
        IdR = [-1] * (L + 1)
        from ..networks.mpo import MPO
        if center == None:
            if staggered:
                CC_MPO = MPO.from_grids(sites, [dWL]*(L//2) + [dWR]*(L-L//2), H_MPO.bc, IdL, IdR, legs = [wR_leg.conj()] * (L+1), max_range=k, explicit_plus_hc=False)
            else:
                CC_MPO = MPO.from_grids(sites, [dW]*(L), H_MPO.bc, IdL, IdR, legs = [wR_leg.conj()] * (L+1), max_range=k, explicit_plus_hc=False)
                self.MPOs[k] = CC_MPO
        else:
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

class VDWExponentiallyDecayingXXZ(CouplingMPOModel):
    r"""Spin-S sites coupled by sum of exponentially-decaying interactions.

    The Hamiltonian reads:

    .. math ::
        H = \sum_{i < j}
              f(|i-j|) (\mathtt{Jx} S^x_i S^x_j + \mathtt{Jy} S^y_i S^y_j 
            + \mathtt{muJ} i/2 (S^{-}_i S^{+}_j - S^{+}_i S^{-}_j))  \\
            + \sum_{i < j}
              g(|i-j|) \mathtt{Jz} S^z_i S^z_j \\
            - \sum_i (\mathtt{hx} S^x_i + \mathtt{hy} S^y_i + \mathtt{hz} S^z_i) \\
            + \sum_i (\mathtt{D} (S^z_i)^2 + \mathtt{E} ((S^x_i)^2 - (S^y_i)^2)) \\
            + \sum_j j F S_z_j

    Here, :math:`i< j` denotes all pairs of sites and :math:`f(|i-j|)` is an interaction
    that depends on the distance between the sites. This interaction is written as a sum
    of exponentially decaying terms, so we must pass in the prefactor and decay constant
    for each exponential term.
    We allow for different interactions for the XY and ZZ terms.
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
        sort_charge : bool
            Whether to sort by charges of physical legs. `True` by default.
        Jx, Jy, Jz, hx, hy, hz, muJ, D, E  : float | array
            Coupling as defined for the Hamiltonian above.

    """

    def init_sites(self, model_params):
        S = model_params.get('S', 0.5, 'real')
        conserve = model_params.get('conserve', 'best', str)
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
        sort_charge = model_params.get('sort_charge', True, bool)
        site = SpinSite(S, conserve, sort_charge)
        return site

    def init_terms(self, model_params):
        Jx = model_params.get('Jx', 1., 'real_or_array')
        Jy = model_params.get('Jy', 1., 'real_or_array')
        Jz = model_params.get('Jz', 1., 'real_or_array')
        hx = model_params.get('hx', 0., 'real_or_array')
        hy = model_params.get('hy', 0., 'real_or_array')
        hz = model_params.get('hz', 0., 'real_or_array')
        D = model_params.get('D', 0., 'real_or_array')
        E = model_params.get('E', 0., 'real_or_array')
        muJ = model_params.get('muJ', 0., 'real_or_array')
        F = model_params.get('F', 0., 'real_or_array')
        G = model_params.get('G', 0., 'real_or_array')

        lambdas1 = model_params['lambdas1']
        prefactors1 = model_params['prefactors1']
        lambdas2 = model_params['lambdas2']
        prefactors2 = model_params['prefactors2']
        
        # We may want to normalize the Hamiltonian once it becomes long range enough.
        Kac_norm = model_params.get('Kac_norm', False)
        # If Delta is large, reduce the overall scale of the Hamiltonian.
        term_norm = model_params.get('term_norm', False)
        L = model_params['L']

        if term_norm:
            term_norm = np.sqrt(Jx**2 + Jy**2 + Jz**2)
            self.logger.info("term_norm: %f", term_norm)
        else:
            term_norm = 1

        if Kac_norm:
            # See Eq. 1 of https://arxiv.org/pdf/1909.01351
            # We use the approximate interation rather than the exact
            approximate_interation1 = sum_of_exp(lambdas1, prefactors1, np.arange(1, L//2))
            Kac_norm1 = np.sqrt(np.sum(approximate_interaction1**2)*2)
            self.logger.info("Kac_norm1: %f", Kac_norm1)
            
            approximate_interation2 = sum_of_exp(lambdas2, prefactors2, np.arange(1, L//2))
            Kac_norm2 = np.sqrt(np.sum(approximate_interaction2**2)*2)
            self.logger.info("Kac_norm2: %f", Kac_norm1)
        else:
            Kac_norm1 = Kac_norm2 = 1

        # (u is always 0 as we have only one site in the unit cell)
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-hx, u, 'Sx')
            self.add_onsite(-hy, u, 'Sy')
            self.add_onsite(-hz, u, 'Sz')
            self.add_onsite(D, u, 'Sz Sz')
            self.add_onsite(E * 0.5, u, 'Sp Sp')
            self.add_onsite(E * 0.5, u, 'Sm Sm')
        
        for i in range(L):
            # Linearly tilted magnetic field
            # 0 of magnetic field is at middle site
            self.add_onsite_term((i - L//2) * F, i, 'Sz')

            # Quadratically tilted magnetic field
            # 0 of magnetic field is at middle site
            self.add_onsite_term((i - L//2)**2 * G, i, 'Sz')

        # Sp = Sx + i Sy, Sm = Sx - i Sy,  Sx = (Sp+Sm)/2, Sy = (Sp-Sm)/2i
        # Sx.Sx = 0.25 ( Sp.Sm + Sm.Sp + Sp.Sp + Sm.Sm )
        # Sy.Sy = 0.25 ( Sp.Sm + Sm.Sp - Sp.Sp - Sm.Sm )
        for lam, pre in zip(lambdas1, prefactors1):
            # Only include terms if the prefactors are non-zero; else we just add
            # unnecessary states (bond dimension) to the FSM (MPO).
            # TODO - fix MPO generation to not do anything if strength == 0.

            # Sp = Sx + i Sy, Sm = Sx - i Sy,  Sx = (Sp+Sm)/2, Sy = (Sp-Sm)/2i
            # Sp Sm + Sm Sp = 2 (Sx Sx + Sy Sy)
            if Jx + Jy != 0.0:
                self.add_exponentially_decaying_coupling(pre * (Jx + Jy) / 4. / Kac_norm1 / term_norm, lam, 'Sp', 'Sm', plus_hc=True)
            if Jx - Jy != 0.0:
                self.add_exponentially_decaying_coupling(pre * (Jx - Jy) / 4. / Kac_norm1 / term_norm, lam, 'Sp', 'Sp', plus_hc=True)
            if muJ != 0.0:
                self.add_exponentially_decaying_coupling(pre * muJ * 0.5j / Kac_norm1, lam, 'Sm', 'Sp', plus_hc=True)

        for lam, pre in zip(lambdas2, prefactors2):
            # Only include terms if the prefactors are non-zero; else we just add
            # unnecessary states (bond dimension) to the FSM (MPO).
            # TODO - fix MPO generation to not do anything if strength == 0.

            # Sp = Sx + i Sy, Sm = Sx - i Sy,  Sx = (Sp+Sm)/2, Sy = (Sp-Sm)/2i
            # Sp Sm + Sm Sp = 2 (Sx Sx + Sy Sy)
            if Jz != 0:
                self.add_exponentially_decaying_coupling(pre * Jz / Kac_norm2 / term_norm, lam, 'Sz', 'Sz')

class ExponentiallyDecayingSpinModel(CouplingMPOModel):
    r"""Spin-S sites coupled by sum of exponentially-decaying interactions.

    The Hamiltonian reads:

    .. math ::
        H = \sum_{i < j}
              f(|i-j|) (\mathtt{Jx} S^x_i S^x_j + \mathtt{Jy} S^y_i S^y_j + \mathtt{Jz} S^z_i S^z_j
            + \mathtt{muJ} i/2 (S^{-}_i S^{+}_j - S^{+}_i S^{-}_j))  \\
            - \sum_i (\mathtt{hx} S^x_i + \mathtt{hy} S^y_i + \mathtt{hz} S^z_i) \\
            + \sum_i (\mathtt{D} (S^z_i)^2 + \mathtt{E} ((S^x_i)^2 - (S^y_i)^2)) \\
            + \sum_j j F S_z_j

    Here, :math:`i< j` denotes all pairs of sites and :math:`f(|i-j|)` is an interaction
    that depends on the distance between the sites. This interaction is written as a sum
    of exponentially decaying terms, so we must pass in the prefactor and decay constant
    for each exponential term.
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
        sort_charge : bool
            Whether to sort by charges of physical legs. `True` by default.
        Jx, Jy, Jz, hx, hy, hz, muJ, D, E  : float | array
            Coupling as defined for the Hamiltonian above.

    """
    def init_sites(self, model_params):
        S = model_params.get('S', 0.5, 'real')
        conserve = model_params.get('conserve', 'best', str)
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
        sort_charge = model_params.get('sort_charge', True, bool)
        site = SpinSite(S, conserve, sort_charge)
        return site

    def init_terms(self, model_params):
        Jx = model_params.get('Jx', 1., 'real_or_array')
        Jy = model_params.get('Jy', 1., 'real_or_array')
        Jz = model_params.get('Jz', 1., 'real_or_array')
        hx = model_params.get('hx', 0., 'real_or_array')
        hy = model_params.get('hy', 0., 'real_or_array')
        hz = model_params.get('hz', 0., 'real_or_array')
        D = model_params.get('D', 0., 'real_or_array')
        E = model_params.get('E', 0., 'real_or_array')
        muJ = model_params.get('muJ', 0., 'real_or_array')
        F = model_params.get('F', 0., 'real_or_array')
        G = model_params.get('G', 0., 'real_or_array')

        lambdas = model_params['lambdas']
        prefactors = model_params['prefactors']
        
        # We may want to normalize the Hamiltonian once it becomes long range enough.
        Kac_norm = model_params.get('Kac_norm', False)
        # If Delta is large, reduce the overall scale of the Hamiltonian.
        term_norm = model_params.get('term_norm', False)
        L = model_params['L']

        if term_norm:
            term_norm = np.sqrt(Jx**2 + Jy**2 + Jz**2**2)
            self.logger.info("term_norm: %f", term_norm)
        else:
            term_norm = 1

        if Kac_norm:
            # See Eq. 1 of https://arxiv.org/pdf/1909.01351
            # We use the approximate interation rather than the exact
            approximate_interation = sum_of_exp(lambdas, prefactors, np.arange(1, L//2))
            Kac_norm = np.sqrt(np.sum(approximate_interaction**2)*2)
            self.logger.info("kac_norm: %f", Kac_norm)
        else:
            Kac_norm = 1

        # (u is always 0 as we have only one site in the unit cell)
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-hx, u, 'Sx')
            self.add_onsite(-hy, u, 'Sy')
            self.add_onsite(-hz, u, 'Sz')
            self.add_onsite(D, u, 'Sz Sz')
            self.add_onsite(E * 0.5, u, 'Sp Sp')
            self.add_onsite(E * 0.5, u, 'Sm Sm')
        
        for i in range(L):
            # Linearly tilted magnetic field
            # 0 of magnetic field is at middle site
            self.add_onsite_term((i - L//2) * F, i, 'Sz')

            # Quadratically tilted magnetic field
            # 0 of magnetic field is at middle site
            self.add_onsite_term((i - L//2)**2 * G, i, 'Sz')

        # Sp = Sx + i Sy, Sm = Sx - i Sy,  Sx = (Sp+Sm)/2, Sy = (Sp-Sm)/2i
        # Sx.Sx = 0.25 ( Sp.Sm + Sm.Sp + Sp.Sp + Sm.Sm )
        # Sy.Sy = 0.25 ( Sp.Sm + Sm.Sp - Sp.Sp - Sm.Sm )
        for lam, pre in zip(lambdas, prefactors):
            # Only include terms if the prefactors are non-zero; else we just add
            # unnecessary states (bond dimension) to the FSM (MPO).
            # TODO - fix MPO generation to not do anything if strength == 0.

            # Sp = Sx + i Sy, Sm = Sx - i Sy,  Sx = (Sp+Sm)/2, Sy = (Sp-Sm)/2i
            # Sp Sm + Sm Sp = 2 (Sx Sx + Sy Sy)
            if Jx + Jy != 0.0:
                self.add_exponentially_decaying_coupling(pre * (Jx + Jy) / 4. / Kac_norm / term_norm, lam, 'Sp', 'Sm', plus_hc=True)
            if Jx - Jy != 0.0:
                self.add_exponentially_decaying_coupling(pre * (Jx - Jy) / 4. / Kac_norm / term_norm, lam, 'Sp', 'Sp', plus_hc=True)
            if muJ != 0.0:
                self.add_exponentially_decaying_coupling(pre * muJ * 0.5j / Kac_norm, lam, 'Sm', 'Sp', plus_hc=True)
            if Jz != 0:
                self.add_exponentially_decaying_coupling(pre * Jz / Kac_norm / term_norm, lam, 'Sz', 'Sz')

class FiniteRangeSpinChain(SpinChain):
    r"""Spin-S chain coupled by finite-range interactions.

    Suppose we have a long-range f(r) interaction that we truncate after some r. Here we explicitly
    add all 1D non-local connections to build this model, where the interactions strengths are given
    by `interaction`.

    """

    def init_terms(self, model_params):
        Jx = model_params.get('Jx', 1., 'real_or_array')
        Jy = model_params.get('Jy', 1., 'real_or_array')
        Jz = model_params.get('Jz', 1., 'real_or_array')
        hx = model_params.get('hx', 0., 'real_or_array')
        hy = model_params.get('hy', 0., 'real_or_array')
        hz = model_params.get('hz', 0., 'real_or_array')
        D = model_params.get('D', 0., 'real_or_array')
        E = model_params.get('E', 0., 'real_or_array')
        muJ = model_params.get('muJ', 0., 'real_or_array')
        interaction = model_params['interaction']

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
        u1 = u2 = 0
        for dx, coup in enumerate(interaction):
            # Unit cell is always 0 for chain
            dx += 1
            self.add_coupling((Jx + Jy) / 4. * coup, 0, 'Sp', 0, 'Sm', dx, plus_hc=True)
            self.add_coupling((Jx - Jy) / 4. * coup, u1, 'Sp', u2, 'Sp', dx, plus_hc=True)
            self.add_coupling(Jz * coup, u1, 'Sz', u2, 'Sz', dx)
            self.add_coupling(muJ * 0.5j * coup, u1, 'Sm', u2, 'Sp', dx, plus_hc=True)
        # done

class DipolarSpinChain(CouplingMPOModel):
    r"""Dipole conserving H3-H4 spin-S chain.

    The Hamiltonian reads:

    .. math ::
        H = - \mathtt{J3} \sum_{i} (S^+_i (S^-_{i + 1})^2 S^+_{i + 2} + \mathrm{h.c.})
            - \mathtt{J4} \sum_{i} (S^+_i S^-_{i + 1} S^-_{i + 2} S^+_{i + 2} + \mathrm{h.c.})

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`DipolarSpinChain` below.

    Options
    -------
    .. cfg:config :: DipolarSpinChain
        :include: CouplingMPOModel

        S : {0.5, 1, 1.5, 2, ...}
            The 2S+1 local states range from m = -S, -S+1, ... +S.
            Defaults to ``S=1``.
        conserve : 'best' | 'dipole' | 'Sz' | 'parity' | None
            What should be conserved. See :class:`~tenpy.networks.site.SpinSite`.
            Note that dipole conservation necessarily includes Sz conservation.
            For ``'best'``, we preserve ``'dipole'``.
        sort_charge : bool | None
            Whether to sort by charges of physical legs.
            See change comment in :class:`~tenpy.networks.site.Site`.
        J3, J4 : float | array
            Coupling as defined for the Hamiltonian above.

    """

    def init_lattice(self, model_params):
        """Initialize a 1D lattice"""
        L = model_params.get('L', 64)
        S = model_params.get('S', 1)
        conserve = model_params.get('conserve', 'best')
        if conserve == 'best':
            conserve = 'dipole'
            self.logger.info('%s: set conserve to %s', self.name, conserve)
        bc_MPS = model_params.get('bc_MPS', 'finite')
        bc = 'periodic' if bc_MPS in ['infinite', 'segment'] else 'open'
        bc = model_params.get('bc', bc)
        sort_charge = model_params.get('sort_charge', None)
        site = SpinSite(S=S, conserve=conserve, sort_charge=sort_charge)
        lattice = Chain(L, site, bc=bc, bc_MPS=bc_MPS)
        return lattice

    def init_terms(self, model_params):
        """Add the onsite and coupling terms to the model"""
        J3 = model_params.get('J3', 1)
        J4 = model_params.get('J4', 0)
        self.add_multi_coupling(-J3, [('Sp', 0, 0), ('Sm', 1, 0), ('Sm', 1, 0), ('Sp', 2, 0)], plus_hc=True)
        self.add_multi_coupling(-J4, [('Sp', 0, 0), ('Sm', 1, 0), ('Sm', 2, 0), ('Sp', 3, 0)], plus_hc=True)
