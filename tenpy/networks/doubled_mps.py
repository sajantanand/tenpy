# Copyright 2018-2023 TeNPy Developers, GNU GPLv3

import copy
import numpy as np
import itertools

from .mps import MPS, MPSEnvironment
from .site import DoubledSite
from ..linalg import np_conserved as npc
from ..tools.math import entropy
from ..tools.misc import lexsort

__all__ = ['DoubledMPS']

# Use MPSEnvironment methods first where there is overlap!
class DoubledMPS(MPS):
    r"""SAJANT - Write documentation later
    """

    # we use q to mean p*, the bra leg of the density matrix or operator
    # `MPS.get_B` & co work, thanks to using labels. `B` just have the additional `q` labels.
    _p_label = ['p', 'q']  # this adjustment makes `get_theta` & friends work
    _B_labels = ['vL', 'p', 'q', 'vR']
    
    # SAJANT - Do we need purification test_sanity?
    #def test_sanity(self):
    
    # For doubled MPS initialization (either density matrix or operator), we will only
    # use from_Bflat, where we pass in the tensors on each site directly. The user
    # will need to define the appropriate matrices.
    
    @classmethod
    def from_lat_product_state(cls, lat, p_state, allow_incommensurate=False, **kwargs):
        raise NotImplementedError()
    
    
    @classmethod
    def from_product_state(cls,
                           sites,
                           p_state,
                           bc='finite',
                           dtype=np.float64,
                           permute=True,
                           form='B',
                           chargeL=None):
        raise NotImplementedError()
        
    @classmethod
    def from_Bflat(cls,
                   sites,
                   Bflat,
                   SVs=None,
                   bc='finite',
                   dtype=None,
                   permute=True,
                   form='B',
                   legL=None):
        """Construct a matrix product state from a set of numpy arrays `Bflat` and singular vals.

        Parameters
        ----------
        sites : list of :class:`~tenpy.networks.site.Site`
            The sites defining the local Hilbert space.
        Bflat : iterable of numpy ndarrays
            The matrix defining the MPS on each site, with legs ``'p', 'q', vL', 'vR'``
            (physical ket, physical bra, virtual left/right).
        SVs : list of 1D array | ``None``
            The singular values on *each* bond. Should always have length `L+1`.
            By default (``None``), set all singular values to the same value.
            Entries out of :attr:`nontrivial_bonds` are ignored.
        bc : {'infinite', 'finite', 'segment'}
            MPS boundary conditions. See docstring of :class:`MPS`.
        dtype : type or string
            The data type of the array entries. Defaults to the common dtype of `Bflat`.
        permute : bool
            The :class:`~tenpy.networks.Site` might permute the local basis states if charge
            conservation gets enabled.
            If `permute` is True (default), we permute the given `Bflat` locally according to
            each site's :attr:`~tenpy.networks.Site.perm`.
            The `p_state` argument should then always be given as if `conserve=None` in the Site.
        form : (list of) {``'B' | 'A' | 'C' | 'G' | None`` | tuple(float, float)}
            Defines the canonical form of `Bflat`. See module doc-string.
            A single choice holds for all of the entries.
        leg_L : LegCharge | ``None``
            Leg charges at bond 0, which are purely conventional.
            If ``None``, use trivial charges.

        Returns
        -------
        dmps : :class:`DoubledMPS`
            An doubled MPS with the matrices `Bflat` converted to npc arrays.
        """
        sites = list(sites)
        L = len(sites)
        Bflat = list(Bflat)
        if len(Bflat) != L:
            raise ValueError("Length of Bflat does not match number of sites.")
        ci = sites[0].leg.chinfo
        if legL is None:
            legL = npc.LegCharge.from_qflat(ci, [ci.make_valid(None)] * Bflat[0].shape[2])
            legL = legL.bunch()[1]
        if SVs is None:
            # Modified to account for the two physical legs
            SVs = [np.ones(B.shape[2]) / np.sqrt(B.shape[2]) for B in Bflat]
            SVs.append(np.ones(Bflat[-1].shape[3]) / np.sqrt(Bflat[-1].shape[3]))
        Bs = []
        if dtype is None:
            dtype = np.dtype(np.common_type(*Bflat))
        for i, site in enumerate(sites):
            B = np.array(Bflat[i], dtype)
            if permute:
                B = B[site.perm, :, :]
            # calculate the LegCharge of the right leg
            # Modified to account for the two physical legs
            legs = [site.leg, site.leg.conj(), legL, None]  # other legs are known
            legs = npc.detect_legcharge(B, ci, legs, None, qconj=-1)
            B = npc.Array.from_ndarray(B, legs, dtype)
            B.iset_leg_labels(['p', 'q', 'vL', 'vR']) # Modified to have multiple physical legs
            Bs.append(B)
            legL = legs[-1].conj()  # prepare for next `i`
        if bc == 'infinite':
            # for an iMPS, the last leg has to match the first one.
            # so we need to gauge `qtotal` of the last `B` such that the right leg matches.
            chdiff = Bs[-1].get_leg('vR').charges[0] - Bs[0].get_leg('vL').charges[0]
            Bs[-1] = Bs[-1].gauge_total_charge('vR', ci.make_valid(chdiff))
        return cls(sites, Bs, SVs, form=form, bc=bc)

    @classmethod
    def from_full(cls,
                  sites,
                  psi,
                  form=None,
                  cutoff=1.e-16,
                  normalize=True,
                  bc='finite',
                  outer_S=None):
        raise NotImplementedError()
        # No need for this, as psi is a dense vector with 2*L legs; this is too much for doubled states.
    
    @classmethod
    def from_singlets(cls,
                      site,
                      L,
                      pairs,
                      up='up',
                      down='down',
                      lonely=[],
                      lonely_state='up',
                      bc='finite'):
        raise NotImplementedError()
    
    def to_regular_MPS(self):
        """
        Convert a doubled MPS to a regular MPS by combining together the 'p' and 'q' legs
        """
        # Build new site of squared dimension
        doubled_sites = [DoubledSite(self.sites[0].dim)] * self.L
        new_Bs = [B.combine_legs(('p', 'q')).replace_label('(p.q)', 'p') for B in self._B]
        pipes = [B.get_leg('p') for B in new_Bs]
        new_MPS = MPS(doubled_sites, new_Bs, self._S, bc='finite', form='B', norm=self.norm)
        return new_MPS, pipes
    
    def from_regular_MPS(self, reg_MPS, pipes):
        """
        Convert a regular MPS back into a doubled MPS. We split the 'p' leg into 'p' and 'q'.
        """
        for B, pipe in zip(reg_MPS._B, pipes):
            B.itranspose(['vL', 'p', 'vR'])
            B.legs[1] = pipe
        self._B = [B.replace_label('p', '(p.q)').split_legs() for B in reg_MPS._B]
        self._S = reg_MPS._S
        self.norm = reg_MPS.norm
        self.form = reg_MPS.form
        self.test_sanity()
    
    # If we want this, need to take care of additional physical legs.
    # This function is needed for SVD compression of (infinite) doubled MPS.
    def set_svd_theta(self, i, theta, trunc_par=None, update_norm=False):
        """SVD a two-site wave function `theta` and save it in `self`.

        Parameters
        ----------
        i : int
            `theta` is the wave function on sites `i`, `i` + 1.
        theta : :class:`~tenpy.linalg.np_conserved.Array`
            The two-site wave function with labels combined into ``"(vL.p0.q0)", "(p1.q1.vR)"``,
            ready for svd.
        trunc_par : None | dict
            Parameters for truncation, see :cfg:config:`truncation`.
            If ``None``, no truncation is done.
        update_norm : bool
            If ``True``, multiply the norm of `theta` into :attr:`norm`.
        """
        i0 = self._to_valid_index(i)
        i1 = self._to_valid_index(i0 + 1)
        self.dtype = np.promote_types(self.dtype, theta.dtype)
        qtotal_LR = [self._B[i0].qtotal, None]
        if trunc_par is None:
            U, S, VH = npc.svd(theta, qtotal_LR=qtotal_LR, inner_labels=['vR', 'vL'])
            renorm = np.linalg.norm(S)
            S /= renorm
            err = None
            if update_norm:
                self.norm *= renorm
        else:
            U, S, VH, err, renorm = svd_theta(theta, trunc_par, qtotal_LR)
            if update_norm:
                self.norm *= renorm
        U = U.split_legs().ireplace_labels(['p0', 'q0'], ['p', 'q'])
        VH = VH.split_legs().ireplace_labels(['p1', 'q1'], ['p', 'q'])
        self._B[i0] = U.itranspose(self._B_labels)
        self.form[i0] = self._valid_forms['A']
        self._B[i1] = VH.itranspose(self._B_labels)
        self.form[i1] = self._valid_forms['B']
        self.set_SR(i, S)
        return err
    
    #def get_theta(self, i, n=2, cutoff=1.e-16, formL=1., formR=1.):
        # This function is returns the theta matrix for $n$ sites with $2n + 2$ indices, a 'p'
        # and 'q' for each tensor and a 'vL' and 'vR' at the ends.
    
    r"""
    Entanglement entropy for density matrices (or operators) is weird, since we don't need two copies
    of the dMPS to get the density matrix. Instead, a single copy of the dMPS is the density matrix itself
    (as the name suggests).
    
    The below functions would treat the dMPS as a pure state and calculate the traditional entanglement entropies.
    So given some region A, what we are calculating is something like $-Tr((\rho^2)_A \ln (\rho^2)_A)$, 
    where $\rho^2_A$ is found by taking two copies of the dMPS (easiest to think of it as a vectorized
    MPS with local Hilbert space d**2 rather than having two physical legs of dimension d) and tracing 
    over the complement of A (A-bar). So then we are left with a density matrix of region $A$ that has
    4 |A| legs (2 bras, 2 kets). We then project this into the bond space using the isometry tensors of
    \rho, which gets us back to an RDM just defined by the singular values on the bond (assuming that we
    are interested in a bipartition).
    
    We turn off any function that treats the doubled MPS as an MPS with combined physical legs. If you
    want this functionality, just convert to a regular MPS and then call the function. We reimplement
    certain functions to properly treat the density matrix as a density matrix, i.e. only ever use one
    copy of the state.
    """
    
    def entanglement_entropy(self, n=1, bonds=None, for_matrix_S=False):
        raise NotImplementedError()
    
    def entanglement_entropy_segment2(self, segment, n=1):
        raise NotImplementedError()
        
    def entanglement_spectrum(self, by_charge=False):
        raise NotImplementedError()
        
    def trace(self):
        # Needed for expectation values.
        return self.get_rho_segment([]).squeeze()
    
    def get_rho_segment(self, segment, proj_Bs=None):
        """Return reduced density matrix for a segment, treating the doubled MPS as a density
        matrix directly. So we don't need two copies of the MPS. We simply trace over the
        legs not contained in the desired region.

        Note that the dimension of rho_A scales exponentially in the length of the segment.

        Parameters
        ----------
        segment : iterable of int
            Sites for which the reduced density matrix is to be calculated.
            Assumed to be sorted.
        proj_Bs : list of npc_arrays
            If we want to get the conditional reduced density matrix, we need to use the matrices
            having already projected out some physical legs.
        Returns
        -------
        rho : :class:`~tenpy.linalg.np_conserved.Array`
            Reduced density matrix of the segment sites.
            Labels ``'p0', 'p1', ..., 'pk', 'q0', 'q1', ..., 'qk'`` with ``k=len(segment)``.
        """
        if len(segment) > 6:
            warnings.warn("{0:d} sites in the segment, that's much!".format(len(segment)),
                          stacklevel=2)
        if len(segment) > 10:
            raise ValueError("too large segment; this is exponentially expensive!")
        segment = np.sort(segment)
        # We don't get any benefit from the canonical form here since we are not working with
        # two copies of the tensors. We want the new L1 canonical form (name in development).
        
        # So for each site not in segment, we need to contract the 'p' and 'q' legs with a trace.
        if self.bc != 'finite':
            raise NotImplementedError('Only works for finite dMPS (for now).')
        
        not_segment = list(set(range(self.L)) - set(segment))
        if proj_Bs is None:
            Ts = copy.deepcopy(self._B)
        else:
            Ts = copy.deepcopy(proj_Bs)
        for i in not_segment:
            if Ts[i].ndim == 4:
                Ts[i] = npc.trace(Ts[i], leg1='p', leg2='q')
            elif Ts[i].ndim != 2:
                raise ValueError('Too many legs.')
        rho = Ts[0]
        if 0 in segment:
            rho = self._replace_p_label(rho, str(0))
        for i, T in enumerate(Ts[1:]):
            rho = npc.tensordot(rho, T, axes=(['vR'], ['vL']))
            if i+1 in segment:
                rho = self._replace_p_label(rho, str(i+1))
        return rho
    
    def entanglement_entropy_segment(self, segment=[0], first_site=None, n=1):
        """Calculate entanglement entropy for general geometry of the bipartition, treating
        the doubled MPS as a density matrix.

        See documentation of `entanglement_entropy_segment` for function parameter details.
        
        To get a bipartite entanglement entropy, we could use this function and have segment
        specify the bipartition. But this will be expensive.
        """
        segment = np.sort(segment)
        if first_site is None:
            if self.finite:
                first_site = range(0, self.L - segment[-1])
            else:
                first_site = range(self.L)
        comb_legs = [
            self._get_p_labels(len(segment), False)[:len(segment)],
            self._get_p_labels(len(segment), False)[len(segment):]
        ]
        res = []
        for i0 in first_site:
            rho = self.get_rho_segment(segment + i0)
            rho = rho.combine_legs(comb_legs, qconj=[+1, -1])
            p = npc.eigvalsh(rho)
            res.append(entropy(p, n))
        return np.array(res)
    
    def probability_per_charge(self, bond=0):
        raise NotImplementedError()
        
    def average_charge(self, bond=0):
        raise NotImplementedError()

    def charge_variance(self, bond=0):
        raise NotImplementedError()
        
    def mutinf_two_site(self, max_range=None, n=1):
        """Calculate the two-site mutual information :math:`I(i:j)`, treating
        the doubled MPS as a density matrix.

        See documentation of `mutinf_two_site` for function parameter details.
        """
        # This is not very optimized; Each S_{ij} is calculated independently, which
        # is wasteful. SAJANT - Fix this?
        if max_range is None:
            max_range = self.L
        S_i = self.entanglement_entropy_segment(n=n)  # single-site entropy
        mutinf = []
        coord = []
        for i in range(self.L):
            jmax = i + max_range + 1
            if self.finite:
                jmax = min(jmax, self.L)
            for j in range(i + 1, jmax):
                rho_ij = self.get_rho_segment_rho([i, j])
                S_ij = entropy(npc.eigvalsh(rho_ij), n)
                mutinf.append(S_i[i] + S_i[j % self.L] - S_ij)
                coord.append((i, j))
        return np.array(coord), np.array(mutinf)
    
    
    # The MPS function samples as if the doubled MPS were an MPS; i.e. treate dMPS as purification.
    # Here we get the probability distribution on each site by taking traces over external sites.
    # We don't get any benefit from the canonical form -> L1 canonical form.
    def sample_measurements(self,
                            first_site=0,
                            last_site=None,
                            ops=None,
                            rng=None,
                            norm_tol=1.e-12):
        """Sample measurement results in the computational basis, treating the dMPS as
        a density matrix.
        
        Look at MPS.sample_measurements for documentation. One difference is that we return
        the total_prob rather than total_weight, which is the square root of total_prob with phase
        information. When working with density matrices, we don't have the phase. Additionally,
        we return the norm (i.e. traces) of the RDMs on each site.
        """
        assert self.bc == 'finite', "Infinite systems are weird without the L1 canonical form."
        if last_site is None:
            last_site = self.L - 1
        if rng is None:
            rng = np.random.default_rng()
        sigmas = []
        norms = []
        total_prob = 1.
        proj_Bs = copy.deepcopy(self._B)
        rho = self.get_rho_segment([first_site], proj_Bs=proj_Bs).replace_labels(['p0', 'q0'], ['p', 'q'])
        rho = rho.squeeze()
        assert rho.shape == (2,2)
        norm = npc.trace(rho, leg1='p', leg2='q')
        rho = rho / norm
        norms.append(norm)
        for i in range(first_site, last_site + 1):
            # rho = reduced density matrix on site i in basis vL [sigmas...] p p* vR
            # where the `sigmas` are already fixed to the measurement results
            
            # Check that rho is Hermitian and has trace 1
            # Trace 1 will fail since canonicalization messes up the norm, unless we normalize
            # which we do above.
            assert np.isclose(npc.trace(rho, leg1='p', leg2='q'), 1.0), "Not normalized"
            assert np.isclose(npc.norm(rho - rho.conj().transpose()), 0.0), "Not Hermitian"
            assert np.alltrue(npc.eig(rho)[0] > -1.e-8), "Not positive semidefinite"
            
            i0 = self._to_valid_index(i)
            site = self.sites[i0]
            if ops is not None:
                op_name = ops[(i - first_site) % len(ops)]
                op = site.get_op(op_name).transpose(['p', 'p*'])
                if npc.norm(op - op.conj().transpose()) > 1.e-13:
                    raise ValueError(f"measurement operator {op_name!r} not hermitian")
                W, V = npc.eigh(op)
                rho = npc.tensordot(V.conj(), theta, axes=['p*', 'p']).replace_label('p*', 'p')
                rho = npc.tensordot(theta, V, axes=(['q', 'p'])) # 'p', 'p*'
            else:
                W = np.arange(site.dim)
            rho_diag = np.abs(np.diag(rho.to_ndarray()))  # abs: real dtype & roundoff err
            if abs(np.sum(rho_diag) - 1.) > norm_tol:
                raise ValueError("not normalized to `norm_tol`")
            rho_diag /= np.sum(rho_diag)
            sigma = rng.choice(site.dim, p=rho_diag)  # randomly select index from probabilities
            sigmas.append(W[sigma])
            proj_Bs[i] = proj_Bs[i].take_slice([sigma, sigma], ['p','q'])  # project to sigma in theta for remaining rho
            total_prob *= rho_diag[sigma]
            if i != last_site:
                rho = self.get_rho_segment([i+1], proj_Bs=proj_Bs).replace_labels(['p' + str(i+1), 'q' + str(i+1)], ['p', 'q'])
                rho = rho.squeeze()
                assert rho.shape == (2,2)
                norm = npc.trace(rho, leg1='p', leg2='q')
                rho = rho / norm
                norms.append(norm)
        return sigmas, total_prob, norms
    
    def correlation_length(self, target=1, tol_ev0=1.e-8, charge_sector=0, return_charges=False):
        raise NotImplementedError()

    def correlation_length_charge_sectors(self, drop_symmetric=True, include_0=True):
        raise NotImplementedError()

    # SAJANT - need to write new versions of the above functions to work with the dMPS as a
    # density matrix. If we trace over the p, p* leg and contract together a unit cell, we get
    # a chi x chi transfer matrix. The eigenvalues of this should define correlation lengths.
        
    # We modify this so that the d x d operator is applied to both physical 'p' and conjugate
    # 'q' leg. So if we have a density matrix or operator rho, it becomes U rho U^\dagger.
    def apply_local_op(self, i, op, unitary=None, renormalize=False, cutoff=1.e-13,
                       understood_infinite=False):
        raise NotImplementedError()
    
    def apply_product_op(self, ops, unitary=None, renormalize=False):
        raise NotImplementedError()
    
    # We need to group legs to a regular MPS, perturb, and then convert back to a doulbed MPS.
    # Need to perturb with block diagonal U such that U = W \otimes W^\dagger to keep the 
    # density matrix / operator well defined.
    def perturb(self, randomize_params=None, close_1=True, canonicalize=None):
        raise NotImplementedError()
    
    # Bosonic (no JW operators) swap with 'p' and 'q' legs.
    # SAJANT - for fermions, we might want this option back.
    def swap_sites(self, i, swap_op='auto', trunc_par=None):
        assert swap_op is None
        if trunc_par is None:
            trunc_par = {}
        siteL, siteR = self.sites[self._to_valid_index(i)], self.sites[self._to_valid_index(i + 1)]
        theta = self.get_theta(i, n=2)
        C = self.get_theta(i, n=2, formL=0.)  # inversion free, see also TEBDEngine.update_bond()
        # just replace the labels, effectively this is a transposition.
        theta.ireplace_labels(['p0', 'p1', 'q0', 'q1'], ['p1', 'p0', 'q1', 'q0'])
        C.ireplace_labels(['p0', 'p1', 'q0', 'q1'], ['p1', 'p0', 'q1', 'q0'])
        theta = theta.combine_legs([('vL', 'p0', 'q0'), ('vR', 'p1', 'q1')], qconj=[+1, -1])
        U, S, V, err, renormalize = svd_theta(theta, trunc_par, inner_labels=['vR', 'vL'])
        B_R = V.split_legs(1).ireplace_labels(['p1', 'q1'], ['p', 'q'])
        B_L = npc.tensordot(C.combine_legs(('vR', 'p1', 'q1'), pipes=theta.legs[1]),
                            V.conj(),
                            axes=['(vR.p1.q1)', '(vR*.p1*.q1*)'])
        B_L.ireplace_labels(['vL*', 'p0', 'q0'], ['vR', 'p', 'q'])
        B_L /= renormalize  # re-normalize to <psi|psi> = 1
        self.set_SR(i, S)
        self.set_B(i, B_L, 'B')
        self.set_B(i + 1, B_R, 'B')
        self.sites[self._to_valid_index(i)] = siteR  # swap 'sites' as well
        self.sites[self._to_valid_index(i + 1)] = siteL
        return err
        
    def compute_K(self,
                  perm,
                  swap_op='auto',
                  trunc_par=None,
                  canonicalize=1.e-6,
                  verbose=None,
                  expected_mean_k=0.):
        raise NotImplementedError()
    
    # Below functions are copied from PurificationMPS, but inherited from BaseMPSExpectationValue
    
    def _replace_p_label(self, A, s):
        """Return npc Array `A` with replaced label, ``'p' -> 'p'+s, 'q' -> 'q'+s``."""
        return A.replace_labels(self._p_label, self._get_p_label(s))

    def _get_p_label(self, s, star=False):
        """return  self._p_label with additional string `s`."""
        return ['p' + s, 'q' + s]

    def _get_p_labels(self, ks, star=False):
        """join ``self._get_p_label(str(k) {+'*'} ) for k in range(ks)`` to a single list."""
        if star:
            return [lbl + str(k) + '*' for k in range(ks) for lbl in self._p_label]
        else:
            return [lbl + str(k) for k in range(ks) for lbl in self._p_label]
    
    #def _to_valid_index(self, i):
    
    # Below we turn off function assosiated with BaseMPSExpectationValue, as we don't want to take MPS
    # style measurements of a doubled MPS. Use MPSEnvironment instead.
    
    def expectation_value(self, ops, sites=None, axes=None):
        raise NotImplementedError()
        
    def expectation_value_multi_sites(self, operators, i0):
        raise NotImplementedError()
        
    def correlation_function(self,
                             ops1,
                             ops2,
                             sites1=None,
                             sites2=None,
                             opstr=None,
                             str_on_first=True,
                             hermitian=False,
                             autoJW=True):
        raise NotImplementedError()
        
    def expectation_value_term(self, term, autoJW=True):
        raise NotImplementedError()
        
    def term_correlation_function_right(self,
                                        term_L,
                                        term_R,
                                        i_L=0,
                                        j_R=None,
                                        autoJW=True,
                                        opstr=None):
        raise NotImplementedError()
        
    def term_correlation_function_left(self,
                                       term_L,
                                       term_R,
                                       i_L=None,
                                       j_R=0,
                                       autoJW=True,
                                       opstr=None):
        raise NotImplementedError()
        
    def term_list_correlation_function_right(self,
                                             term_list_L,
                                             term_list_R,
                                             i_L=0,
                                             j_R=None,
                                             autoJW=True,
                                             opstr=None):
        raise NotImplementedError()
        
    def _term_to_ops_list(self, term, autoJW=True, i_offset=0, JW_from_right=False):
        raise NotImplementedError()
        
    def _corr_up_diag(self, ops1, ops2, i, j_gtr, opstr, str_on_first, apply_opstr_first):
        raise NotImplementedError()
        
    def _corr_ops_LP(self, operators, i0):
        raise NotImplementedError()
        
    def _corr_ops_RP(self, operators, i0):
        raise NotImplementedError()
        
    def _expectation_value_args(self, ops, sites, axes):
        raise NotImplementedError()
        
    def _correlation_function_args(self, ops1, ops2, sites1, sites2, opstr):
        raise NotImplementedError()
        
    def get_op(self, op_list, i):
        raise NotImplementedError()
    
    def _normalize_exp_val(self, value):
        raise NotImplementedError()
        
    def _contract_with_LP(self, C, i):
        raise NotImplementedError()
        
    def _contract_with_RP(self, C, i):
        raise NotImplementedError()
        
    def _get_bra_ket(self):
        raise NotImplementedError()
    
# Bra for density matrix expectation value.
def trace_identity_DMPS(DMPS):
    d = DMPS.sites[0].dim
    I = np.eye(d).reshape(d, d, 1, 1)
    return DoubledMPS.from_Bflat(DMPS.sites,
                               [I] * DMPS.L,
                               SVs=None,
                               bc='finite',
                               dtype=None,
                               permute=True,
                               form='B', # Form doesn't matter since it's a product state?
                               legL=None)