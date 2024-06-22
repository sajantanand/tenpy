# Copyright 2018-2023 TeNPy Developers, GNU GPLv3

import copy
import numpy as np
import itertools
import warnings

from .mps import MPS, MPSEnvironment, BaseEnvironment
from .site import DoubledSite
from ..linalg import np_conserved as npc
from ..linalg.charges import LegPipe
from ..tools.math import lcm, entropy
from ..tools.misc import lexsort
from ..tools.cache import DictCache

__all__ = ['DoubledMPS', 'NonTrivialStackedDoubledMPSEnvironment']

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
                B = B[site.perm, :, :, :][:, site.perm, :, :]
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

    def to_regular_MPS(self, hermitian=True, doubled_site=None, warn=True):
        """
        Convert a doubled MPS to a regular MPS by combining together the 'p' and 'q' legs
        """
        # Build new site of squared dimension
        if doubled_site is None:
            doubled_sites = [DoubledSite(s.dim, s.conserve, s.leg.sorted, hermitian) for s in self.sites]# * self.L
        else:
            doubled_sites = [doubled_site] * self.L

        new_Bs = [B.combine_legs(('p', 'q')).replace_label('(p.q)', 'p') for B in self._B]
        #pipes = [B.get_leg('p') for B in new_Bs]
        if warn and not np.isclose(self.norm, 1.0):
            warnings.warn("to_regular_MPS: DMPS has norm != 1; this IS copied over to the MPS! DMPS norm: " + str(self.norm), stacklevel=3)
        new_MPS = MPS(doubled_sites, new_Bs, self._S, bc='finite', form='B', norm=self.norm)
        new_MPS.canonical_form(renormalize=False) # norm now contains the rescaling factor needed to establish
        # newMPS as a normalized MPS.
        return new_MPS#, pipes

    def from_regular_MPS(self, reg_MPS): #, pipes):
        """
        Convert a regular MPS back into a doubled MPS. We split the 'p' leg into 'p' and 'q'.
        """
        #for B, pipe in zip(reg_MPS._B, pipes):
            #B.itranspose(['vL', 'p', 'vR'])
            #B.legs[1] = pipe
        # Only want to split the physical leg. Sometime the vL and vR legs might be pipes too.
        self._B = [B.replace_label('p', '(p.q)').split_legs('(p.q)') for B in reg_MPS._B]
        for B in self._B:
            B.itranspose(self._B_labels)
        self._S = reg_MPS._S
        self.norm = reg_MPS.norm
        self.form = reg_MPS.form
        # Don't want to do this as of 05/09/2024; when doing backflow experiments, we want to have finite MPS with a dangling
        # right vR leg; test_sanity() doesn't allow for this.
        #self.test_sanity()

    def outer_product(self):
        """
        Take outer product of each tensor; O \rightarrow O \otimes O
        """

        grouped_sites = []
        for s in s.sites:
            gs = GroupedSite([s, s], ['0', '1'], 'same')
            grouped_sites.append(gs)

        new_Bs = []
        new_Ss = []
        for B in self._B:
            new_B = npc.outer(B.replace_labels(['p', 'vL', 'vR'], ['p0', 'vL0', 'vR0']),
                              B.replace_labels(['p', 'vL', 'vR'], ['p1', 'vL1', 'vR1']))
            new_B = new_B.combine_legs([['p0', 'p1'], ['vL0', 'vL1'], ['vR0', 'vR1']],
                                        qconj=[B.get_leg('p').qconj, B.get_leg('vL'), B.get_leg('vR')])
            new_B.ireplace_labels(['(p0.p1)', '(vL0.vL1)', '(vR0.vR1)'], ['p', 'vL', 'vR'])
            new_Bs.append(new_B)
            new_Ss.append(np.kron(new_Ss, new_Ss))
        if not np.isclose(self.norm, 1.0):
            warnings.warn("outer_product: DMPS has norm != 1; this IS copied over to the outer producted DMPS! DMPS norm: " + str(self.norm), stacklevel=3)
        new_MPS = MPS(grouped_sites, new_Bs, new_Ss, bc='finite', form='B', norm=self.norm)
        return new_MPS

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

    def trace(self, trace_env=None):
        # trace = Tr(I * rho) * rho.norm
        # Needed for expectation values.
        if trace_env is None:
            from ..algorithms import dmt_utils as dmt
            trace_env = MPSEnvironment(dmt.trace_identity_DMPS(self), self) # Includes norm of self
            # bra has norm 1 (i.e. we don't set the DMPS norm)
        return trace_env.full_contraction(0), trace_env

    def get_1RDM(self, site, left_env=None, right_env=None):
        if self.bc != 'finite':
            raise NotImplementedError('Only works for finite dMPS (for now).')
        site = self._to_valid_index(site)

        if left_env is None:
            # calculate left_env
            left_env = npc.eye_like(self.get_B(0), axis='vL', labels=['vL', 'vR'])
            left_envs = [left_env]
            for i in range(site):
                B = npc.trace(self.get_B(i, form='B', copy=False), leg1='p', leg2='q')
                left_env = npc.tensordot(left_env, B, axes=(['vR', 'vL']))
                left_envs.append(left_env)
            assert len(left_envs) == site + 1
        else:
            left_envs = [left_env]

        if right_env is None:
            # calculate right_env
            right_env = npc.eye_like(self.get_B(self.L-1), axis='vR', labels=['vL', 'vR'])
            right_envs = [right_env]
            for i in reversed(range(site+1, self.L)):
                B = npc.trace(self.get_B(i, form='B', copy=False), leg1='p', leg2='q')
                right_env = npc.tensordot(B, right_env, axes=(['vR', 'vL']))
                right_envs.append(right_env)
            right_envs = right_envs[::-1]
            assert len(right_envs) == self.L - site
        else:
            right_envs = [right_env]

        B = self.get_B(site, form='B', copy=True)
        rho = npc.tensordot(npc.tensordot(left_env, B, axes=(['vR', 'vL'])), right_env, axes=(['vR', 'vL']))
        return rho, left_envs, right_envs

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
            Ts = [self.get_B(i, form='B', copy=True) for i in range(self.L)]
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
        return rho * self.norm

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
                            norm_tol=1.e-12,
                            verbose=False,
                            right_envs=None):
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

        if right_envs is None:
            rho, left_envs, right_envs = self.get_1RDM(0, left_env=None, right_env=None)
        else:
            rho, left_envs, _ = self.get_1RDM(0, left_env=None, right_env=right_envs[0])

        left_env = left_envs[0] # Identity to the right of site 0
        rho = rho.squeeze()
        assert rho.shape == (2,2)

        norm = npc.trace(rho, leg1='p', leg2='q')
        rho = rho / norm
        norms.append(norm)

        for i in range(first_site, last_site + 1):
            # rho = reduced density matrix on site i in basis vL [sigmas...] p p* vR
            # where the `sigmas` are already fixed to the measurement results

            # Check that rho is Hermitian and has trace 1
            # Trace 1 will fail since canonicalization messes up the norm, unless we normalize which we do above.
            # Additionally, rho will be neither hermitian nor positive since we truncate.

            # Let's not do these assert statements since they may fail and we are OK with that.
            # assert np.isclose(npc.trace(rho, leg1='p', leg2='q'), 1.0), f"Not normalized, {npc.trace(rho, leg1='p', leg2='q')}."
            # assert np.isclose(npc.norm(rho - rho.conj().transpose()), 0.0), f"Not Hermitian, {npc.norm(rho - rho.conj().transpose())}."
            # assert np.all(npc.eig(rho)[0] > -1.e-8), f"Not positive semidefinite, {npc.eig(rho)[0]}."
            if verbose:
                print("Metrics on site ", i)
                print("Trace: ", npc.trace(rho, leg1='p', leg2='q'), abs(np.sum(np.abs(np.diag(rho.to_ndarray()))) - 1.))
                print("Hermiticity: ", npc.norm(rho - rho.conj().transpose()))
                lamb = npc.eig(rho)[0]
                #print("Positivity: ", np.all(lamb > -1.e-8))
                print("Positivity: ", lamb[lamb < 0])

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
            rho = rho.to_ndarray()

            # Make Hermitian to be safe
            rho = (rho + rho.T.conj()) / 2

            rho_diag = np.diag(rho).real
            assert np.all(rho_diag > 0), "1-site RDM on site " + str(i)+ "  is not positive."

            # Guaranteed to be Hermitian and positive at this point; is it normalized?
            if abs(np.sum(rho_diag) - 1.) > norm_tol:
                print(abs(np.sum(rho_diag) - 1))
                raise ValueError("not normalized up to `norm_tol`")
            rho_diag = rho_diag / np.sum(rho_diag)

            sigma = rng.choice(site.dim, p=rho_diag)  # randomly select index from probabilities
            sigmas.append(W[sigma])
            sliced_B = self.get_B(i, form='B').take_slice([sigma, sigma], ['p','q'])  # project to sigma in theta for remaining rho
            total_prob *= rho_diag[sigma]
            if i != last_site:
                # Get 1-site conditional RDM on site i+1 using the result on site i
                left_env = npc.tensordot(left_env, sliced_B, axes=(['vR', 'vL']))
                rho, _, _ = self.get_1RDM(i+1, left_env=left_env, right_env=right_envs[i+1])
                rho = rho.squeeze()
                assert rho.shape == (2,2)

                norm = npc.trace(rho, leg1='p', leg2='q')
                rho = rho / norm
                norms.append(norm)
        return sigmas, total_prob, norms, right_envs

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

    def translate(self, Tx):
        """Translate the MPS by some number of sites, wrapping around the edges. This is used when we
        want to approximate having periodic boundary conditions applying the translation operator to the
        doubled MPS

        Parameters
        ----------
        Tx : int
            How many sites do we shift to the RIGHT
            If Tx=1, [0,1,2,3] -> [3,0,1,2]
        Returns
        -------
        rho : :class:`~tenpy.networks.DoubledMPS`
            Doubled MPS after we shift. This is a copy of the original dMPS, even if we don't shift.
        """
        Tx *= -1    # The way I implemented this, we shift to the left
        # Tx = L has no effect
        Tx = Tx % self.L
        if Tx > 0:
            Bs = list(self._B[Tx:]) + list(self._B[0:Tx])
            #SVs = list(self._S[Tx:]) + list(self._S[0:Tx])
            # SVs on first and last bond are the trivial; we need to have non-trivial and different SVs on the bonds now.
            SVs = list(self._S[Tx:]) + list(self._S[1:Tx+1])
            forms = list(self.form[Tx:]) + list(self.form[0:Tx])
            sites = list(self.sites[Tx:]) + list(self.sites[0:Tx])
            tpsi = self.__class__(sites, Bs, SVs, form=forms, bc='infinite', norm = self.norm)
            tpsi.bc = 'finite'
            return tpsi
        else:
            Tx == 0
            return self.copy()

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

class NonTrivialStackedDoubledMPSEnvironment(BaseEnvironment):
    """Class for computing trace of an arbitrary number of :class:`DoubledMPS` stacked one on top of
    one another. We want to calculate Tr(A B C D ...) where each is a :class:`DoubledMPS`. This will
    take the trace into account.

    Crucially, the MPS will have non-trivial virtual boundary conditions on the leftmost and rightmost bonds.
    We do NOT require that the the left (right) virtual leg of bra is compatible to that of the ket; all we require is
    that the left (right) leg of the bra is compatible with that of the ket.

    This class stores the partial contractions up to each bond.

    The MPSs `kets` have to be in canonical form.
    All the environments are constructed without the singular values on the open bond.
    In other words, we contract left-canonical `A` to the left parts `LP`
    and right-canonical `B` to the right parts `RP`.

    This is essentially a bastardization of BaseEnvironment and NonTrivialEnvironment from mps.py.

    The following is the label convention (where arrows indicate `qconj`) for NonTrivialEnvironments from mps.py.
    The left environments we actually work with have an arbitrary number of vL legs; vL1, vL2, ...

        |   vL1 ->-.-->- vR1         vL1 ->-.-->- vR1
        |          |                        |
        |          LP                       RP
        |          |                        |
        |   vL2 ->-.-->- vR2         vL2 ->-.-->- vR2

    Parameters - See BaseEnvironment for documentation of parameters.
    ----------
    kets : list of :class:`~tenpy.networks.mps.MPS`
        Each MPS should be given in usual 'ket' form;
        we never call `conj()` on the matrices directly.
        Stored in place, without making copies.
        If necessary to match charges, we call :meth:`~tenpy.networks.mps.MPS.gauge_total_charge`.
    cache : :class:`~tenpy.tools.cache.DictCache` | None
        Cache in which the tensors should be saved. If ``None``, a new `DictCache` is generated.
    **init_env_data :
        Further keyword arguments with initialization data, as returned by
        :meth:`get_initialization_data`.
        See :meth:`initialize_first_LP_last_RP` for details on these parameters.

    Attributes - See BaseEnvironment for documentation of attributes.
    ----------
    """

    def __init__(self, kets, cache=None, **init_env_data):
        self.num_kets = len(kets)
        self.kets = kets
        #assert self.num_kets > 1
        for i in range(1, self.num_kets):
            kets[i] = kets[0]._gauge_compatible_vL_vR(kets[i])  # ensure matching charges

        self.dtype = kets[0].dtype
        self.L = L = kets[0].L
        for i in range(1, self.num_kets):
            self.dtype = np.promote_types(self.dtype, kets[i].dtype)
            self.L = L = lcm(L, kets[i].L)

        # We do not allow calculations with a Hamiltonian

        # Only works for finite DMPS and no segments (SAJANT: TODO? Do we want to implement this?)
        # Some random parts of the segments code is already implemented. . .
        for i in range(self.num_kets):
            assert self.kets[i].finite
            assert self.kets[i].bc == "finite"

        self.finite = self.kets[0].finite  # just for _to_valid_index
        self.sites = self.kets[0].sites * (L // self.kets[0].L)
        self._LP_keys = ['LP_{0:d}'.format(i) for i in range(L)]
        self._RP_keys = ['RP_{0:d}'.format(i) for i in range(L)]
        self._LP_age = [None] * L
        self._RP_age = [None] * L
        if cache is None:
            cache = DictCache.trivial()
        self.cache = cache
        if not self.cache.long_term_storage.trivial and L < 8:
            warnings.warn("non-trivial cache for short-length environment: "
                          "Much overhead for a little RAM saving. Necessary?")
        self.init_first_LP_last_RP(**init_env_data)
        self.test_sanity()

    def init_first_LP_last_RP(self,
                              init_LP=None,
                              init_RP=None,
                              age_LP=0,
                              age_RP=0,
                              start_env_sites=0):
        """(Re)initialize first LP and last RP from the given data.

        Parameters
        ----------
        init_LP : ``None`` | :class:`~tenpy.linalg.np_conserved.Array`
            Initial very left part ``LP``. If ``None``, build one with :meth`init_LP`.
        init_RP : ``None`` | :class:`~tenpy.linalg.np_conserved.Array`
            Initial very right part ``RP``. If ``None``, build one with :meth:`init_RP`.
        age_LP : int
            The number of physical sites involved into the contraction of `init_LP`.
        age_RP : int
            The number of physical sites involved into the contraction of `init_RP`.
        start_env_sites : int
            If `init_LP` and `init_RP` are not specified, contract each `start_env_sites` for them.
        """
        init_LP, init_RP = self._check_compatible_legs(init_LP, init_RP, start_env_sites)
        kets_U, kets_V = [], []
        for i in range(self.num_kets):
            ket_U, ket_V = self.kets[i].segment_boundaries
            kets_U.append(ket_U)
            kets_V.append(ket_V)

        if init_LP is None:
            init_LP = self.init_LP(0, start_env_sites)
            age_LP = start_env_sites
        else:
            for i in range(self.num_kets):
                if kets_U[i] is not None:
                    init_LP = npc.tensordot(init_LP, kets_U[i], axes=['vR' + str(i), 'vL']).replace_label('vR', 'vR' + str(i))
        if init_RP is None:
            init_RP = self.init_RP(self.L - 1, start_env_sites)
            age_RP = start_env_sites
        else:
            for i in range(self.num_kets):
                if kets_V[i] is not None:
                    init_RP = npc.tensordot(kets_V[i], init_RP, axes=['vR', 'vL' + str(i)]).replace_label('vL' + str(i), 'vL')
        self.set_LP(0, init_LP, age=age_LP)
        self.set_RP(self.L - 1, init_RP, age=age_RP)

    def test_sanity(self):
        """Sanity check, raises ValueErrors, if something is wrong."""
        for i in range(self.num_kets):
            assert (self.kets[i].finite == self.finite)
        assert any(key in self.cache for key in self._LP_keys)
        assert any(key in self.cache for key in self._RP_keys)

    def _check_compatible_legs(self, init_LP, init_RP, start_env_sites):
        if init_LP is not None or init_RP is not None:
            vL_kets, vR_kets = [], []
            for i in range(self.num_kets):
                if start_env_sites == 0:
                    vL_ket, vR_ket = self.kets[i].outer_virtual_legs()
                    vL_kets.append(vL_ket)
                    vR_kets.append(vR_ket)
                else:
                    vL_ket = self.kets[i].get_B(-start_env_sites, 'A').get_leg('vL')
                    vR_ket = self.kets[i].get_B(self.L - 1 + start_env_sites, 'B').get_leg('vR')
                    vL_kets.append(vL_ket)
                    vR_kets.append(vR_ket)
        if init_LP is not None:
            for i in range(self.num_kets):
                compatible = (init_LP.get_leg('vR' + str(i)) == vL_kets[i].conj())
                if not compatible:
                    warnings.warn("dropping `init_LP` with incompatible MPS legs")
                    init_LP = None
        if init_RP is not None:
            for i in range(self.num_kets):
                compatible = (init_RP.get_leg('vL' + str(i)) == vR_kets[i].conj())
                if not compatible:
                    warnings.warn("dropping `init_RP` with incompatible MPS legs")
                    init_RP = None
        return init_LP, init_RP

    def init_LP(self, i, start_env_sites=0):
        """Build initial left part ``LP``.

        This is the environment you get contracting the overlaps from the left infinity
        up to bond left of site `i`.

        For segment MPS, the :attr:`~tenpy.networks.mps.MPS.segment_boundaries` are read out
        (if set).

        Parameters
        ----------
        i : int
            Build ``LP`` left of site `i`.
        start_env_sites : int
            How many sites to contract to converge the `init_LP`; the initial `age_LP`.

        Returns
        -------
        init_LP : :class:`~tenpy.linalg.np_conserved.Array`
            Identity contractible with the `vL` leg of ``ket.get_B(i)``, labels ``'vR*', 'vR'``.
        """
        vL_kets = []
        for j in range(self.num_kets):
            vL_kets.append(self.kets[j].get_B(i - start_env_sites, None).get_leg('vL'))
        combined_leg_ket = LegPipe(vL_kets, vL_kets[0].qconj)
        #print(combined_leg_ket)

        #if self.num_kets > 1:
        combined_label_vL = "".join(['('] + ['vL' + str(i) + '.' for i in range(self.num_kets-1)] + ['vL'] + [str(self.num_kets-1)] + [')'])
        combined_label_vR = "".join(['('] + ['vR' + str(i) + '.' for i in range(self.num_kets-1)] + ['vR'] + [str(self.num_kets-1)] + [')'])
        #print(combined_label_vL, combined_label_vR)

        init_LP = npc.diag(1., combined_leg_ket, dtype=self.dtype, labels=[combined_label_vL, combined_label_vR]) # has 2 * self.num_kets legs
        init_LP = init_LP.split_legs()

        for j in range(i - start_env_sites, i):
            init_LP = self._contract_LP(j, init_LP)

        #print(init_LP)
        return init_LP

        """
        if self.ket.bc == "segment":
            U_bra, V_bra = self.bra.segment_boundaries
            U_ket, V_ket = self.ket.segment_boundaries
            if U_bra is not None or U_ket is not None:
                if U_bra is not None and U_ket is not None:
                    init_LP = npc.tensordot(U_bra.conj(), U_ket, axes=['vL*', 'vL'])
                elif U_bra is not None:
                    init_LP = U_bra.conj().ireplace_label('vL*', 'vR')
                else:
                    init_LP = U_ket.replace_label('vL', 'vR*')
                return init_LP
        """

    def init_RP(self, i, start_env_sites=0):
        """Build initial right part ``RP`` for an MPS/MPOEnvironment.

        If `bra` and `ket` are the same and in right canonical form, this is the environment
        you get contracting from the right infinity up to bond right of site `i`.

        For segment MPS, the :attr:`~tenpy.networks.mps.MPS.segment_boundaries` are read out
        (if set).

        Parameters
        ----------
        i : int
            Build ``RP`` right of site `i`.
        start_env_sites : int
            How many sites to contract to converge the `init_RP`; the initial `age_RP`.

        Returns
        -------
        init_RP : :class:`~tenpy.linalg.np_conserved.Array`
            Identity contractible with the `vR` leg of ``ket.get_B(i)``, labels ``'vL*', 'vL'``.
        """
        vR_kets = []
        for j in range(self.num_kets):
            vR_kets.append(self.kets[j].get_B(i + start_env_sites, None).get_leg('vR'))
        combined_leg_ket = LegPipe(vR_kets, vR_kets[0].qconj)
        #print(combined_leg_ket)

        #if self.num_kets > 1:
        combined_label_vL = "".join(['('] + ['vL' + str(i) + '.' for i in range(self.num_kets-1)] + ['vL'] + [str(self.num_kets-1)] + [')'])
        combined_label_vR = "".join(['('] + ['vR' + str(i) + '.' for i in range(self.num_kets-1)] + ['vR'] + [str(self.num_kets-1)] + [')'])
        #print(combined_label_vL, combined_label_vR)

        init_RP = npc.diag(1., combined_leg_ket, dtype=self.dtype, labels=[combined_label_vR, combined_label_vL]) # has 2 * self.num_kets legs
        init_RP = init_RP.split_legs()

        for j in range(i + start_env_sites, i, -1):
            init_RP = self._contract_RP(j, init_RP)

        #print(init_RP)
        return init_RP

    def get_initialization_data(self, first=0, last=None, include_bra=False, include_ket=False):
        raise NotImplementedError("Not sure we need this.")

    def _full_contraction_LP_RP(self, i0):
        if self.finite and i0 + 1 == self.L:
            # special case to handle `_to_valid_index` correctly:
            # get_LP(L) is not valid for finite b.c, so we use need to calculate it explicitly.
            LP = self.get_LP(i0, store=False)
            LP = self._contract_LP(i0, LP)
        else:
            LP = self.get_LP(i0 + 1, store=False)
        # multiply with `S` on bra and ket side
        for i in range(self.num_kets):
            S = self.kets[i].get_SR(i0)
            if isinstance(S, npc.Array):
                LP = npc.tensordot(LP, S, axes=['vR' + str(i), 'vL']).replace_label('vR', 'vR' + str(i))
            else:
                LP = LP.scale_axis(S, 'vR' + str(i))

        RP = self.get_RP(i0, store=False)
        return LP, RP

    # Functions from "BaseMPSExpectationValue"
    def _to_valid_index(self, i):
        """Make sure `i` is a valid index (depending on `finite`)."""
        if not self.finite:
            return i % self.L
        if i < 0:
            i += self.L
        if i >= self.L or i < 0:
            raise KeyError("i = {0:d} out of bounds for finite MPS".format(i))
        return i

    # Functions from "MPSEnvironment"
    def full_contraction(self, i0):
        """Calculate the overlap by a full contraction of the network.

        The full contraction of the environments gives the overlap ``<bra|ket>``,
        taking into account the :attr:`MPS.norm` of both `bra` and `ket`.
        For this purpose, this function contracts ``get_LP(i0+1, store=False)`` and
        ``get_RP(i0, store=False)`` with appropriate singular values in between.

        Parameters
        ----------
        i0 : int
            Site index.
        """
        LP, RP = self._full_contraction_LP_RP(i0)
        combined_label_vL = ['vL' + str(i)  for i in range(self.num_kets)]
        combined_label_vR = ['vR' + str(i)  for i in range(self.num_kets)]

        contr = npc.inner(LP, RP, axes=[combined_label_vR + combined_label_vL, combined_label_vL + combined_label_vR], do_conj=False)
        return self._normalize_exp_val(contr)

    def _contract_LP(self, i, LP):
        LP = npc.tensordot(LP, self.kets[0].get_B(i, form='A'), axes=('vR0', 'vL')).replace_label('vR', 'vR0')
        if self.num_kets > 1:
            for j in range(1, self.num_kets-1):
                axes = (['q', 'vR' + str(j)], ['p', 'vL'])
                LP = npc.tensordot(LP, self.kets[j].get_B(i, form='A'), axes=axes).replace_label('vR', 'vR' + str(j))
            j = self.num_kets-1
            axes = (['p', 'q', 'vR' + str(j)], ['q', 'p', 'vL'])
            LP = npc.tensordot(LP, self.kets[j].get_B(i, form='A'), axes=axes).replace_label('vR', 'vR' + str(j))
        else:
            LP = npc.trace(LP, leg1='p', leg2='q')
        return LP

    def _contract_RP(self, i, RP):
        RP = npc.tensordot(self.kets[0].get_B(i, form='B'), RP, axes=('vR', 'vL0')).replace_label('vL', 'vL0')
        if self.num_kets > 1:
            for j in range(1, self.num_kets-1):
                axes = (['p', 'vR'], ['q', 'vL' + str(j)])
                RP = npc.tensordot(self.kets[j].get_B(i, form='B'), RP, axes=axes).replace_label('vL', 'vL' + str(j))
            j = self.num_kets-1
            axes = (['p', 'q', 'vR'], ['q', 'p', 'vL'  + str(j)])
            RP = npc.tensordot(self.kets[j].get_B(i, form='B'), RP, axes=axes).replace_label('vL', 'vL' + str(j))
        else:
            RP = npc.trace(RP, leg1='p', leg2='q')
        return RP

    # methods for Expectation values
    def _get_bra_ket(self):
        raise NotImplementedError("Not sure we need this.")

    def _normalize_exp_val(self, value):
        # this ensures that
        #     MPSEnvironment(psi, psi.apply_local_op('B', i)).expectation_value('A', j)
        # gives the same as
        #     psi.correlation_function('A', 'B', sites1=[i], sites2=[j])
        # and psi.apply_local_op('Adagger', i).overlap(psi.apply_local_op('B', j)
        # for initially normalized psi
        return np.real_if_close(value) * np.prod([self.kets[i].norm for i in range(self.num_kets)])

    def _contract_with_LP(self, C, i):
        raise NotImplementedError("Not sure we need this.")

    def _contract_with_RP(self, C, i):
        raise NotImplementedError("Not sure we need this.")
