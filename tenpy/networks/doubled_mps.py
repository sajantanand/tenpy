r"""This module contains an MPS class representing an density matrix by purification.

Usually, an MPS represents a pure state, i.e. the density matrix is :math:`\rho = |\psi><\psi|`,
describing observables as :math:`<O> = Tr(O|\psi><\psi|) = <\psi|O|\psi>`.
Clearly, if :math:`|\psi>` is the ground state of a Hamiltonian, this is the density matrix at
`T=0`.

At finite temperatures :math:`T > 0`, we want to describe a mixed density matrix
:math:`\rho = \exp(-H/T)`. The following approaches have been used to lift the power of tensor
network ansätze (representing pure states= to finite temperatures (and mixed states in general).

1. Naively represent the density matrix as an MPO. This has the disadvantage that truncation can
   quickly lead to non-positive (and hence unphysical) density matrices.
2. Minimally entangled typical thermal states (METTS) as introduced in :cite:`white2009`.
3. Use Purification to represent the mixed density matrix by pure states in the doubled Hilbert
   space.
   In the literature, this is also referred to as matrix product density operators (MPDO) or
   locally purified density operator (LPDO).


Here, we follow the third approach.
In addition to the physical space `P`, we introduce a second 'auxiliar' space `Q`
and define the density matrix
of the physical system as :math:`\rho = Tr_Q(|\phi><\phi|)`, where :math:`|\phi>` is a pure state
in the combined physical and auxiliary system.

For :math:`T=\infty`, the density matrix :math:`\rho_\infty` is the identity matrix.
In other words, expectation values are sums over all possible states
:math:`<O> = Tr_P(\rho_\infty O) = Tr_P(O)`.
Saying that each ``:`` on top is to be connected with the corresponding ``:`` on the bottom,
the trace is simply a contraction::

    |         :   :   :   :   :   :
    |         |   |   |   |   |   |
    |         |-------------------|
    |         |        O          |
    |         |-------------------|
    |         |   |   |   |   |   |
    |         :   :   :   :   :   :

Clearly, we get the same result, if we insert an identity operator, written as MPO, on the top
and bottom::

    |         :   :   :   :   :   :
    |         |   |   |   |   |   |
    |         B---B---B---B---B---B
    |         |   |   |   |   |   |
    |         |-------------------|
    |         |        O          |
    |         |-------------------|
    |         |   |   |   |   |   |
    |         B*--B*--B*--B*--B*--B*
    |         |   |   |   |   |   |
    |         :   :   :   :   :   :

We  use the following label convention::

    |         q
    |         ^
    |         |
    |  vL ->- B ->- vR
    |         |
    |         ^
    |         p

You can view the `MPO` as an MPS by combining the `p` and `q` leg and defining every physical
operator to act trivial on the `q` leg. In expectation values, you would then sum over
over the `q` legs, which is exactly what we need.
In other words, the choice :math:`B = \delta_{p,q}` with trivial (length-1) virtual bonds yields
infinite temperature expectation values for operators action only on the `p` legs!

Now, you go a step further and also apply imaginary time evolution (acting only on `p` legs)
to the initial infinite temperature state.
For example, the normalized state :math:`|\psi> \propto \exp(-\beta/2 H)|\phi>`
yields expectation values

.. math ::
    <O>  = Tr(\exp(-\beta H) O) / Tr(\exp(-\beta H))
    \propto <\phi|\exp(-\beta/2 H) O \exp(-\beta/2 H)|\phi>.

An additional real-time evolution allows to calculate time correlation functions:

.. math ::
    <A(t)B(0)> \propto <\phi|\exp(-\beta H/2) \exp(+i H t) A \exp(-i H t) B \exp(-\beta H/2) |\phi>

Time evolution algorithms (TEBD and MPO application) are adjusted in the module
:mod:`~tenpy.algorithms.purification`.

See also :cite:`karrasch2013` for additional tricks! One of their crucial observations is, that
one can apply arbitrary unitaries on the auxiliar space (i.e. the `q`) without changing the result.
This can actually be used to reduce the necessary virtual bond dimensions:
From the definition, it is easy to see that if we apply :math:`exp(-i H t)` to the `p` legs of
:math:`|\phi>`, and :math:`\exp(+iHt)` to the `q` legs, they just cancel out!
(They commute with :math:`\exp(-\beta H/2)`...)
If the state is modified (e.g. by applying `A` or `B` to calculate correlation functions),
this is not true any more. However, we still can find unitaries, which are 'optimal' in the sense
of reducing the entanglement of the MPS/MPO to the minimal value.
For a discussion of `Disentanglers` (implemented in :mod:`~tenpy.algorithms.disentanglers`),
see :cite:`hauschild2018`.

.. note ::
    The classes :class:`~tenpy.linalg.networks.mps.MPSEnvironment` and
    :class:`~tenpy.linalg.networks.mps.TransferMatrix` should also work for the
    :class:`PurificationMPS` defined here.
    For example, you can use :meth:`~tenpy.networks.mps.MPSEnvironment.expectation_value`
    for the expectation value of operators between different PurificationMPS.
    However, this makes only sense if the *same* disentangler was applied
    to the `bra` and `ket` PurificationMPS.

.. note ::
    The literature (e.g. section 7.2 of :cite:`schollwoeck2011` or :cite:`karrasch2013`) suggests
    to use a `singlet` as a maximally entangled state.
    Here, we use instead the identity :math:`\delta_{p,q}`, since it is easier to
    generalize for `p` running over more than two indices, and allows a simple use of charge
    conservation with the above `qconj` convention.
    Moreover, we don't split the physical and auxiliar space into separate sites, which makes
    TEBD as costly as :math:`O(d^6 \chi^3)`.
"""
# Copyright 2018-2023 TeNPy Developers, GNU GPLv3

import copy
import numpy as np
import itertools

from .mps import MPS
from .site import DoubledSite
from ..linalg import np_conserved as npc
from ..tools.math import entropy
from ..tools.misc import lexsort

__all__ = ['DoubledMPS']


class DoubledMPS(MPS):
    r"""SAJANT - Write documentation later
    """

    # we use q to mean p*, the bra leg of the density matrix or operator
    # `MPS.get_B` & co work, thanks to using labels. `B` just have the additional `q` labels.
    _p_label = ['p', 'q']  # this adjustment makes `get_theta` & friends work
    _B_labels = ['vL', 'p', 'q', 'vR']

    # SAJANT - Check; this was taken from PurificationMPS
    # Thanks to using `self._replace_p_label`,
    # correlation_function works as it should, if we adjust _corr_up_diag

    def test_sanity(self):
        """Sanity check, raises ValueErrors, if something is wrong."""
        for B in self._B:
            if not set(['vL', 'vR', 'p', 'q']) <= set(B.get_leg_labels()):
                raise ValueError("B has wrong labels " + repr(B.get_leg_labels()))
        super().test_sanity() # MPS check sanity
    
    #def copy(self):
    
    #def save_hdf5(self, hdf5_saver, h5gr, subpath):
    
    #def from_hdf5(cls, hdf5_loader, h5gr, subpath):
    
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
        new_MPS = MPS(doubled_sites, new_Bs, self._S, bc='finite', form='B')#, norm=norm)
        return new_MPS
    
    def from_regular_MPS(self, reg_MPS):
        """
        Convert a regular MPS back into a doubled MPS. We split the 'p' leg into 'p' and 'q'.
        """
        self._B = [B.replace_label('p', '(p.q)').split_legs() for B in reg_MPS._B]
        self._S = reg_MPS._S
        self.norm = reg_MPS.norm
        self.form = reg_MPS.form
        self.test_sanity()
    
    #def L(self):
    
    #def dim(self):
    
    #def finite(self):
    
    #def chi(self):
    
    #def nontrivial_bonds(self):
    
    #def get_B(self, i, form='B', copy=False, cutoff=1.e-16, label_p=None):
    
    #def set_B(self, i, B, form='B'):
    
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
        
    #def get_SL(self, i):
    
    #def get_SR(self, i):
    
    #def set_SL(self, i, S):
    
    #def set_SR(self, i, S):
    
    #def get_theta(self, i, n=2, cutoff=1.e-16, formL=1., formR=1.):
        # This function is returns the theta matrix for $n$ sites with $2n + 2$ indices, a 'p'
        # and 'q' for each tensor and a 'vL' and 'vR' at the ends.
        
    #def convert_form(self, new_form='B'):
    
    #def increase_L(self, new_L=None):
        # Depreciated in favor of enlarge_mps_unit_cell
        
    #def enlarge_mps_unit_cell(self, factor=2):
        # For infinite BCs
        
    #def roll_mps_unit_cell(self, shift=1):
    
    #def enlarge_chi(self, extra_legs, random_fct=np.random.normal):
    
    #def spatial_inversion(self):
    
    #def group_sites(self, n=2, grouped_sites=None):
    
    #def group_split(self, trunc_par=None):
    
    #def get_grouped_mps(self, blocklen):
    
    #def extract_segment(self, first, last):
    
    #def get_total_charge(self, only_physical_legs=False):
    
    #def gauge_total_charge(self, qtotal=None, vL_leg=None, vR_leg=None):
    
    r"""
    Entanglement entropy for density matrices (or operators) is weird, since we don't need two copies
    of the dMPS to get the density matrix. Instead, a single copy of the dMPS is the density matrix itself
    (as the name suggests).
    
    The below functions treat the dMPS as a pure state and calculate the traditional entanglement entropies.
    So given some region A, what we are calculating is something like $-Tr((\rho^2)_A \ln (\rho^2)_A)$, 
    where $\rho^2_A$ is found by taking two copies of the dMPS (easiest to think of it as a vectorized
    MPS with local Hilbert space d**2 rather than having two physical legs of dimension d) and tracing 
    over the complement of A (A-bar). So then we are left with a density matrix of region $A$ that has
    4 |A| legs (2 bras, 2 kets). We then project this into the bond space using the isometry tensors of
    \rho, which gets us back to an RDM just defined by the singular values on the bond (assuming that we
    are interested in a bipartition).
    
    As usual, all of this is done in terms of singular values wherever possible.
    """
    
    #def entanglement_entropy(self, n=1, bonds=None, for_matrix_S=False):
    
    #def entanglement_entropy_segment(self, segment=[0], first_site=None, n=1):
    
    #def entanglement_entropy_segment2(self, segment, n=1):
    
    #def entanglement_spectrum(self, by_charge=False):
    
    # This function treats the dMPS as a pure state
    #def get_rho_segment(self, segment):
    
    def get_rho_segment_rho(self, segment):
    """Return reduced density matrix for a segment, treating the doubled MPS as a density
        matrix directly. So we don't need two copies of the MPS. We simply trace over the
        legs not contained in the desired region.

        Note that the dimension of rho_A scales exponentially in the length of the segment.

        Parameters
        ----------
        segment : iterable of int
            Sites for which the reduced density matrix is to be calculated.
            Assumed to be sorted.

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
        
        # So for each site not in segment, we need to contract the 'p' and 'q' legs with a
        # trace.
        if self.bc != 'finite':
            raise NotImplementedError('Only works for finite dMPS (for now).')
        
        not_segment = list(set(range(self.L)) - set(segment))
        Ts = copy.deepcopy(self._B)
        for i in not_segment:
            Ts[i] = npc.trace(Ts[i], leg1='p', leg2='p')
        rho = Ts[0]
        for T in Ts[1:]:
            rho = npc.tensordot(rho, T, axes=(['vR'], ['vL']))
        rho = self._replace_p_label(rho, str(k))
        return rho
    
    def entanglement_entropy_segment_rho(self, segment=[0], first_site=None, n=1):
        r"""Calculate entanglement entropy for general geometry of the bipartition, treating
        the doubled MPS as a density matrix.

        See documentation of `entanglement_entropy_segment` for function parameter details.
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
            rho = self.get_rho_segment_rho(segment + i0)
            rho = rho.combine_legs(comb_legs, qconj=[+1, -1])
            p = npc.eigvalsh(rho)
            res.append(entropy(p, n))
        return np.array(res)
    
    #def probability_per_charge(self, bond=0):
    
    #def average_charge(self, bond=0):
    
    #def charge_variance(self, bond=0):
    
    #def mutinf_two_site(self, max_range=None, n=1):
    
    # SAJANT - write this
    #def mutinf_two_site_rho(self, max_range=None, n=1):
    
    
    
    # Need to finish going through MPS functions - Sajant
    # Include new function for getting actual reduced density matrix of some small region.
    
    def _normalize_exp_val(self, value):
        return np.real_if_close(value) #/ self.norm
    
    def _get_bra_ket(self):
        return trace_identity_DMPS(self), self
    
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

def trace_identity_DMPS(DMPS):
    d = DMPS.sites[0].dim
    I = np.eye(d).reshape(d, d, 1, 1)
    return DoubledMPS.from_Bflat(DMPS.sites,
                               [I] * DMPS.L,
                               SVs=None,
                               bc='finite',
                               dtype=None,
                               permute=True,
                               form='B',
                               legL=None)