"""Circuit evolution by discrete quantum gates; multi qubit gates are specified by MPOs."""

# Copyright (C) TeNPy Developers, Apache license

import numpy as np
from copy import deepcopy
import time
import typing
import warnings
import logging
from functools import reduce

logger = logging.getLogger(__name__)

from .algorithm import TimeEvolutionAlgorithm
from ..linalg.truncation import svd_theta, TruncationError, _machine_prec_trunc_par
from ..linalg import np_conserved as npc
from ..networks.mpo import MPO
from ..tools.misc import consistency_check

__all__ = ['CircuitEvolution']


class CircuitEvolution(TimeEvolutionAlgorithm):
    """Discrete circuit evolution of an MPS``.

    We want to evolve a quantum state by gates, both single and multi site. To do this,
    the user passes in the sequence of gates that form one layer of the circuit
    evolution. Single site gates are given as npc arrays while multi site gates are
    packaged together into MPOs.


    Parameters
    ----------
    psi :
        Tensor network to be updated by the algorithm.
    model : list of :class:`~tenpy.linalg.npc.Array` or :class:`~tenpy.networks.mpo.MPO`
        Gates that make up a single layer of the circuit
        We do not take a regular TeNPy model.
    options : dict-like
        Optional parameters for the algorithm.
        In the online documentation, you can find the correct set of options in the
        :ref:`cfg-config-index`.
    resume_data : None | dict
        Can only be passed as keyword argument.
        By default (``None``) ignored. If a `dict`, it should contain the data returned by
        :meth:`get_resume_data` when intending to continue/resume an interrupted run.
        If it contains `psi`, this takes precedence over the argument `psi`.
    cache : None | :class:`DictCache`
        The cache to be used to reduce memory usage.
        None defaults to a new, trivial :class:`DictCache` which keeps everything in RAM.

    Options
    -------
    .. cfg:config :: ExpMPOEvolution
        :include: ApplyMPO, TimeEvolutionAlgorithm

    Attributes
    ----------
    _U : list of :class:`~tenpy.linalg.npc.Array` or :class:`~tenpy.networks.mpo.MPO`
        Gates that make up a single layer of the circuit
    """
    def __init__(self, psi, model, options, **kwargs):
        super().__init__(psi, None, options, **kwargs)
        options = self.options
        self._U = model

    # run from TimeEvolutionAlgorithm

    def prepare_evolve(self, dt):
        # Nothing to do here. MPOs are already made.
        pass

    def evolve_step(self, dt):
        trunc_err = TruncationError()
        for U in self._U:
            if isinstance(U, npc.Array):
                self.psi.apply_product_op([U])
            elif isinstance(U, MPO):
                trunc_err += U.apply(self.psi, self.options)
                
                # Below is for DMT; need to check this.
                MPO_envs = self.options.get('MPO_envs', None)
                if MPO_envs is not None:
                    for Me in MPO_envs:
                        Me.clear()
            else:
                raise ValueError('U is neither an MPO nor an npc.array:', type(U))

        return trunc_err


########## Generate coupling MPO given two-site gate ##########

def U_machinery(U, cutoff=1.e-12):
    """
    Given a two-site gate U, we want to decompose it as U = sum_i A_i otimes B_i
    """
    A, S, B, trunc_err, renormalize = svd_theta(U.combine_legs([['p0','p0*'],['p1','p1*']]),
                                                {'chi_max': None, 'svd_min': cutoff},
                                                [U.qtotal, None],
                                                inner_labels=['wR', 'wL'],
                                                renormalize=False)
    A.iscale_axis(np.sqrt(S), axis='wR')
    B.iscale_axis(np.sqrt(S), axis='wL')
    A = A.split_legs().replace_labels(['p0','p0*'],['p','p*'])
    B = B.split_legs().replace_labels(['p1','p1*'],['p','p*'])
    A = A.add_trivial_leg(axis=2, label='wL', qconj=1)
    B = B.add_trivial_leg(axis=1, label='wR', qconj=-1)
    A.itranspose(['p','p*','wL','wR'])
    B.itranspose(['p','p*','wL','wR'])

    D = len(S)
    d = U.get_leg('p0').ind_len
    
    leg_phys = U.get_leg('p0')
    Id_npc = npc.Array.from_ndarray(np.eye(d), [leg_phys, leg_phys.conj()], labels=['p', 'p*'])

    Id2     = np.empty((D,D), dtype=object)
    for i in range(D):
        Id2[i,i] = Id_npc

    Open    = A.sort_legcharge()[1]
    Close   = B.sort_legcharge()[1]
    Id1     = Id_npc.add_trivial_leg(2, label='wL', qconj=1).add_trivial_leg(3, label='wR', qconj=-1).sort_legcharge()[1]
    Id2     = npc.grid_outer(Id2, grid_legs=[B.get_leg('wL'), A.get_leg('wR')], grid_labels=['wL', 'wR']).sort_legcharge()[1]
    Id2.itranspose(['p','p*','wL','wR'])

    return Open, Close, Id1, Id2

def outer_tensor(A, B):
    """
    Contract over the physical leg to put together the two tensors.
    
       p
       |
    ---A---
       |
       |
    ---B---
       |
       p*

    """

    T = npc.tensordot(A, B.replace_labels(['wL', 'wR'], ['vL', 'vR']), axes=(['p*'],['p']))
    T = T.combine_legs([['wL', 'vL'], ['wR', 'vR']], qconj=[1,-1]).replace_labels(['(wL.vL)', '(wR.vR)'], ['wL', 'wR']).sort_legcharge()[1]
    #T.legs[T.get_leg_index('wL')] = T.get_leg('wL').to_LegCharge()
    #T.legs[T.get_leg_index('wR')] = T.get_leg('wR').to_LegCharge()
    return T

def U_MPO_disjoint(site, L, pairs, Open, Close, Id1, Id2, lonely = []):
    # sort by smaller site of the pair
    pairs = [((i, j) if i < j else (j, i)) for (i, j) in pairs]

    # sort by starting site
    pairs.sort(key=lambda x: x[0])

    # ending characters
    pairs.append((L, L))
    lonely = sorted(lonely) + [L]

    Bs = []
    #forms = []
    open_singlets = []  # the k-th open singlet should be closed at site open_singlets[k]
    Ts = []  # the tensors on the current site
    #labels_L = []
    for i in range(L):
        #labels_R = labels_L[:]
        next_Ts = Ts[:]
        if i == pairs[0][0]:  # open a new singlet
            j = pairs[0][1]
            pairs.pop(0)
            open_singlets.append(j)
            next_Ts.append(Id2)
            Ts.append(Open)
        elif i == lonely[0]:  # just a lonely state
            #Ts.append(np.ones((1,1,1,1), dtype=np.complex128))
            Ts.append(Id1)
            lonely.pop(0)
        else:  # close a singlet
            k = open_singlets.index(i)
            Ts[k] = Close
            next_Ts.pop(k)
            open_singlets.pop(k)
        B = reduce(outer_tensor, Ts)
        Bs.append(B)
        Ts = next_Ts

    Bnpc = [B.transpose(['wL', 'wR', 'p', 'p*']) for B in Bs]
    U_MPO = MPO([site]*L, Bnpc, bc='finite', IdL=0, IdR =-1)
    return U_MPO, Bs

def U_MPO_joint(site, L, pairs, Open, Close, Id1, Id2, lonely = []):
    # sort by smaller site of the pair
    pairs = [((i, j) if i < j else (j, i)) for (i, j) in pairs]

    # sort by starting site
    pairs.sort(key=lambda x: x[0])

    Bs = []
    Ts = [[Id1] for _ in range(L)]  # the tensors on the current site
    for p in pairs:
        Ts[p[0]].append(Open)
        Ts[p[1]].append(Close)
        for i in range(p[0] + 1, p[1]):
            Ts[i].append(Id2)
    Bs = [reduce(outer_tensor, T) for T in Ts]

    Bnpc = [B.transpose(['wL', 'wR', 'p', 'p*']) for B in Bs]
    U_MPO = MPO([site]*L, Bnpc, bc='finite', IdL=0, IdR =-1)
    return U_MPO, Bs

def disjoint_pairs(pairs, L, verbose=False):
    pairs = list(set(pairs))
    num_pairs = len(pairs)
    pairs = [((i, j) if i < j else (j, i)) for (i, j) in pairs]
    pairs.sort(key=lambda x: (x[0], x[1]))
    # Interleave the pairs to try and reduce the number of crossings
    #pairs = pairs[0::2] + pairs[1::2]

    disjoint_pairs = []
    disjoint_sites = []
    while len(pairs) > 0:
        p = pairs.pop(0)
        # Find first set that doesn't contain these sites.
        valid = False
        for i, dd in enumerate(zip(disjoint_pairs, disjoint_sites)):
            dp, ds = dd
            if p[0] not in ds and p[1] not in ds:
                valid=True
                break

        if valid:
            disjoint_pairs[i].append(p)
            disjoint_sites[i].extend((p[0],p[1]))
            disjoint_sites[i] = list(set(disjoint_sites[i]))
        else:
            disjoint_pairs.append([p])
            disjoint_sites.append([p[0],p[1]])
        if verbose:
            print(p, disjoint_pairs, disjoint_sites)

    ds_len = [len(ds) for ds in disjoint_sites]
    dp_num = [len(dp) for dp in disjoint_pairs]
    assert np.sum(dp_num) == num_pairs

    lonely_sites = []
    for ds in disjoint_sites:
        ls = list(set(list(range(L))) - set(ds))
        lonely_sites.append(ls)
    return disjoint_pairs, lonely_sites

def disjoint_pairs_crossing_aware(pairs, L, verbose=False):
    pairs = list(set(pairs))
    num_pairs = len(pairs)
    pairs = [((i, j) if i < j else (j, i)) for (i, j) in pairs]
    pairs.sort(key=lambda x: (x[0], x[1]))
    # Interleave the pairs to try and reduce the number of crossings
    #pairs = pairs[0::2] + pairs[1::2]

    disjoint_pairs = []
    disjoint_sites = []
    while len(pairs) > 0:
        p = pairs.pop(0)
        # Find set that doesn't contain these sites with the minimal number of crossings, if added.
        crossings = np.zeros(len(disjoint_pairs))
        valid = False
        for i, dd in enumerate(zip(disjoint_pairs, disjoint_sites)):
            dp, ds = dd
            if p[0] not in ds and p[1] not in ds:
                crossings[i] = np.sum(count_crossings(dp + [p], L))
                valid = True
            else:
                crossings[i] = np.inf
        if len(disjoint_pairs):
            i = np.argmin(crossings)

        if valid:
            disjoint_pairs[i].append(p)
            disjoint_sites[i].extend((p[0],p[1]))
            disjoint_sites[i] = list(set(disjoint_sites[i]))
        else:
            disjoint_pairs.append([p])
            disjoint_sites.append([p[0],p[1]])
        if verbose:
            print(p, disjoint_pairs, disjoint_sites)

    ds_len = [len(ds) for ds in disjoint_sites]
    dp_num = [len(dp) for dp in disjoint_pairs]
    assert np.sum(dp_num) == num_pairs

    lonely_sites = []
    for ds in disjoint_sites:
        ls = list(set(list(range(L))) - set(ds))
        lonely_sites.append(ls)
    return disjoint_pairs, lonely_sites

def build_MPOs(site, pairs, U, L, crossing_limit=0, crossing_aware=True, verbose=False, disjoint_endpoints=True):
    # disjoint_endpoints=True is for backwards consistency, but I think False should always be used.
    # It should be more efficient.

    pairs = deepcopy(pairs)     # make copy of pairs

    Open, Close, Id1, Id2 = U_machinery(U)
    if disjoint_endpoints:
        # Separate the pairs so that each site is active in at most one gate.
        if crossing_aware:
            # Distribute pairs in a greedy fashion, trying to minimize crossings.
            pairs, lonely_sites = disjoint_pairs_crossing_aware(pairs, L, verbose=verbose)
        else:
            pairs, lonely_sites = disjoint_pairs(pairs, L, verbose=verbose)
    else:
        # We don't separate into disjoint sets; we allow a site to be active in multiple gates,
        # i.e. a gate can both begin and end on a site in a single layer.
        # Thus, a brick wall can be written as single MPO rather than 2.
        # If our gates commute, as we assume they do, this is fine.
        # If we have non-commuting gates, we should make multiple class fo build_MPOs, one for each set of
        # pairs for the non-commuting gates.
        # Honestly, not sure when we'd use the above disjoint_endpoints=True.
        active_sites = []
        for p in pairs:
            active_sites.extend(p)
        lonely_sites = list(set(list(range(L))) - set(active_sites))
        pairs = [pairs]
        lonely_sites = [lonely_sites]
        
    # What are breaks? It tell us where the MPOs for the next set of disjoint pairs begins.
    # So if we have 3 disjoint pairs from above, breaks will be [0,1], meaning that after MPO0 and MPO1,
    # we move to different sets.
    # At the moment, breaks will be after each MPO (supposing disjoint_endpoints=True), as we haven't worried
    # about crossings. We simply separate the pairs into sets where each site is active in at most 1 gate.
    breaks = list(range(len(pairs) - 1))    # Empty if only 1 list of pairs
    print("Original breaks and crossings:", breaks, [int(np.max(count_crossings(p, L))) for p in pairs])
    if crossing_limit > 0:
        # We have separated the pairs into disjoint sets, but we have paid little 
        # attention to how many of the pairs cross one another, which leads to
        # increased bond dimension. To fix this, we separate a set of pairs into
        # multiple sets by fixing the max number of crossing.
        new_pairs, new_lonely = [], []
        for i, pls in enumerate(zip(pairs, lonely_sites)):
            p, ls = pls
            pn, nl = separate_pairs(p, ls, L, crossing_limit=crossing_limit)
            new_pairs.extend(pn)
            new_lonely.extend(nl)
            if len(breaks) and i < len(pairs) - 1:
                # Now, breaks tells us when an MPO represents a subset of pairs from a set.
                # So if we wanted to insert single qubit gates between non-commuting layers,
                # breaks gives us the information necessary.
                breaks[i] = len(new_pairs) - 1
        print("Old # pairs: %d, New # pairs: %d" % (len(pairs), len(new_pairs)))
        pairs = new_pairs
        lonely_sites = new_lonely
    print("Modified breaks:", breaks, flush=True)
    
    MPOs = []
    for dp, ls in zip(pairs, lonely_sites):
        if disjoint_endpoints:
            MPOs.append(U_MPO_disjoint(site, L, dp, Open, Close, Id1, Id2, lonely = ls)[0])
        else:
            # Lonely isn't actually used here.
            MPOs.append(U_MPO_joint(site, L, dp, Open, Close, Id1, Id2, lonely = ls)[0])
    return MPOs, breaks, pairs, lonely_sites

def count_crossings(pairs, L):
    crossings = np.zeros(L-1)
    for pair in pairs:
        for i in range(pair[0], pair[1]):
            crossings[i] += 1
    return crossings

def separate_pairs(pairs, lonely, L, crossing_limit=1):
    pl = len(pairs)
    new_pairs = []
    new_lonely = []
    while len(pairs) > 0:
        current_pairs = []
        for p in pairs:
            crossing = count_crossings(current_pairs + [p], L)
            if np.all(crossing <= crossing_limit):
                current_pairs.append(p)

        for p in current_pairs:
            pairs.remove(p)
        nl = lonely + list(sum(pairs, ())) #https://stackoverflow.com/questions/10632839/transform-list-of-tuples-into-a-flat-list-or-a-matrix
        for pn in new_pairs:
            nl.extend(list(sum(pn, ())))
        nl.sort()
        new_pairs.append(current_pairs)
        new_lonely.append(nl)
    assert np.sum([len(pn) for pn in new_pairs]) == pl
    return new_pairs, new_lonely
