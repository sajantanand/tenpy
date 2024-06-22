import numpy as np
from copy import deepcopy
import time

from ..models.lattice import Chain, TrivialLattice
from ..models.model import MPOModel, NearestNeighborModel
from ..networks.mps import MPSEnvironment, MPS
from ..networks.doubled_mps import DoubledMPS
from ..networks.site import DoubledSite
from ..linalg import np_conserved as npc
from ..linalg.charges import LegPipe
from ..algorithms.truncation import svd_theta, TruncationError, _machine_prec_trunc_par
import warnings

def double_model(H_MPO, NN=False, doubled=False, conjugate=False, hermitian=True):
    """
    args:
        NN: Boolean
            Whether the model only contains NN terms and thus should be made into a `NearestNeighborModel` for TEBD
        doubled: Boolean
            Whether the model is ALREADY in doubled Hilbert space
        conjugate: Boolean
            Do we conjugate the MPO by change of basis matrices; typically one moves to Hermitian or charge-conserving basis
        hermitian: Boolean
            Is the basis Hermitian or charge conserving?
    returns:
        doubled_model: TeNPy model
            Either `MPOModel` or `NearestNeighborModel` depending on the `NN` parameter
    """
    if not doubled:
        doubled_MPO = H_MPO.make_doubled_MPO(hermitian)
    else:
        doubled_MPO = deepcopy(H_MPO)

    if conjugate:
        doubled_MPO.conjugate_MPO([s.s2d for s in doubled_MPO.sites])

    #doubled_lat = lat(H_MPO.L, doubled_MPO.sites[0]) # SAJANT - what if we have different types of sites in the lattice?
    doubled_lat = TrivialLattice(doubled_MPO.sites) # Trivial lattice uses the sites of the doubled MPO.
    # What if we don't want a chain lattice?

    doubled_model = MPOModel(doubled_lat, doubled_MPO)
    if NN:
        doubled_model = NearestNeighborModel.from_MPOModel(doubled_model)

    return doubled_model

def generate_pairs(lat, key='nearest_neighbors'):
    """
    Generate pairs of couplings for 2D lattice with k-local conservation

    args:
        lat: TenPy Lattice
            What lattice (square, chain, Kagome, etc.) is used for simulation
        key: str
            Which pairs are we interested in preserving

    returns:
        pairs: list
            list of couplings we wish to preserve with DMT
    """
    idXs, idYs = [], []
    for dx1, dx2, bv in lat.pairs[key]:
        idX, idY = lat.possible_couplings(dx1,dx2,bv)[:2]
        idXs.append(idX)
        idYs.append(idY)
    pairs = [sorted((a,b)) for a,b in zip(np.concatenate(idXs), np.concatenate(idYs))]
    return pairs

def distribute_pairs(pairs, bi, symmetric=True):
    """
    Distribute pairs on bond `bi` to left and right lists so we know which operators to preserve in which density matrix

    args:
        pairs: list
            output from `generate_pairs`
        bi: int
            which bond we are currently truncating
        symmetric: Boolean
            Are bonds included in both left and right lists? `symmetric=True` requires larger bond dimension but is what is done in original DMT

    returns:
        left_points: list
            on which sites to the left of the cut should we preserve operators
        right_points: list
    """
    cut_pairs = [a for a in pairs if a[0] <= bi and a[1] > bi]
    left_points, right_points =[], []
    if symmetric:
        left_points = list(set([cp[0] for cp in cut_pairs]))
        right_points = list(set([cp[1] for cp in cut_pairs]))
    else:
        for cp in cut_pairs:
            if cp[0] not in left_points and cp[1] not in right_points:
                # add to preserve list
                if len(left_points) < len(right_points):
                    left_points.append(cp[0])
                else:
                    right_points.append(cp[1])
    return left_points, right_points

def trace_identity_DMPS(DMPS, traceful_id=None):
    """
    Bra for density matrix expectation value; we simply represent the identity matrix.
    We are in the computational / standard basis. Make doubled MPS with two physical legs
    per site, using the identity on each site
    """
    assert traceful_id == None, "Not used here since the doubled MPS has both bra and ket legs"
    Is = [np.eye(ds.dim).reshape(ds.dim, ds.dim, 1, 1) for ds in DMPS.sites]
    return DoubledMPS.from_Bflat(DMPS.sites,
                                 Is,
                                 SVs=None,
                                 bc='finite',
                                 dtype=None,
                                 permute=True, # The site is typically permutted, so we NEED this to be true.
                                 form='B', # Form doesn't matter since it's a product state?
                                 legL=None)

def make_swap_gate(d):
    swap = np.zeros((d, d))
    d2 = int(np.sqrt(d)) # Local Hilbert space dimension of the operator
    for i in range(d):
        for j in range(d):
            a, b = i // d2, i % d2
            e, f = j // d2, j % d2
            if a == f and b == e:
                swap[i,j] = 1
    return swap

def trace_swap_DMPS(DMPS, traceful_id=None):
    """
    Bra for quadrupled space OTOCs; the matrix represents a swap operator in the physical
    space. Given a quadupled DMPS of local Hilbert space dimension :math:`d`, we want the swap
    in this space that represents two operators.
    """
    assert traceful_id == None, "Not used here since the doubled MPS has both bra and ket legs"
    Swaps = []
    for ds in DMPS.sites:
        swap = make_swap_gate(ds.dim).reshape(ds.dim, ds.dim, 1, 1)
        Swaps.append(swap)

    return DoubledMPS.from_Bflat(DMPS.sites,
                                 Swaps,
                                 SVs=None,
                                 bc='finite',
                                 dtype=None,
                                 permute=True, # The site is typically permutted, so we NEED this to be true.
                                 form='B', # Form doesn't matter since it's a product state?
                                 legL=None)

def trace_identity_MPS(DMPS):#, traceful_id=None):
    """
    Bra for density matrix expectation value once flattened. We represent the identity matrix as a vector.
    This is in the rotated basis, so we only have 1 index representing the identity operator.

    We pass in a flattened MPS.
    """
    # For grouped sites, this is no longer true!
    #for ds in DMPS.sites:
    #    assert type(ds) is DoubledSite

    Is = []
    for ds in DMPS.sites:
        I = np.zeros((ds.dim,1,1))
        """
        try:
            ch_ind = list(np.argsort(ds.charges))
            ti = [ch_ind.index(i) for i in ds.traceful_ind]
        except AttributeError as e:
            # No charges!
            ti = ds.traceful_ind
        """
        ti = 0 #ds.traceful_ind
        I[ti,0,0] = 1 # 1 for every operator with trace 1.
        Is.append(I)
    return MPS.from_Bflat(DMPS.sites,
                          Is,
                          SVs=None,
                          bc='finite',
                          dtype=None,
                          permute=True, # The site is typically permutted, so we NEED this to be true.
                          form='B', # Form doesn't matter since it's a product state?
                          legL=None)

def trace_swap_MPS(DMPS):#, traceful_id=None):
    """
    Bra for density matrix expectation value once flattened. We represent the swap operator as a vector.
    To get this, we first get the swap matrix, flatten it into an MPS tensor, build an MPS, and then
    rotate the MPS into the desired basis.
    """
    # For grouped sites, this is no longer true!
    #for ds in DMPS.sites:
    #    assert type(ds) is DoubledSite

    Swaps = []
    for ds in DMPS.sites:
        d = ds.dim
        d2 = int(np.sqrt(d))
        swap_matrix = make_swap_gate(d2)
        swap = swap_matrix.reshape(d, 1, 1)
        Swaps.append(swap)
    swap_MPS = MPS.from_Bflat(DMPS.sites,
                          Swaps,
                          SVs=None,
                          bc='finite',
                          dtype=None,
                          permute=True, # The site is typically permutted, so we NEED this to be true.
                          form='B', # Form doesn't matter since it's a product state?
                          legL=None)

    # Now we need to rotate this to the desired basis.
    swap_MPS.apply_product_op([s.s2d for s in swap_MPS.sites], unitary=True)
    return swap_MPS

"""
In principle we could combine all DMT methods into just the MPO method and use a different MPO for EACH operator we wish to preserve.
So at bond `i` for local `(1,1)` DMT, we would need 7 MPOs for the operators (II, IX, IY, IZ, XI, YI, ZI). Redundancy removal would return
this to 4 operators on each side. The issue is that we cannot reused contracted environments between sites since local operators would require
new MPOs on each site. This is not desirable. So while we can think about this for pedagongical purposes, we will not implement the code
this way.

Instead, we should put the MPO to preserve INTO the state $\rho$ we trace against (typically the infintie temperature density matrix).
"""

def build_QR_matrix_R(dMPS, i, dmt_params, trace_env, MPO_envs):
    """
    Construct change of basis matrices for the right Hilbert space on bond `i`. There are three types of DMT that can be combined:
    (1) Local DMT - preserve the tensor product of operators a desired `radius` from the truncated bond; radius can be different on left and right
    (2) MPO DMT - preserve (sum of) operators specifieid by an MPO
    (3) Conjoined DMT - prserve the direct sum of operators on specific sites given by physical, local connections on lattice

    args:
        dMPS: MPS (flattened to have just 1 physical leg)
            The density matrix / operator that is to be truncated
        i: int
            Bond on which we truncate; bond `i` is to the right of site `i`
        dmt_params: dictionary
            collects all paramters for local and conjoined DMT
        trace_env: TeNPy MPOEnvironment
            Used to getting operators when taking trace against a state (presumably the identity, but this can be generalized to any state)
        MPO_envs: list of TeNPy MPOEnvironments
            Each MPO is SEPARATELY conserved

    returns:
        QR_R: npc Array
            Matrix that defines the change of basis matrix once a QR is done
        keep_R: int
            How many independent sets of operators are preserved
        trace_env: MPOEnvironment
            Updated environment (what is stored has changed) for trace with respect to a particular state
        MPO_envs: MPOEnvironments
            Updated MPO environments
    """
    #############################################
    # MYABE - restructure DMT to only be phrased in terms of MPOs to preserve; then all types of DMT can be viewed as the exact same.
    #############################################

    # Bond i between sites i and i+1; truncating bond i
    QR_Rs = []
    keep_R = 0
    local_params = dmt_params.get('k_local_params', None)
    conjoined_params = dmt_params.get('conjoined_params', None)
    if i == dMPS.L-1: # Bond to the right of last site; do nothing
        return trace_env.get_RP(i).replace_label('vL*', 'p'), 1, trace_env, MPO_envs
    if local_params is not None:
        k_local = local_params.get('k_local') # How many sites to include on either side of cut
        end_R = np.min([i+1+np.max([k_local[1], 1]), dMPS.L]) # do not include endpoint

        # Accounts for non-uniform local Hilbert space
        keep_R += np.prod([1] + [dMPS.dim[k] for k in range(i+1, end_R)]) if k_local[1] > 0 else 1

        if k_local[1] == 0:
            QR_R = trace_env.get_RP(i, store=True).replace_label('vL*', 'p')
            # vL* points out
        else:
            # Need to make each p leg of Bs point out before contracting.
            # TODO won't work if taking trace against non-trivial rho
            QR_R = trace_env.get_RP(end_R-1, store=True).add_trivial_leg(0, 'p', qconj=-1).squeeze('vL*') # strictly right of end_R-1
            for j in reversed(range(i+1, end_R)):
                B = dMPS.get_B(j, form='B').replace_label('p', 'p1')
                B = B.gauge_total_charge('p1', new_qconj=B.get_leg('p1').qconj*-1) # +1 -> -1
                QR_R = npc.tensordot(B, QR_R, axes=['vR', 'vL'])  # axes_p + vL
                QR_R = QR_R.combine_legs(['p', 'p1'], qconj=QR_R.get_leg('p').qconj).ireplace_label('(p.p1)', 'p')
        QR_R.itranspose(['vL', 'p'])
        QR_Rs.append(QR_R)

    if MPO_envs is not None:
        for Me in MPO_envs:
            # Need to flip the leg of the W wL tensor, as this becomes the p leg of QR_L
            QR_R = Me.get_RP(i+1, store=True) # vL (ket), wL, vL* (bra)
            B = dMPS.get_B(i+1, form='B')
            QR_R = npc.tensordot(B, QR_R, axes=(['vR'], ['vL'])) #  vL (ket, including site i+1), p, wL, vL* (bra)
            W = Me.H.get_W(i+1)
            W = W.gauge_total_charge('wL', new_qconj=W.get_leg('wL').qconj*-1) # +1 -> -1
            QR_R = npc.tensordot(W, QR_R, axes=(['wR', 'p*'], ['wL', 'p'])) #  vL (ket, including site i+1), p (from MPO on site i+1), wL (inlcuding site i+1), vL* (bra)
            #trace_tensor = trace_identity_MPS(dMPS).get_B(i+1)
            trace_tensor = trace_env.bra.get_B(i+1)
            QR_R = npc.tensordot(QR_R, trace_tensor.conj(), axes=(['p', 'vL*'],['p*', 'vR*'])) # vL (ket), wL , vL* (bra); all including site i+1
            QR_R = QR_R.combine_legs(['vL*', 'wL'], qconj=-1).replace_label('(vL*.wL)', 'p').transpose(['vL', 'p'])
            QR_Rs.append(QR_R)
            keep_R += QR_R.shape[QR_R.get_leg_index('p')]

    if conjoined_params is not None:
        pairs = conjoined_params.get('pairs')
        symmetric = conjoined_params.get('symmetric', True)
        _, right_pairs = distribute_pairs(pairs, i, symmetric=symmetric)

        # We want to keep the direct sum of the operator Hilbert spaces on each site; we don't want to overcount the Identity, so there are 3 non-trivial operators per site.
        #keep_R += int(np.sum([dMPS.dim[k]-1 for k in right_pairs])) + 1
        keep_R += int(np.sum([dMPS.dim[k] for k in right_pairs]))

        if len(right_pairs) == 0:
            # If we are at edge of chain and have no right sites assigned, preserve the identity operator.
            QR_R = trace_env.get_RP(i, store=True).replace_label('vL*', 'p')
            QR_Rs.append(QR_R)
            keep_R += 1
        # TODO won't work if taking trace against non-trivial rho
        for rp in right_pairs:
            QR_R = trace_env.get_RP(rp, store=True) #.squeeze('vL*')
            B = dMPS.get_B(rp, form='B')
            B = B.gauge_total_charge('p', new_qconj=B.get_leg('p').qconj*-1) # +1 -> -1
            QR_R = npc.tensordot(B, QR_R, axes=['vR', 'vL'])  # axes_p + vL + vL* (technically vL* is from site rp+1, but it doesn't matter since it's trivial)

            for j in reversed(range(i+1, rp)):
                TT = trace_env.bra.get_B(j) # Tensor trace; the identity element used for tracing over a site;
                B = npc.tensordot(TT.conj(), dMPS.get_B(j, form='B'), axes=(['p*'], ['p'])) # vL*, vR*, vL, VR # Trace over this site
                QR_R = npc.tensordot(B, QR_R, axes=(['vR*', 'vR'], (['vL*', 'vL'])))  # axes_p + (vR*, vR)
            QR_R = QR_R.squeeze('vL*').transpose(['vL', 'p'])
            QR_Rs.append(QR_R)

    for QR_R in QR_Rs:
        if npc.norm(QR_R.unary_blockwise(np.imag)) < dmt_params.get('imaginary_cutoff', 1.e-12):
            QR_R.iunary_blockwise(np.real)
        p_index = QR_R.get_leg_index('p')
        if isinstance(QR_R.get_leg('p'), LegPipe):
            # Why? Maybe this reduces overhead if the pipe is very deep?
            QR_R.legs[p_index] = QR_R.get_leg('p').to_LegCharge()
    # QR_R: labels=('vL', 'p), qconj=(+1, -1)
    #print('QR Legs:', [(qr_R, qr_R.legs) for qr_R in QR_Rs])
    QR_R = npc.concatenate(QR_Rs, axis='p')
    assert QR_R.shape[QR_R.get_leg_index('p')] == keep_R
    return QR_R, keep_R, trace_env, MPO_envs

def build_QR_matrix_L(dMPS, i, dmt_params, trace_env, MPO_envs):
    """
    See documentation for build_QR_matrix_R
    """

    # Bond i between sites i and i+1; truncating bond i
    QR_Ls = []
    keep_L = 0
    local_params = dmt_params.get('k_local_params', None)
    conjoined_params = dmt_params.get('conjoined_params', None)
    if i == -1: # Bond to the left of first site; do nothing
        return trace_env.get_LP(0).replace_label('vR*', 'p'), 1, trace_env, MPO_envs
    if local_params is not None:
        k_local = local_params.get('k_local') # How many sites to include on either side of cut
        start_L = np.max([i+1-np.max([k_local[0], 1]), 0]) # include endpoint

        # Accounts for non-uniform local Hilbert space
        keep_L += np.prod([1] + [dMPS.dim[k] for k in range(start_L, i+1)]) if k_local[0] > 0 else 1

        if k_local[0] == 0:
            QR_L = trace_env.get_LP(i+1, store=True).replace_label('vR*', 'p')
            # vR* points in
        else:
            # TODO won't work if taking trace against non-trivial rho
            QR_L = trace_env.get_LP(start_L, store=True).add_trivial_leg(0, 'p', qconj=+1).squeeze('vR*') # strictly left of start_L
            for j in range(start_L, i+1):
                A = dMPS.get_B(j, form='A').replace_label('p', 'p1')
                QR_L = npc.tensordot(QR_L, A, axes=['vR', 'vL'])  # axes_p + vL
                QR_L = QR_L.combine_legs(['p', 'p1'], qconj=QR_L.get_leg('p').qconj).ireplace_label('(p.p1)', 'p')
        QR_L.itranspose(['p','vR'])
        QR_Ls.append(QR_L)

    if MPO_envs is not None:
        for Me in MPO_envs:
            # Need to flip the leg of the W wR tensor, as this becomes the p leg of QR_L
            QR_L = Me.get_LP(i, store=True) # vR (ket), wR, vR* (bra)
            A = dMPS.get_B(i, form='A')
            QR_L = npc.tensordot(QR_L, A, axes=(['vR'], ['vL'])) #  vR (ket, including site i), p, wR, vR* (bra)
            W = Me.H.get_W(i)
            W = W.gauge_total_charge('wR', new_qconj=W.get_leg('wR').qconj*-1) # -1 -> 1
            QR_L = npc.tensordot(QR_L, W, axes=(['wR', 'p'], ['wL', 'p*'])) # vR (ket, including site i), p (from MPO on site i), wR (including site i), vR* (bra)
            #trace_tensor = trace_identity_MPS(dMPS).get_B(i)
            trace_tensor = trace_env.bra.get_B(i)
            QR_L = npc.tensordot(trace_tensor.conj(), QR_L, axes=(['p*', 'vL*'], ['p', 'vR*'])) # vR (ket), wR , vR* (bra); all including site i
            QR_L = QR_L.combine_legs(['vR*', 'wR'], qconj=+1).replace_label('(vR*.wR)', 'p').transpose(['p', 'vR'])
            QR_Ls.append(QR_L)
            keep_L += QR_L.shape[QR_L.get_leg_index('p')]

    if conjoined_params is not None:
        pairs = conjoined_params.get('pairs')
        symmetric = conjoined_params.get('symmetric', True)
        left_pairs, _ = distribute_pairs(pairs, i, symmetric=symmetric)

        # We want to keep the direct sum of the operator Hilbert spaces on each site; we don't want to overcount the Identity, so there are 3 non-trivial operators per site.
        #keep_R += int(np.sum([dMPS.dim[k]-1 for k in left_pairs]))
        keep_L += int(np.sum([dMPS.dim[k] for k in left_pairs]))

        if len(left_pairs) == 0:
            # If we are at edge of chain and have no left sites assigned, preserve the identity operator.
            QR_L = trace_env.get_LP(i+1, store=True).replace_label('vR*', 'p')
            QR_Ls.append(QR_L)
            keep_L += 1
        # TODO won't work if taking trace against non-trivial rho
        for lp in left_pairs:
            QR_L = trace_env.get_LP(lp, store=True) #.squeeze('vR*')
            A = dMPS.get_B(lp, form='A')
            QR_L = npc.tensordot(QR_L, A, axes=['vR', 'vL'])  # axes_p + vR + vR* (technically vR* is from site sp-1, but it doesn't matter since it's trivial)

            for j in range(lp+1, i+1):
                TT = trace_env.bra.get_B(j) # Tensor trace; the identity element used for tracing over a site;
                A = npc.tensordot(TT.conj(), dMPS.get_B(j, form='A'), axes=(['p*'], ['p'])) # vL*, vR*, vL, VR # Trace over this site
                QR_L = npc.tensordot(QR_L, A, axes=(['vR*', 'vR'], (['vL*', 'vL'])))  # axes_p + (vR*, vR)
            QR_L = QR_L.squeeze('vR*').transpose(['p', 'vR'])
            QR_Ls.append(QR_L)

    for QR_L in QR_Ls:
        if npc.norm(QR_L.unary_blockwise(np.imag)) < dmt_params.get('imaginary_cutoff', 1.e-12):
            QR_L.iunary_blockwise(np.real)
        p_index = QR_L.get_leg_index('p')
        if isinstance(QR_L.get_leg('p'), LegPipe):
            # Why? Maybe this reduces overhead if the pipe is very deep?
            QR_L.legs[p_index] = QR_L.get_leg('p').to_LegCharge()
    # QR_L: labels=('p', 'vR), qconj=(+1, -1)
    #print('QL Legs:', [(qr_L, qr_L.legs) for qr_L in QR_Ls])
    QR_L = npc.concatenate(QR_Ls, axis='p')
    assert QR_L.shape[QR_L.get_leg_index('p')] == keep_L
    return QR_L, keep_L, trace_env, MPO_envs

def build_QR_matrices(dMPS, i, dmt_params, trace_env, MPO_envs):
    """
    Actually call the functions above; see funcations above for documentation.
    """

    QR_L, keep_L, trace_env, MPO_envs = build_QR_matrix_L(dMPS, i, dmt_params, trace_env, MPO_envs)
    QR_R, keep_R, trace_env, MPO_envs = build_QR_matrix_R(dMPS, i, dmt_params, trace_env, MPO_envs)

    return QR_L, QR_R, keep_L, keep_R, trace_env, MPO_envs

def remove_redundancy_QR(QR_L, QR_R, keep_L, keep_R, R_cutoff):
    """
    We may have redundant copies of operators to preserve; most commonly, we will have several copies of the identity.
    Here we remove them by doing a QR and seeing what is unneeded (i.e. zero diagonals)

    args:
        QR_L: npc Array
            matrix defining change of basis on the left Hilbert space
        QR_R: npc Array
            matrix defining change of basis on the right Hilbert space
        keep_L, keep_R: int, int
            Number of independent operator combinations to preserve on left and right
        R_cutoff: float
            What do we consider 0 when removing redundancy

    returns:
        Q_L: npc Array
            Rotation matrix for left Hilbert space
        Q_R: npc Array
            Rotation matrix for right Hilbert space
        keep_L, keep_R: int, int
            Number of independent operator combinations to preserve on left and right after redundancy removed
    """
    def get_indices(R, cutoff):
        """
        Determine which rows of R have norm < cutoff. These are the columns of Q that we can eventually
        truncate once we get the M matrix.
        """
        indices = np.zeros(R.legs[0].ind_len, dtype=bool)
        for s, d in zip(R._qdata, [np.linalg.norm(d, axis=1) > cutoff for d in R._data]):
            # indices is True for rows that have norm > cutoff
            indices[R.legs[0].slices[s[0]]:R.legs[0].slices[s[0]+1]] = d
        return np.logical_not(indices), np.sum(indices)

    Q_L, R_L = npc.qr(QR_L.itranspose(['vR', 'p']),
                          mode='complete',
                          inner_labels=['vR*', 'vR'],
                          #cutoff=1.e-12, # Need this to be none to have square Q
                          pos_diag_R=True,
                          qtotal_Q=QR_L.qtotal,
                          inner_qconj=QR_L.get_leg('vR').conj().qconj)
    Q_R, R_R = npc.qr(QR_R.itranspose(['vL', 'p']),
                          mode='complete',
                          inner_labels=['vL*', 'vL'],
                          #cutoff=1.e-12, # Need this to be none to have square Q
                          pos_diag_R=True,
                          qtotal_Q=QR_R.qtotal,
                          inner_qconj=QR_R.get_leg('vL').conj().qconj)
    """
    If any of the diagonal elements of R_L/R are zero, we get the warning:
    WARNING : /global/common/software/m3859/tenpy_sajant/tenpy/linalg/np_conserved.py:3993: RuntimeWarning: invalid value encountered in true_divide
    phase = r_diag / np.abs(r_diag)
    """
    # SAJANT TODO - Do I need to worry about charge of Q and R? Q is chargeless by default, but I put the charge into Q

    proj_L, new_keep_L = get_indices(R_L, R_cutoff)
    proj_R, new_keep_R = get_indices(R_R, R_cutoff)

    return Q_L, R_L, Q_R, R_R, new_keep_L, new_keep_R, proj_L, proj_R

def remove_redundancy_SVD(QR_L, QR_R, keep_L, keep_R, svd_cutoff=1.e-14):
    """
    We may have redundant copies of operators to preserve; most commonly, we will have several copies of the identity.
    Here we remove them by doing a QR and seeing what is unneeded (i.e. zero diagonals)


    args:
        QR_L: npc Array
            matrix defining change of basis on the left Hilbert space
        QR_R: npc Array
            matrix defining change of basis on the right Hilbert space
        keep_L, keep_R: int, int
            Number of independent operator combinations to preserve on left and right
        R_cutoff: float
            What do we consider 0 when removing redundancy

    returns:
        Q_L: npc Array
            Rotation matrix for left Hilbert space
        Q_R: npc Array
            Rotation matrix for right Hilbert space
        keep_L, keep_R: int, int
            Number of independent operator combinations to preserve on left and right after redundancy removed
    """
    raise NotImplementedError("Use the QR version; it is faster.")

    # I use the labels Q and R even though we are doing SVD)
     #tenpy.linalg.np_conserved.svd(a, full_matrices=False, compute_uv=True, cutoff=None, qtotal_LR=[None, None], inner_labels=[None, None], inner_qconj=1)[source]
    QR_L.itranspose(['vR', 'p'])
    QR_R.itranspose(['vL', 'p'])
    if svd_cutoff > 0.0:
        Q_L, s_L, R_L = npc.svd(QR_L,
                              full_matrices=True,
                              cutoff=None,
                              inner_labels=['vR*', 'vR'],
                                qtotal_LR=[QR_L.qtotal, None],
                              inner_qconj=QR_L.get_leg('vR').conj().qconj)
        projs_L = s_L > svd_cutoff
        if np.any(projs_L == False):
            p_index = QR_L.get_leg_index('p')
            if isinstance(QR_L.legs[p_index], LegPipe):
                QR_L.legs[p_index] = QR_L.legs[p_index].to_LegCharge()
            QR_L.iproject(projs_L, 'p')
            keep_L = np.sum(projs_L)

        Q_R, s_R, R_R = npc.svd(QR_R,
                              full_matrices=True,
                              cutoff=None,
                              inner_labels=['vL*', 'vL'],
                                qtotal_LR=[QR_R.qtotal, None],
                              inner_qconj=QR_R.get_leg('vL').conj().qconj)
        projs_R = s_R > svd_cutoff
        if np.any(projs_R == False):
            p_index = QR_R.get_leg_index('p')
            if isinstance(QR_R.legs[p_index], LegPipe):
                QR_R.legs[p_index] = QR_R.legs[p_index].to_LegCharge()
            QR_R.iproject(projs_R, 'p')
            keep_R = np.sum(projs_R)

    Q_L, R_L = npc.qr(QR_L,
                      mode='complete',
                      inner_labels=['vR*', 'vR'],
                      #cutoff=1.e-12, # Need this to be none to have square Q
                      pos_diag_R=True,
                      qtotal_Q=QR_L.qtotal,
                      inner_qconj=QR_L.get_leg('vR').conj().qconj)

    Q_R, R_R = npc.qr(QR_R,
                      mode='complete',
                      inner_labels=['vL*', 'vL'],
                      #cutoff=1.e-12, # Need this to be none to have square Q
                      pos_diag_R=True,
                      qtotal_Q=QR_R.qtotal,
                      inner_qconj=QR_R.get_leg('vL').conj().qconj)

    # We want the rows that are zero; these define the "lower-right" block.
    np_R_L = R_L.to_ndarray()
    proj_L = np.logical_not(np.any(np_R_L, axis=1))

    np_R_R = R_R.to_ndarray()
    proj_R = np.logical_not(np.any(np_R_R, axis=1))

    #assert keep_L == R_L.get_leg('vR').ind_len - np.sum(proj_L)
    #assert keep_R == R_R.get_leg('vL').ind_len - np.sum(proj_R)
    return Q_L, R_L, Q_R, R_R, keep_L, keep_R, proj_L, proj_R

def truncate_M(M, svd_trunc_params, connected, keep_L, keep_R, proj_L, proj_R, traceful_ind_L=0, traceful_ind_R=0):
    """
    Truncate the lower right block once we've moved to desired basis

    args:
        M: npc Array
            Bond tensor after moving into desired basis so that we can truncate lower right block
        svd_trunc_params: dictionary
            Standard TeNPy truncation parameters
        connected: Boolean
            Do we perform the operation in Eq. 25 of https://arxiv.org/pdf/1707.01506.pdf?
        keep_L, keep_R: int, int
            Number of independent operator combinations to preserve; needed to  extract lower right block
        proj_L, proj_R: list (bools), list (bools)
            which indices of M do we keep for the "lower right" block; length of each arguement is the dimension of M
    """
    # Connected component - traceful_ind_L/R tell us which entry of M corresponds to the identity operator
    # With MPOs, we can't really use `connected=True` since there will not be a single operator corresponding to the Identity.
    if connected:
        orig_M = M.copy()
        if np.isclose(orig_M[traceful_ind_L,traceful_ind_R], 0.0): # traceless op
            print("Tried 'connected=True' on traceless operator; you sure about this?")
            assert False
        else:
            M = orig_M - npc.outer(orig_M.take_slice([traceful_ind_R], ['vR']),
                                   orig_M.take_slice([traceful_ind_L], ['vL'])) / orig_M[traceful_ind_L, traceful_ind_R]

    M_DR = M[proj_L, proj_R]

    # Do SVD of M_prime block, truncating according to svd_trunc_par
    # We DO NOT normalize the SVs of M_DR before truncating, so the svd_trunc_par['svd_min'] is the
    # cutoff used to discard singular values. The error returned by svd_theta is the total weight of
    # discard SVs**2, divided by the original norm squared of all of the SVs.
    # This is the error in SVD truncation, given a non-normalized initial state.
    svd_trunc_params = deepcopy(svd_trunc_params)
    svd_trunc_params['chi_max'] = np.max([0, svd_trunc_params['chi_max'] - keep_L - keep_R + connected])
    try:
        UM, sM, VM, err, renormalization = svd_theta(M_DR, svd_trunc_params, renormalize=False)
        # All we want to do here is reduce the rank of M_DR
        if svd_trunc_params.get('chi_max', None) == 0:
            # Only keep the part necessary to preserve local properties; not sure when we would want to do this.
            # So we kill the entire lower right block
            sM *= 0
        # SAJANT - do something with err so that truncation error is recorded?

        # Lower right block; norm is `renormalization`
        M_DR_trunc = npc.tensordot(UM, VM.scale_axis(sM, axis='vL'), axes=(['vR', 'vL']))
    except RuntimeError as e:
        # Lower right block has norm 0, so we just remove it.
        print(e)
        assert np.isclose(npc.norm(M_DR), 0.0)
        M_DR_trunc = M_DR * 0
        err = TruncationError()
    M[proj_L, proj_R] = M_DR_trunc

    if connected:
        M = M + npc.outer(orig_M.take_slice([traceful_ind_R], ['vR']),
                          orig_M.take_slice([traceful_ind_L], ['vL'])) / orig_M[traceful_ind_L, traceful_ind_R]
    return M, err

def dmt_theta(dMPS, i, svd_trunc_params, dmt_params,
              trace_env, MPO_envs,
              svd_trunc_params_2=_machine_prec_trunc_par,
              timing=False):
    """
    Performs Density Matrix Truncation (DMT) on an MPS representing a density matrix or operator.
    We truncate on the bond between site i to the left and i+1 to the right. This however requires
    the entire state, as we use non-local properties to do the truncation.

    The DMT algorithm was propsed in https://arxiv.org/abs/1707.01506 and extended to 2D and
    long range interactions in https://arxiv.org/abs/2312.XXXXX; LOL; I should've known better.

    See documentation of svd_theta, which we follow.

    We assume that the orthogonality center is already on bond i of the MPS. So tensor i should be in left ('A')
    form while tensor i+1 should be in right ('B') form. TEBD moves the OC by applying gates, but if we want to
    only truncate a dMPS using this function, we need to explicitly move the OC.
    """
    if timing:
        time1 = time.time()
    # Check that the MPS has the proper form; need A to the left of (i,i+1) and B to the right
    assert dMPS.form[:i] == [(1.0, 0.0) for _ in range(i)], dMPS.form[:i]
    assert dMPS.form[i+2:] == [(0.0, 1.0) for _ in range(dMPS.L - i - 2)], dMPS.form[i+2:]

    # Make trace_env if none
    # Used to speed up identity contractions as we can reuse environments
    if trace_env is None:
        trace_env = MPSEnvironment(trace_identity_MPS(dMPS), dMPS)
    elif trace_env.ket is not dMPS:
        raise ValueError("Ket in 'trace_env' is not the current doubled MPS.")

    # We want to remove the environments containing the bond i
    trace_env.del_RP(i)
    trace_env.del_LP(i+1)
    if MPO_envs is not None:
        for j, Me in enumerate(MPO_envs):
            if Me.ket is not dMPS:
                raise ValueError(f"Ket in 'MPO ENV' {j} is not the current doubled MPS.")
            Me.del_RP(i)
            Me.del_LP(i+1)
    else:
        # MPO preservation doesn't work with connected correlations
        assert dmt_params.get('connected', False) == False

    S = dMPS.get_SR(i) # singular values to the right of site i
    chi = len(S)
    if chi == 1: # Cannot do any truncation, so give up.
        svd_trunc_params.touch('chi_max')
        svd_trunc_params.touch('svd_min')
        return TruncationError(), 1, trace_env, MPO_envs

    QR_L, QR_R, keep_L, keep_R, trace_env, MPO_envs = build_QR_matrices(dMPS, i, dmt_params, trace_env, MPO_envs)
    if timing:
        time2 = time.time()
        print('Get QR Time:', time2 - time1, flush=True)
        time1 = time2
    # Always need to call this function, as it performs the QR; remove redundancy if R_cutoff > 0.0
    Q_L, _, Q_R, _, keep_L, keep_R, proj_L, proj_R = remove_redundancy_QR(QR_L, QR_R, keep_L, keep_R, dmt_params.get('R_cutoff', 1.e-18))#1.e-14))
    if timing:
        time2 = time.time()
        print('Remove redundancy Time:', time2 - time1, flush=True)
        time1 = time2
    #Q_L, _, Q_R, _, keep_L, keep_R, proj_L, proj_R = remove_redundancy_SVD(QR_L, QR_R, keep_L, keep_R, dmt_params.get('R_cutoff', 1.e-18))

    if keep_L >= chi or keep_R >= chi:
        # We cannot truncate, so return.
        # Nothing is done to the MPS, except for moving the OC one site ot the left
        return TruncationError(), 1, trace_env, MPO_envs

    # Build M matrix, Eqn. 15 of paper
    M = npc.tensordot(Q_L, Q_R.scale_axis(S, axis='vL'), axes=(['vR', 'vL'])).ireplace_labels(['vR*', 'vL*'], ['vL', 'vR'])
    M_norm = npc.norm(M)

    # traceful_ind_L(R) should be where the identity operator is located in the virtual left (right) Hilbert space.
    # When building the QR_L(R) matrix with local or conjoined methods, there will be one (or more) operators that correspond
    # to the identity operator. This should be doubled_site.traceful_ind since we directly imbed the operator space basis
    # into the virtual Hilbert space. We find where this is by using the site permutation. For spin-1/2, the permutation is
    # [1,0,3,2] for operators [S^-,I,Z,S^+]. Even if we have tensor product or direct sum of operator Hilbert spaces, the
    # Identity is still in the permuted 0th index.
    # For MPO preservation, there is not guaranteed to be one operator for the identity.
    M_trunc, err = truncate_M(M, svd_trunc_params, dmt_params.get('connected', False),
                              keep_L, keep_R, proj_L, proj_R,
                              traceful_ind_L=np.argsort(dMPS.sites[i].perm)[0], traceful_ind_R=np.argsort(dMPS.sites[i+1].perm)[0])
    if timing:
        time2 = time.time()
        print('Truncate M Block Time:', time2 - time1, flush=True)
        time1 = time2

    # SAJANT - Set svd_min to 0 to make sure no SVs are dropped? Or do we need some cutoff to remove the
    # SVs corresponding to the rank we removed earlier from M_DR
    U, S, VH, err2, renormalization2 = svd_theta(M_trunc, svd_trunc_params_2, renormalize=True)
    err2 = TruncationError.from_norm(renormalization2, norm_old=M_norm)
    # M_trunc (if normalized) would have norm 1 (or the original norm) if we did no truncation; so the new norm (given by `renormalization2`) is akin to
    # the error.
    if timing:
        time2 = time.time()
        print('SVD M Time:', time2 - time1, flush=True)
        time1 = time2

    # We want to remove the environments containing the bond i
    trace_env.del_RP(i)
    trace_env.del_LP(i+1)
    if MPO_envs is not None:
        for Me in MPO_envs:
            Me.del_RP(i)
            Me.del_LP(i+1)
    # Put new tensors back into the MPS
    new_A = npc.tensordot(npc.tensordot(dMPS.get_B(i, form='A'), Q_L.conj(), axes=(['vR', 'vR*'])), U, axes=(['vR', 'vL']))
    new_B = npc.tensordot(VH, npc.tensordot(Q_R.conj(), dMPS.get_B(i+1, form='B'), axes=(['vL*', 'vL'])),axes=(['vR', 'vL']))

    dMPS.set_SR(i, S)
    dMPS.set_B(i, new_A, form='A')
    dMPS.set_B(i+1, new_B, form='B')
    #dMPS.test_sanity() # SAJANT - remove this for miniscule speed boost?

    return err2, renormalization2, trace_env, MPO_envs
