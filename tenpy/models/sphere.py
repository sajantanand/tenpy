"""Nearest-neighbor spin-S models on triangularly refined icosahedrons.

Refined icosahedron lattice of spin-S sites, coupled by nearest-neighbor interactions.
"""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np
from scipy.special import comb
from scipy.sparse import coo_matrix

from ..networks.site import SpinSite
from .model import CouplingMPOModel, NearestNeighborModel
from .lattice import Chain
from ..tools.params import asConfig
from ..linalg.charges import LegCharge, ChargeInfo

__all__ = ['SphereSpinModel']

def make_icosphere(f, R=1.0, return_all_distances=True, rotate_away_from_pole=False, sort=False):
    """
    Class-I geodesic icosahedral refinement.

    Returns:
        cart:  (N, 3) Cartesian coordinates on radius R sphere
        polar: (N, 2) columns (theta, phi), theta in [0,pi], phi in [0,2pi)
        faces_refined: (20 f^2, 3) triangle indices
        edges: (30 f^2, 2) nearest-neighbor graph edges
        A_geo: sparse weighted adjacency with spherical NN distances
        D_sphere: optional dense all-pairs spherical distance matrix
        D_chord:  optional dense all-pairs chord distance matrix
    """
    phi = (1 + np.sqrt(5)) / 2

    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [0,  1,  phi], [0, -1, -phi], [0,  1, -phi],
        [ phi, 0, -1], [phi, 0,  1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)

    verts /= np.linalg.norm(verts, axis=1)[:, None]
    
    phi2, theta = np.arccos(phi / np.sqrt(1 + phi**2)), 0
    U1 = np.array([[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]])
    U2 = np.array([[1, 0, 0], [0, np.cos(phi2), -np.sin(phi2)], [0, np.sin(phi2), np.cos(phi2)]])
    verts = verts @ U1.conj().T @ U2.conj().T

    faces = np.array([
        [0,11,5], [0,5,1], [0,1,7], [0,7,10], [0,10,11],
        [1,5,9], [5,11,4], [11,10,2], [10,7,6], [7,1,8],
        [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9],
        [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1],
    ], dtype=int)

    points = []
    point_index = {}
    faces_refined = []

    def add_point(p):
        p = p / np.linalg.norm(p)
        key = tuple(np.round(p, 12))
        if key not in point_index:
            point_index[key] = len(points)
            points.append(p)
        return point_index[key]

    for face in faces:
        a, b, c = verts[face]

        # local triangular grid
        idx = {}
        for i in range(f + 1):
            for j in range(f + 1 - i):
                p = ((f - i - j) * a + i * b + j * c) / f
                idx[(i, j)] = add_point(p)

        # small triangles
        for i in range(f):
            for j in range(f - i):
                v0 = idx[(i, j)]
                v1 = idx[(i + 1, j)]
                v2 = idx[(i, j + 1)]
                faces_refined.append([v0, v1, v2])

                if i + j < f - 1:
                    v3 = idx[(i + 1, j + 1)]
                    faces_refined.append([v1, v3, v2])

    cart = R * np.array(points)
    faces_refined = np.array(faces_refined, dtype=int)
    N = len(cart)

    # build triangulation edges
    edge_set = set()
    for tri in faces_refined:
        for u, v in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            if u > v:
                u, v = v, u
            edge_set.add((u, v))

    edges = np.array(sorted(edge_set), dtype=int)

    if sort:
        if sort == 3:
            x, y, z = cart.T
            (new_order, edge_distance) = sort_points3(cart, np.mod(np.arctan2(y, x), 2 * np.pi), edges)
            total_order = (new_order, edge_distance)
        elif sort == 2:
            total_order = new_order = sort_points2(cart, faces_refined, verbose=False)
        else:
            total_order = new_order = sort_points(cart)
        inverse_order = np.argsort(new_order)
        cart = cart[new_order,:]
        faces_refined = inverse_order[faces_refined]
        edges = inverse_order[edges]
    else:
        total_order = None

    if rotate_away_from_pole:
        phi2, theta = np.arccos(phi / np.sqrt(1 + phi**2))/2, 0.3
        U1 = np.array([[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]])
        U2 = np.array([[1, 0, 0], [0, np.cos(phi2), -np.sin(phi2)], [0, np.sin(phi2), np.cos(phi2)]])
        cart = cart @ U1.conj() @ U2.conj()

    # polar coordinates
    x, y, z = cart.T
    theta = np.arccos(np.clip(z / R, -1.0, 1.0))
    phi_angle = np.mod(np.arctan2(y, x), 2 * np.pi)
    polar = np.column_stack([theta, phi_angle])

    # nearest-neighbor spherical distances
    u = cart[edges[:, 0]] / R
    v = cart[edges[:, 1]] / R
    edge_geo = R * np.arccos(np.clip(np.sum(u * v, axis=1), -1.0, 1.0))

    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([edge_geo, edge_geo])
    A_geo = coo_matrix((data, (row, col)), shape=(N, N)).tocsr()

    if not return_all_distances:
        return cart, polar, faces_refined, edges, A_geo

    dots = np.clip((cart @ cart.T) / R**2, -1.0, 1.0)
    D_sphere = R * np.arccos(dots)
    D_chord = np.sqrt(np.maximum(
        0.0,
        np.sum(cart**2, axis=1)[:, None]
        + np.sum(cart**2, axis=1)[None, :]
        - 2 * cart @ cart.T
    ))

    return cart, polar, faces_refined, edges, A_geo, D_sphere, D_chord, total_order

def sort_points3(cart, phi_angles, edges, verbose=False):
    new_order = []
    available_points = list(range(cart.shape[0]))
    
    tip = np.where(cart[:,2]==1)[0].item()
    new_order.append(tip)

    reached = [tip]
    distance = np.zeros(cart.shape[0], dtype=int)
    list_edges = [[int(e) for e in ed] for ed in edges]
    while len(list_edges):
        current_edges = []
        which_edges = []
        for we, ed in enumerate(list_edges):
            if np.any([r in ed for r in reached]):
                current_edges.append(ed)
                which_edges.append(we)
        for we in which_edges[::-1]:
            del list_edges[we]

        for ed in current_edges:
            assert ed[0] in reached or ed[1] in reached
            if ed[0] in reached and ed[1] in reached:
                continue
            new = ed[0] if ed[0] not in reached else ed[1]
            old = ed[0] if ed[0] in reached else ed[1]
            distance[new] = distance[old] + 1
            reached.append(new)

    max_dist = max(distance)

    prev_phi = 0
    for i in range(1, max_dist + 1):
        levels = np.where(distance == i)[0]
        phis = phi_angles[levels]
        phis[phis < prev_phi] += 2*np.pi
        order = np.argsort(phis)
        new_order.extend([int(l) for l in levels[order]])
        prev_phi = np.max(phis)

    distance = distance[new_order]
    
    return new_order, distance

class SphereSpinModel(CouplingMPOModel):
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
        sort_charge : bool
            Whether to sort by charges of physical legs. `True` by default.
        Jx, Jy, Jz, hx, hy, hz, muJ, D, E  : float | array
            Coupling as defined for the Hamiltonian above.
            Defaults to Heisenberg ``Jx=Jy=Jz=1.`` with other couplings 0.

    """

    default_lattice = Chain
    force_default_lattice = True

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
        
        edges = model_params.get('edges', [], list)
        
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
        #for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
        for (u1, u2) in edges:
            self.add_coupling_term((Jx + Jy) / 4.0, u1, u2, 'Sp', 'Sm', plus_hc=True)
            self.add_coupling_term((Jx - Jy) / 4.0, u1, u2, 'Sp', 'Sp', plus_hc=True)
            self.add_coupling_term(Jz, u1, u2, 'Sz', 'Sz')
            self.add_coupling_term(muJ * 0.5j, u1, u2, 'Sm','Sp', plus_hc=True)
        # done

def sort_edges(edges):
    sorted_edges = []
    for (e1, e2) in [[int(e) for e in ed] for ed in edges]:
        if e1 > e2:
            e3 = e1
            e1 = e2
            e2 = e3
        sorted_edges.append((e1, e2))
    return sorted_edges
