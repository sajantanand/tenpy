"""Defines a class describing the local physical Hilbert space.

The :class:`Site` is the prototype, read it's docstring.

"""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np
import itertools
import copy
import warnings
import logging
logger = logging.getLogger(__name__)

from ..linalg import np_conserved as npc
from ..tools.misc import inverse_permutation, find_subclass
from ..tools.hdf5_io import Hdf5Exportable

__all__ = [
    'Site',
    'GroupedSite',
    'group_sites',
    'set_common_charges',
    'kron',
    'SpinHalfSite',
    'SpinSite',
    'FermionSite',
    'SpinHalfFermionSite',
    'SpinHalfHoleSite',
    'BosonSite',
    'ClockSite',
    'spin_half_species',
    'DoubledSite',
]


class Site(Hdf5Exportable):
    """Collects necessary information about a single local site of a lattice.

    This class defines what the local basis states are: it provides the :attr:`leg`
    defining the charges of the physical leg for this site.
    Moreover, it stores (local) on-site operators, which are directly available as attribute,
    e.g., ``self.Sz`` is the Sz operator for the :class:`SpinSite`.
    Alternatively, operators can be obtained with :meth:`get_op`.
    The operator names ``Id`` and ``JW`` are reserved for the identity and Jordan-Wigner strings.

    .. warning ::
        The order of the local basis can change depending on the charge conservation!
        This is a *necessary* feature since we need to sort the basis by charges for efficiency.
        We use the :attr:`state_labels` and :attr:`perm` to keep track of these permutations.

    .. versionchanged :: 0.10
        Added `sort_charge` defaulting to `False`.

    .. versionchanged :: 1.0
        Make `sort_charge` default to `True`.

    Parameters
    ----------
    leg : :class:`~tenpy.linalg.charges.LegCharge`
        Charges of the physical states, to be used for the physical leg of MPS.
    state_labels : None | list of str
        Optionally a label for each local basis states. ``None`` entries are ignored / not set.
    **site_ops :
        Additional keyword arguments of the form ``name=op`` given to :meth:`add_op`.
        The identity operator ``'Id'`` is automatically included.
        If no ``'JW'`` for the Jordan-Wigner string is given,
        ``'JW'`` is set as an alias to ``'Id'``.
    sort_charge : bool
        Whether :meth:`sort_charge` should be called at the end of initialization.
        This is usually a good idea to reduce potential overhead when using charge conservation.
        Note that this might permute the order of the local basis states!

    Attributes
    ----------
    leg : :class:`~tenpy.linalg.charges.LegCharge`
        Charges of the local basis states.
    state_labels : {str: int}
        (Optional) labels for the local basis states.
    opnames : set
        Labels of all onsite operators (i.e. ``self.op`` exists if ``'op'`` in ``self.opnames``).
        Note that :meth:`get_op` allows arbitrary concatenations of them.
    need_JW_string : set
        Labels of all onsite operators that need a Jordan-Wigner string.
        Used in :meth:`op_needs_JW` to determine whether an operator anticommutes or commutes
        with operators on other sites.
    ops : :class:`~tenpy.linalg.np_conserved.Array`
        Onsite operators are added directly as attributes to self.
        For example after ``self.add_op('Sz', Sz)`` you can use ``self.Sz`` for the `Sz` operator.
        All onsite operators have labels ``'p', 'p*'``.
    perm : 1D array
        Index permutation of the physical leg compared to `conserve=None`,
        i.e. ``OP_conserved = OP_nonconserved[np.ix_(perm,perm)]`` and
        ``perm[state_labels_conserved["some_state"]] == state_labels_nonconserved["some_state"]``.
    JW_exponent : 1D array
        Exponents of the ``'JW'`` operator, such that
        ``self.JW.to_ndarray() = np.diag(np.exp(1.j*np.pi* JW_exponent))``
    hc_ops : dict(str->str)
        Mapping from operator names to their hermitian conjugates.
        Use :meth:`get_hc_op_name` to obtain entries.
    charge_to_JW_parity : None | 1D array
        If set, it is a list of factors, one per charge, such that
        ``(-1)**np.mod(np.sum(charges * charge_to_JW_parity, axis=-1), 2)`` is the
        Jordan-Wigner sign associated to a given set of `charges`.
        See :meth:`charge_to_JW_signs` for more details.
        Often not defined at all or `None`, which indicates that charge information is not enough
        to extract the Jordan-Wigner signs, i.e., we might not have total fermion number as
        well-defined charge.
    used_sort_charge : bool
        Whether :meth:`sort_charge` was called.
        Note that the default argument for `permute_dense` in :meth:`add_op` changes to True in
        that case, to ensure a consistent use.

    Examples
    --------
    The following generates a site for spin-1/2 with Sz conservation.
    Note that ``Sx = (Sp + Sm)/2`` violates Sz conservation and is thus not a valid
    on-site operator.

    .. testsetup :: Site

        from tenpy.linalg import np_conserved as npc
        from tenpy.networks.site import Site

    .. doctest :: Site

        >>> chinfo = npc.ChargeInfo([1], ['2 * Sz'])
        >>> ch = npc.LegCharge.from_qflat(chinfo, [1, -1])
        >>> Sp = [[0, 1.], [0, 0]]
        >>> Sm = [[0, 0], [1., 0]]
        >>> Sz = [[0.5, 0], [0, -0.5]]
        >>> site = Site(ch, ['up', 'down'], Splus=Sp, Sminus=Sm, Sz=Sz)
        >>> print(site.Splus.to_ndarray())
        [[0. 0.]
         [1. 0.]]
        >>> print(site.get_op('Sminus').to_ndarray())
        [[0. 1.]
         [0. 0.]]
        >>> print(site.get_op('Splus Sminus').to_ndarray())
        [[0. 0.]
         [0. 1.]]

    Note that sorting the charges (which happens by default!) may lead to unintuitive
    matrix representations of the operators, because physicists are typically not used to
    writing them in the sorted basis (in this case ``['down', 'up']``);

    We get the unchanged order by setting ``sort_charges=False``. This is discouraged though,
    as it can introduce overhead.

    .. testsetup :: Site_sort_charge_False

        from tenpy.linalg import np_conserved as npc
        from tenpy.networks.site import Site
        chinfo = npc.ChargeInfo([1], ['Sz'])
        ch = npc.LegCharge.from_qflat(chinfo, [1, -1])
        Sp = [[0, 1.], [0, 0]]
        Sm = [[0, 0], [1., 0]]
        Sz = [[0.5, 0], [0, -0.5]]

    .. doctest :: Site_sort_charge_False

        >>> site = Site(ch, ['up', 'down'], Splus=Sp, Sminus=Sm, Sz=Sz, sort_charge=False)
        >>> print(site.Splus.to_ndarray())
        [[0. 1.]
         [0. 0.]]
        >>> print(site.get_op('Sminus').to_ndarray())
        [[0. 0.]
         [1. 0.]]
        >>> print(site.get_op('Splus Sminus').to_ndarray())
        [[1. 0.]
         [0. 0.]]

    """

    def __init__(self, leg, state_labels=None, sort_charge=True, **site_ops):
        self.used_sort_charge = False
        self.leg = leg
        self.state_labels = dict()
        if state_labels is not None:
            for i, v in enumerate(state_labels):
                if v is not None:
                    self.state_labels[str(v)] = i
        self.opnames = set()
        self.need_JW_string = set(['JW'])
        self.hc_ops = {}
        if not hasattr(self, 'perm'):  # default permutation for the local states
            self.perm = np.arange(self.dim)
        self.add_op('Id', npc.diag(1., self.leg), hc='Id')
        for name, op in site_ops.items():
            self.add_op(name, op)
        if 'JW' not in self.opnames:
            # include trivial `JW` to allow combinations
            # of bosonic and fermionic sites in an MPS
            self.add_op('JW', self.Id, hc='JW')
        if sort_charge:
            self.sort_charge()
        self.test_sanity()

    def change_charge(self, new_leg_charge=None, permute=None):
        """Change the charges of the site (in place).

        Parameters
        ----------
        new_leg_charge : :class:`LegCharge` | None
            The new charges to be used. If ``None``, use trivial charges.
        permute : ndarray | None
            The permutation applied to the physical leg,
            which also gets used to adjust :attr:`state_labels` and :attr:`perm`.
            If you sorted the previous leg with ``perm_qind, new_leg_charge = leg.sort()``,
            use ``old_leg.perm_flat_from_perm_qind(perm_qind)``.
            Ignored if ``None``.
        """
        if new_leg_charge is None:
            new_leg_charge = npc.LegCharge.from_trivial(self.dim)
        self.leg = new_leg_charge
        if permute is not None:
            permute = np.asarray(permute, dtype=np.intp)
            inv_perm = inverse_permutation(permute)
            self.perm = self.perm[permute]
            self.state_labels = dict((lbl, int(inv_perm[i])) for lbl, i in self.state_labels.items())
        for opname in self.opnames.copy():
            op = self.get_op(opname).to_ndarray()
            self.opnames.remove(opname)
            delattr(self, opname)
            if permute is not None:
                op = op[np.ix_(permute, permute)]
            # need_JW and hc_ops are still set
            self.add_op(opname, op, need_JW=False, hc=False, permute_dense=False)
        if hasattr(self, 'charge_to_JW_parity'):
            # might no longer be valid (unclear!), so better delete.
            del self.charge_to_JW_parity
        # done

    def sort_charge(self, bunch=True):
        """Sort the :attr:`leg` charges (in place).

        Parameters
        ----------
        bunch : bool
            Whether to also group equal charges into larger blocks (usually a good idea).

        Returns
        -------
        perm : 1D ndarray
            The permutation
        """
        if self.leg.sorted and (not bunch or self.leg.bunched):
            return np.arange(self.dim, dtype=np.intp)  # nothing to do
        perm_qind, leg_sorted = self.leg.sort(bunch)
        perm_flat = self.leg.perm_flat_from_perm_qind(perm_qind)
        charge_to_JW_parity = getattr(self, 'charge_to_JW_parity', None)
        self.change_charge(leg_sorted, perm_flat)
        # change_charge updates self.state_label and self.perm
        self.used_sort_charge = True
        if charge_to_JW_parity is not None:
            # preserve charge_to_JW_parity
            self.charge_to_JW_parity = charge_to_JW_parity
        return perm_flat

    def test_sanity(self):
        """Sanity check, raises ValueErrors, if something is wrong."""
        for lab, ind in self.state_labels.items():
            if not isinstance(lab, str):
                raise ValueError("wrong type of state label")
            if not 0 <= ind < self.dim:
                raise ValueError("index of state label out of bounds")
        for name in self.opnames:
            if not hasattr(self, name):
                raise ValueError("missing onsite operator " + name)
        for op in self.onsite_ops.values():
            if op.rank != 2:
                raise ValueError("only rank-2 onsite operators allowed")
            op.legs[0].test_equal(self.leg)
            op.legs[1].test_contractible(self.leg)
            op.test_sanity()
        for op in self.need_JW_string:
            assert op in self.opnames
        np.testing.assert_array_almost_equal(np.diag(np.exp(1.j * np.pi * self.JW_exponent)),
                                             self.JW.to_ndarray(), 15)
        if hasattr(self, 'hc_ops'):
            for op1, op2 in self.hc_ops.items():
                assert op1 in self.opnames and op2 in self.opnames
                op1 = self.get_op(op1)
                op2 = self.get_op(op2)
                assert op1.conj().transpose() == op2
        if getattr(self, 'charge_to_JW_parity', None) is not None:
           JW_diag = np.diag(self.JW.to_ndarray())
           JW_signs = self.charge_to_JW_signs(self.leg.to_qflat())
           np.testing.assert_array_almost_equal(JW_diag, JW_signs, 14)

    @property
    def dim(self):
        """Dimension of the local Hilbert space."""
        return self.leg.ind_len

    @property
    def onsite_ops(self):
        """Dictionary of on-site operators for iteration.

        Single operators are accessible as attributes.
        """
        return dict([(name, getattr(self, name)) for name in sorted(self.opnames)])

    def add_op(self, name, op, need_JW=False, hc=None, permute_dense=None):
        """Add one on-site operators.

        Parameters
        ----------
        name : str
            A valid python variable name, used to label the operator.
            The name under which `op` is added as attribute to self.
        op : np.ndarray | :class:`~tenpy.linalg.np_conserved.Array`
            A matrix acting on the local hilbert space representing the local operator.
            Dense numpy arrays are automatically converted to
            :class:`~tenpy.linalg.np_conserved.Array`.
            LegCharges have to be ``[leg, leg.conj()]``.
            We set labels ``'p', 'p*'``.
        need_JW : bool
            Whether the operator needs a Jordan-Wigner string.
            If ``True``, add `name` to :attr:`need_JW_string`.
        hc : None | False | str
            The name for the hermitian conjugate operator, to be used for :attr:`hc_ops`.
            By default (``None``), try to auto-determine it.
            If ``False``, disable adding entries to :attr:`hc_ops`.
        permute_dense : bool | None
            Flag to enable/disable permutations when converting `op` from numpy to
            np_conserved arrays.
            If True, the operator is permuted with :attr:`perm` to account for permutations
            induced by sorting charges; False disables the permutations.
            By default (``None``), the value of :attr:`used_sort_charge` is used.
        """
        name = str(name)
        if not name.isidentifier():
            raise ValueError("Invalid operator name: " + name)
        if name in self.opnames:
            raise ValueError("Operator with that name already existent: " + name)
        if hasattr(self, name):
            raise ValueError("Site already has that attribute name: " + name)
        if not isinstance(op, npc.Array):
            op = np.asarray(op)
            if op.shape != (self.dim, self.dim):
                raise ValueError("wrong shape of on-site operator")
            if permute_dense is None:
                permute_dense = self.used_sort_charge
            if permute_dense:
                perm = self.perm
                op = op[np.ix_(perm, perm)]
            try:
                op = npc.Array.from_ndarray(op, [self.leg, self.leg.conj()])
            except ValueError as e:
                # just add a more help-ful error message printing the operators
                raise ValueError('\n'.join([
                    f"Can't convert operator {name!r} to npc Array", "Flat charges:",
                    str(self.leg.to_qflat()), "Operator:",
                    str(op)
                ])) from e
        if op.rank != 2:
            raise ValueError("only rank-2 on-site operators allowed")
        op.legs[0].test_equal(self.leg)
        op.legs[1].test_contractible(self.leg)
        op.test_sanity()
        op.iset_leg_labels(['p', 'p*'])
        setattr(self, name, op)
        self.opnames.add(name)
        if need_JW:
            self.need_JW_string.add(name)
        # keep track of h.c. operators
        if hc is None and not name in self.hc_ops:
            if op.conj().transpose() == op:
                hc = name
            else:
                for other in self.opnames:
                    other_op = self.get_op(other)
                    if other_op.conj().transpose() == op:
                        hc = other
                        break
        if hc:
            self.hc_ops[hc] = name
            self.hc_ops[name] = hc
        if name == 'JW':
            self.JW_exponent = np.angle(np.real_if_close(np.diag(op.to_ndarray()))) / np.pi

    def rename_op(self, old_name, new_name):
        """Rename an added operator.

        Parameters
        ----------
        old_name : str
            The old name of the operator.
        new_name : str
            The new name of the operator.
        """
        if old_name == new_name:
            return
        if new_name in self.opnames:
            raise ValueError("new_name already exists")
        old_hc_name = self.hc_ops.get(old_name, None)
        op = getattr(self, old_name)
        need_JW = old_name in self.need_JW_string
        hc_op_name = self.get_hc_op_name(old_name)
        self.remove_op(old_name)
        setattr(self, new_name, op)
        self.opnames.add(new_name)
        if need_JW:
            self.need_JW_string.add(new_name)
        if new_name == 'JW':
            self.JW_exponent = np.real_if_close(np.angle(np.diag(op.to_ndarray())) / np.pi)
        if old_hc_name is not None:
            if old_hc_name == old_name:
                self.hc_ops[new_name] = new_name
            else:
                self.hc_ops[new_name] = old_hc_name
                self.hc_ops[old_hc_name] = new_name

    def remove_op(self, name):
        """Remove an added operator.

        Parameters
        ----------
        name : str
            The name of the operator to be removed.
        """
        hc_name = self.hc_ops.get(name, None)
        if hc_name is not None:
            del self.hc_ops[name]
            if hc_name != name:
                del self.hc_ops[hc_name]
        self.opnames.remove(name)
        delattr(self, name)
        self.need_JW_string.discard(name)

    def state_index(self, label):
        """Return index of a basis state from its label.

        Parameters
        ----------
        label : int | string
            either the index directly or a label (string) set before.

        Returns
        -------
        state_index : int
            the index of the basis state associated with the label.
        """
        res = self.state_labels.get(label, label)
        try:
            res = int(res)
        except ValueError:
            raise KeyError("label not found: " + repr(label))
        return res

    def state_indices(self, labels):
        """Same as :meth:`state_index`, but for multiple labels."""
        return [self.state_index(lbl) for lbl in labels]

    def get_op(self, name):
        """Return operator of given name.

        Parameters
        ----------
        name : str
            The name of the operator to be returned.
            In case of multiple operator names separated by whitespace,
            we multiply them together to a single on-site operator
            (with the one on the right acting first).

        Returns
        -------
        op : :class:`~tenpy.linalg.np_conserved`
            The operator given by `name`, with labels ``'p', 'p*'``.
            If name already was an npc Array, it's directly returned.
        """
        names = name.split()
        op = getattr(self, names[0], None)
        if op is None:
            raise ValueError("{0!r} doesn't have the operator {1!r}".format(self, names[0]))
        for name2 in names[1:]:
            op2 = getattr(self, name2, None)
            if op2 is None:
                raise ValueError("{0!r} doesn't have the operator {1!r}".format(self, name2))
            op = npc.tensordot(op, op2, axes=['p*', 'p'])
        return op

    def get_hc_op_name(self, name):
        """Return the hermitian conjugate of a given operator.

        Parameters
        ----------
        name : str
            The name of the operator to be conjugated.
            Multiple operators separated by whitespace are interpreted as an operator product,
            exactly as :meth:`get_op` does.

        Returns
        -------
        hc_op_name : str
            Operator name for the hermitian conjugate operator.
        """
        names = name.split()
        hc_names = []
        for name2 in reversed(names):
            hc_name_2 = self.hc_ops.get(name2)
            if hc_name_2 is None:
                raise ValueError("hermitian conjugate of operator {0!s} unknown".format(name2))
            hc_names.append(hc_name_2)
        return ' '.join(hc_names)

    def op_needs_JW(self, name):
        """Whether an (composite) onsite operator is fermionic and needs a Jordan-Wigner string.

        Parameters
        ----------
        name : str
            The name of the operator, as in :meth:`get_op`.

        Returns
        -------
        needs_JW : bool
            Whether the operator needs a Jordan-Wigner string, judging from :attr:`need_JW_string`.
        """
        names = name.split()
        need_JW = bool(names[0] in self.need_JW_string)
        for op in names[1:]:
            if op in self.need_JW_string:
                need_JW = not need_JW  # == need_JW xor (op in self.need_JW_string)
        return need_JW

    def valid_opname(self, name):
        """Check whether 'name' labels a valid onsite-operator.

        Parameters
        ----------
        name : str
            Label for the operator. Can be multiple operator(labels) separated by whitespace,
            indicating that they should  be multiplied together.

        Returns
        -------
        valid : bool
            ``True`` if `name` is a valid argument to :meth:`get_op`.
        """
        for name2 in name.split():
            if name2 not in self.opnames:
                return False
        return True

    def multiply_op_names(self, names):
        """Multiply operator names together.

        Join the operator names in `names` such that `get_op` returns the product of the
        corresponding operators.

        Parameters
        ----------
        names : list of str
            List of valid operator labels.

        Returns
        -------
        combined_opname : str
            A valid operator name
            Operator name representing the product of operators in `names`.
        """
        if len(names) == 0:
            return 'Id'
        return ' '.join(names)

    def multiply_operators(self, operators):
        """Multiply local operators (possibly given by their names) together.

        Parameters
        ----------
        operators : list of {str | :class:`~tenpy.linalg.np_conserved.Array`}
            List of valid operator names (to be translated with :meth:`get_op`) or
            directly on-site operators in the form of npc arrays with ``'p', 'p*'`` label.
            The operators are multiplied left-to-right.

        Returns
        -------
        combined_operator : :class:`~tenpy.linalg.np_conserved.Array`
            The product of the given `operators` in a left-to-right multiplication following the
            usual mathematical convention. For example, if ``operators=['Sz', 'Sp', 'Sx']``,
            the final operator is equivalent to ``site.get_op('Sz Sp Sx')``, with the ``'Sx'``
            operator acting first on any physical state.
        """
        if len(operators) == 0:
            return self.Id
        op = operators[0]
        if isinstance(op, str):
            op = self.get_op(op)
        for next_op in operators[1:]:
            if isinstance(next_op, str):
                next_op = self.get_op(next_op)
            op = npc.tensordot(op, next_op, axes=['p*', 'p'])
        return op

    def __repr__(self):
        """Debug representation of self."""
        return "<Site, d={dim:d}, ops={ops!r}>".format(dim=self.dim, ops=self.opnames)

    def charge_to_JW_signs(self, charges):
        """Convert charge values to Jordan-Wigner parity.

        Often, charge conservation contains the (parity of) the total fermion number.
        This information is enough to lift a Jordan-Wigner string applied on the left of a given
        bond to the virtual leg of an MPS: given the total parity number of fermions
        ``parity[alpha] = N_fermions[alpha] % 2`` in each Schmidt state ``|alpha>``,
        simply send ``|alpha> --> (-1)**parity[alpha] |alpha>``.
        Given the charges values of the Schmidt states ``|alpha>``, this function returns the
        corresponding ``(-1)**parity`` Jordan-Wigner signs.

        Parameters
        ----------
        charges : 2D or 1D array
            Charge values, last dimension is len ``chinfo.qnumber``.
            We choose the convention that these charge values correspond to an "incoming" leg
            with ``qconj=+1``.

        Returns
        -------
        JW_signs :
            Should only have values +1 or -1.
        """
        charge_to_JW_parity = getattr(self, 'charge_to_JW_parity', None)
        if charge_to_JW_parity is not None:
            charges = self.leg.chinfo.make_valid(charges)
            parity = np.mod(np.sum(charges * charge_to_JW_parity, axis=-1), 2)
            # parity has values in [0, 1]
            return 1. - 2. * parity  # values +/- 1, same as (-1)**parity
        raise ValueError("`charge_to_JW_parity` not defined!")


class GroupedSite(Site):
    """Group two or more :class:`Site` into a larger one.

    A typical use-case is that you want a NearestNeighborModel for TEBD although you have
    next-nearest neighbor interactions: you just double your local Hilbertspace to consist of
    two original sites.
    Note that this is a 'hack' at the cost of other things (e.g., measurements of 'local'
    operators) getting more complicated/computationally expensive.

    If the individual sites indicate fermionic operators (with entries in `need_JW_string`),
    we construct the new on-site operators of `site1` to include the JW string of `site0`,
    i.e., we use the Kronecker product of ``[JW, op]`` instead of ``[Id, op]`` if necessary
    (but always ``[op, Id]``).
    In that way the onsite operators of this DoubleSite automatically fulfill the
    expected commutation relations. See also :doc:`/intro/JordanWigner`.

    Parameters
    ----------
    sites : list of :class:`Site`
        The individual sites being grouped together. Copied before use if ``charges!='same'``.
    labels :
        Include the Kronecker product of each onsite operator `op` on ``sites[i]`` and
        identities on other sites with the name ``opname+labels[i]``.
        Similarly, set state labels for ``' '.join(state[i]+'_'+labels[i])``.
        Defaults to ``[str(i) for i in range(n_sites)]``, which for example grouping two SpinSites
        gives operators name like ``"Sz0"`` and state labels like ``'up_0 down_1'``.
    charges : ``'same' | 'drop' | 'independent'``
        How to handle charges, defaults to 'same'.
        ``'same'`` means that all `sites` have the same `ChargeInfo`, and the total charge
        is the sum of the charges on the individual `sites`.
        ``'independent'`` means that the `sites` have possibly different `ChargeInfo`,
        and the charges are conserved separately, i.e., we have `n_sites` conserved charges.
        For ``'drop'``, we drop any charges, such that the remaining legcharges are trivial.
        For more complex situations, you can call :func:`set_common_charges` beforehand.

    Attributes
    ----------
    n_sites : int
        The number of sites grouped together, i.e. ``len(sites)``.
    sites : list of :class:`Site`
        The sites grouped together into self.
    labels: list of str
        The labels using which the single-site operators are added during construction.
    """

    def __init__(self, sites, labels=None, charges='same'):
        self.n_sites = n_sites = len(sites)
        self.sites = sites
        self.charges = charges
        assert n_sites > 0
        if labels is None:
            labels = [str(i) for i in range(n_sites)]
        self.labels = labels
        if charges == 'same':
            pass  # nothing to do
        elif charges == 'drop':
            legs = [npc.LegCharge.from_drop_charge(sites[0].leg)]
            chinfo = legs[0].chinfo
            for site in sites[1:]:
                legs.append(npc.LegCharge.from_drop_charge(sites[0].leg, chargeinfo=chinfo))
        elif charges == 'independent':
            # charges are separately conserved
            legs = []
            for i in range(n_sites):
                d = sites[i].dim
                # trivial charges
                legs_triv = [npc.LegCharge.from_trivial(d, s.leg.chinfo) for s in sites]
                legs_triv[i] = sites[i].leg  # except on site i
                chinfo = None if i == 0 else legs[0].chinfo
                leg = npc.LegCharge.from_add_charge(legs_triv, chinfo)  # combine the charges
                legs.append(leg)
        else:
            raise ValueError("Unknown option for `charges`: " + repr(charges))
        c2JWps = [getattr(s, 'charge_to_JW_parity', None) for s in sites]  # maybe keep it below
        if charges != 'same':
            sites = [copy.copy(s) for s in sites]  # avoid modifying the existing sites.
            # sort legs
            for i in range(n_sites):
                perm_qind, leg_s = legs[i].sort()
                sites[i].change_charge(leg_s, legs[i].perm_flat_from_perm_qind(perm_qind))
        chinfo = sites[0].leg.chinfo
        for s in sites[1:]:
            assert s.leg.chinfo == chinfo  # check for compatibility
        legs = [s.leg for s in sites]
        pipe = npc.LegPipe(legs)
        self.leg = pipe  # needed in kroneckerproduct
        JW_all = self.kroneckerproduct([s.JW for s in sites])

        # initialize Site
        Site.__init__(self, pipe, None, JW=JW_all)
        # note: the pipe is sorted, so sort_charge option doesn't matter

        # set state labels
        for states_labels in itertools.product(*[s.state_labels.items() for s in sites]):
            inds = [v for k, v in states_labels]  # values of the dictionaries
            ind_pipe = pipe.map_incoming_flat(inds)
            label = ' '.join([st + '_' + lbl for (st, idx), lbl in zip(states_labels, labels)])
            self.state_labels[label] = ind_pipe

        # add remaining operators
        Ids = [s.Id for s in sites]
        JW_Ids = Ids[:]  # in the following loop equivalent to [JW, JW, ... , Id, Id, ...]
        for i in range(n_sites):
            site = sites[i]
            for opname, op in site.onsite_ops.items():
                if opname == 'Id':
                    continue
                need_JW = opname in site.need_JW_string
                hc_opname = site.hc_ops.get(opname, None)
                if hc_opname is None:
                    hc_opname = False
                else:
                    hc_opname = hc_opname + labels[i]
                ops = JW_Ids if need_JW else Ids
                ops[i] = op
                self.add_op(opname + labels[i], self.kroneckerproduct(ops), need_JW, hc_opname)
                Ids[i] = site.Id
                JW_Ids[i] = site.JW

        # propagate `charge_to_JW_parity` if safe/clear what it should be
        # read it into c2JWps for each site before calling Site.change_charge() deleting it
        if charges == 'same':
            # already same charges, so could/should have same `charge_to_JW_parity`
            if all(p is not None and all(p == c2JWps[0]) for p in c2JWps):
                self.charge_to_JW_parity = c2JWps[0]
        elif charges == 'independent':
            if all(p is not None for p in c2JWps):
                self.charge_to_JW_parity = np.concatenate(c2JWps)
        # other cases: not immediately clear what charge_to_JW_parity should be / is still valid

    def kroneckerproduct(self, ops):
        r"""Return the Kronecker product :math:`op0 \otimes op1` of local operators.

        Parameters
        ----------
        ops : list of :class:`~tenpy.linalg.np_conserved.Array`
            One operator (or operator name) on each of the ungrouped sites.
            Each operator should have labels ``['p', 'p*']``.

        Returns
        -------
        prod : :class:`~tenpy.linalg.np_conserved.Array`
            Kronecker product :math:`ops[0] \otimes ops[1] \otimes \cdots`,
            with labels ``['p', 'p*']``.
        """
        sites = self.sites
        op = ops[0].transpose(['p', 'p*'])
        for op2 in ops[1:]:
            op = npc.outer(op, op2.transpose(['p', 'p*']))
        combine = [list(range(0, 2 * self.n_sites - 1, 2)), list(range(1, 2 * self.n_sites, 2))]
        pipe = self.leg
        op = op.combine_legs(combine, pipes=[pipe, pipe.conj()])
        return op.iset_leg_labels(['p', 'p*'])

    def __repr__(self):
        """Debug representation of self."""
        return "GroupedSite({sites!r}, {labels!r}, {charges!r})".format(sites=self.sites,
                                                                        labels=self.labels,
                                                                        charges=self.charges)


def group_sites(sites, n=2, labels=None, charges='same'):
    """Given a list of sites, group each `n` sites together.

    Parameters
    ----------
    sites : list of :class:`Site`
        The sites to be grouped together.
    n : int
        We group each `n` consecutive sites from `sites` together in a :class:`GroupedSite`.
    labels, charges :
        See :class:`GroupedSite`.

    Returns
    -------
    grouped_sites : list of :class:`GroupedSite`
        The grouped sites. Has length ``(len(sites)-1)//n + 1``.
    """
    grouped_sites = []
    if labels is None:
        labels = [str(i) for i in range(n)]
    for i in range(0, len(sites), n):
        group = sites[i:i + n]
        s = GroupedSite(group, labels[:len(group)], charges)
        grouped_sites.append(s)
    return grouped_sites


def set_common_charges(sites, new_charges='same', new_names=None, new_mod=None, sort_charge=True):
    r"""Adjust the charges of the given sites *in place* such that they can be used together.

    Before we can contract operators (and tensors) corresponding to different :class:`Site`
    instances, we first need to define the overall conserved charges, i.e., we need to merge the
    :class:`~tenpy.linalg.charges.ChargeInfo` of them to a single, global `chinfo` and adjust
    the charges of the physical legs. That's what this function does.

    A typical place to do this would be in :meth:`tenpy.models.model.CouplingMPOModel.init_sites`.

    Parameters
    ----------
    sites : list of :class:`Site`
        The sites to be combined. The sites are modified **in place**.
    new_charges : ``'same'`` | ``'drop'`` | ``'independent'`` |  list of list of tuple
        Defines the new, common charges in terms of the old ones.

        list of lists of tuple
            If a list is given, each entry `new_charge` of the list defines one new charge,
            i.e. the new number of charges is ``qnumber=len(new_charges)``.
            Each entry `new_charge` of the outer list is itself a list of 3-tuples,
            ``new_charge = [(factor, site_index, old_charge_index), ...]``.
            where the value of the new charge is the sum of `factor` times the value of the old
            charge, (specified by the `site_index` and the `old_charge_index` within that site),
            and the sum runs over all entries in that list `new_charge`.
            `old_charge_index` can be an integer (=the index) or a string (=the name) of the
            charge in the corresponding ``sites[site_index].leg.chinfo``.
        ``'same'``
            defaults to charges with the same name to match, and charges with different
            names to be independently conserved (see example below);
            ``None``-set names are considered different.
        ``'drop'``
            Drop/remove all charges, equivalent to ``new_charges=[]``.
        ``'independent'``
            For the case that the charges of the different sites are independent and individually
            conserved, even if they have the same name.
    new_names : list of str
        Names for each of the new charges. Defaults to name of the first old charge specified.
    new_mod : list of int
        :attr:`~tenpy.linalg.charges.ChargeInfo.mod` for the new charges, one entry for each list
        in `new_charges`. Defaults to the `mod` of the old charges, if not specified otherwise.
    sort_charge : bool
        Whether to sort the physical legs by charges.

    Returns
    -------
    perms : list of ndarray
        For each site the permutation performed on the physical leg to sort by charges.
        Only returned if `sort_charge` is True.

    Examples
    --------
    When we just initialize some sites, they will in general have different charges.
    For example, we could have a :class:`SpinHalfFermionSite` a spin-1 :class:`SpinSite`.
    For reference, let's also print the names and values of the charges.

    .. doctest :: set_common_charges
        :options: +NORMALIZE_WHITESPACE

        >>> from tenpy.networks.site import *
        >>> ferm = SpinHalfFermionSite(cons_N='N', cons_Sz='Sz')
        >>> ferm.leg.chinfo.names
        ['N', '2*Sz']
        >>> print(ferm.leg.to_qflat())
        [[ 1 -1]
         [ 0  0]
         [ 2  0]
         [ 1  1]]
        >>> spin = SpinSite(1.0, conserve='Sz')
        >>> spin.leg.chinfo.names
        ['2*Sz']
        >>> print(spin.leg.to_qflat())
        [[-2]
         [ 0]
         [ 2]]

    With the default ``new_charges='same'``, this function will combine charges with the same name,
    and hence we will have two conserved quantities, namely
    the fermion particle number
    ``'N' = N_{up_fermions} + N_{down-fermions}``,
    and the total Sz spin
    ``'2*Sz' = N_{up-fermions} + N_{up-spins} - N_{down-fermions} - N_{down-spins}``.
    In this case, there will only appear an extra column of zeros for the charges of the spin leg.

    .. doctest :: set_common_charges
        :options: +NORMALIZE_WHITESPACE

        >>> set_common_charges([ferm, spin], new_charges='same')
        [array([0, 1, 2, 3]), array([0, 1, 2])]
        >>> ferm.leg.chinfo.names
        ['N', '2*Sz']
        >>> print(ferm.leg.to_qflat())  # didn't change (except making a copy)
        [[ 1 -1]
         [ 0  0]
         [ 2  0]
         [ 1  1]]
        >>> spin.leg.chinfo.names   # additional 'N' chargename
        ['N', '2*Sz']
        >>> print(spin.leg.to_qflat())  # additional column of zeros for the 'N' charge
        [[ 0 -2]
         [ 0  0]
         [ 0  2]]

    With ``new_charges='independent'``, we preserve the charges of the old sites individually.
    In this example, we get 3 conserved quantities, namely the fermion particle number
    ``'N_ferm' = N_{up_fermions} + N_{down-fermions}``,
    and the fermionic Sz spin ``'2*Sz_ferm' = N_{up-fermions} - N_{down-fermions}``
    and the Sz spin of the `spin` sites, ``'2*Sz_spin' = N_{up-spins} - N_{down-spins}``.
    (We give the charges new names for clearer distinction.)
    Corresponding zero columns are added to the LegCharges.

    .. doctest :: set_common_charges
        :options: +NORMALIZE_WHITESPACE

        >>> ferm = SpinHalfFermionSite(cons_N='N', cons_Sz='Sz')
        >>> spin = SpinSite(1.0, conserve='Sz')
        >>> set_common_charges([ferm, spin], new_charges='independent',
        ...                    new_names=['N_ferm', '2*Sz_ferm', '2*Sz_spin'])
        [array([0, 1, 2, 3]), array([0, 1, 2])]
        >>> print(ferm.leg.to_qflat())  # additional columns of zeros
        [[ 1 -1  0]
         [ 0  0  0]
         [ 2  0  0]
         [ 1  1  0]]
        >>> print(spin.leg.to_qflat())  # two additional columns of zeros
        [[ 0  0 -2]
         [ 0  0  0]
         [ 0  0  2]]

    With the full specification of the `new_charges` through a list of list of tuples,
    you can create new charges as linear combinations of the charges of the individual sites.
    For example, the `SpinHalfFermionSite` is essentially the product of two `FermionSite`, one for
    the up electrons, and one for the down electrons. The ``'2*Sz'`` charge of the
    `SpinHalfFermionSite` is then equivalent to the difference of individual particle numbers,
    ``'2*Sz' = N_{up} - N_{down}``.

    .. doctest :: set_common_charges
        :options: +NORMALIZE_WHITESPACE

        >>> f_up = FermionSite(conserve='N')
        >>> f_down = FermionSite(conserve='N')
        >>> print(f_up.leg.to_qflat())
        [[0]
         [1]]
        >>> print(f_down.leg.to_qflat())
        [[0]
         [1]]
        >>> f_down.state_labels
        {'empty': 0, 'full': 1}
        >>> set_common_charges([f_up, f_down],
        ...                    new_charges=[[(1, 0, 'N'), ( 1, 1, 'N')],
        ...                                 [(1, 0, 'N'), (-1, 1, 'N')]],
        ...                    new_names=['N_tot', '2*Sz=(N_up-N_down)'])
        [array([0, 1]), array([1, 0])]
        >>> f_down.state_labels  # sorting charges caused permutation of local states
        {'empty': 1, 'full': 0}
        >>> print(f_up.leg.to_qflat())
        [[0 0]
         [1 1]]
        >>> print(f_down.leg.to_qflat()) # top row = full, bottom row=empty
        [[ 1 -1]
         [ 0  0]]

    Another example could be that you have both fermions and bosons,
    and that you have terms :math:`c_i c_j b^\dagger_k + c^\dagger_i c^\dagger_j b_k`,
    where two fermions can merge into a pair forming a boson.
    In this case, neither the fermion number nor the boson number is preserved individually,
    but the combination ``N_{fermions} + 2 * N_{bosons}`` is preserved.

    .. doctest :: set_common_charges
        :options: +NORMALIZE_WHITESPACE

        >>> ferm = FermionSite(conserve='N')
        >>> bos = BosonSite(Nmax=3, conserve='N')
        >>> set_common_charges([ferm, bos], [[(1, 0, 'N'), (2, 1, 'N')]], ['N_f + 2 N_b'])
        [array([0, 1]), array([0, 1, 2, 3])]

    The ``new_charges='drop'`` or ``new_charges=[]`` option is a quick way to remove any charges.

    .. doctest :: set_common_charges
        :options: +NORMALIZE_WHITESPACE

        >>> ferm = SpinHalfFermionSite(cons_N='N', cons_Sz='Sz')
        >>> spin = SpinSite(1.0, conserve='Sz')
        >>> set_common_charges([ferm, spin], new_charges='drop')
        [array([0, 1, 2, 3]), array([0, 1, 2])]
        >>> assert ferm.leg.chinfo.qnumber == spin.leg.chinfo.qnumber == 0  # trivial: no charges
    """
    for s, site in enumerate(sites):
        for site2 in sites[s + 1:]:
            if site2 is site:
                raise ValueError("`sites` contains the same object multiple times. Make copies!")
    old_chinfos = [site.leg.chinfo for site in sites]
    if isinstance(new_charges, str):
        if new_charges == 'same':
            new_charges = []
            name_to_new_idx = {}
            for s, site in enumerate(sites):
                chinfo = site.leg.chinfo
                for i, n in enumerate(chinfo.names):
                    if n is None:
                        new_charges.append([(1, s, i)])  #independent charge
                    else:
                        if n not in name_to_new_idx:
                            name_to_new_idx[n] = len(new_charges)
                            new_charges.append([(1, s, i)])
                        else:
                            new_charges[name_to_new_idx[n]].append((1, s, i))
        elif new_charges == 'drop':
            new_charges = []
        elif new_charges == 'independent':
            new_charges = [[(1, s, i)] for s, site in enumerate(sites)
                           for i in range(site.leg.chinfo.qnumber)]
        else:
            raise ValueError("unknown option for new_charges: " + repr(new_charges))
    else:
        # parse new_charges argument: translate old_charge_idx names to indices and error check
        new_charges = list(new_charges)  # copy: need to modify elements
        for i, new_charge in enumerate(new_charges):
            new_charges[i] = new_charge = list(new_charge)  # copy before modification
            assert len(new_charge) > 0
            for j, (factor, s, old_idx) in enumerate(new_charge):
                if isinstance(old_idx, str):
                    old_idx = old_chinfos[s].names.index(old_idx)
                    new_charge[j] = (factor, s, old_idx)
                if not 0 <= old_idx < old_chinfos[s].qnumber:
                    raise ValueError("wrong `site_index` or `old_charge_index` in new_charges")
    # setup new `chinfo`
    qnumber = len(new_charges)
    if new_names is None:
        new_names = [old_chinfos[lst[0][1]].names[lst[0][2]] for lst in new_charges]
    assert len(new_names) == qnumber
    if new_mod is None:
        new_mod = [old_chinfos[lst[0][1]].mod[lst[0][2]] for lst in new_charges]
        for i, new_charge in enumerate(new_charges):
            for (_, s, oi) in new_charge:
                if old_chinfos[s].mod[oi] != new_mod[i]:
                    # (this is only tested if new_mod isn't set explicitly)
                    raise ValueError("Charges which get combined have different `mod` nature!")
    assert len(new_mod) == qnumber
    new_chinfo = npc.ChargeInfo(new_mod, new_names)

    # get new charge_to_JW_parity if possible
    new_charge_to_JW_parity = _set_common_charges_charge_to_JW_parity(sites, new_charges, new_mod)

    # define the new leg charges and update the sites
    perms = []
    for new_s, site in enumerate(sites):
        old_qflat = site.leg.to_qflat()
        # determine new leg charges
        new_qflat = np.zeros((site.leg.ind_len, qnumber), old_qflat.dtype)
        for new_i, new_charge in enumerate(new_charges):
            for factor, old_s, old_i in new_charge:
                if old_s == new_s:
                    old_qflat_i = factor * old_qflat[:, old_i]
                    if old_qflat_i.dtype != new_qflat.dtype:
                        unrounded_old_qflat_i = old_qflat_i
                        old_qflat_i = np.array(np.rint(old_qflat_i), dtype=new_qflat.dtype)
                        if np.any(np.abs(old_qflat_i - unrounded_old_qflat_i) > 1.e-5):
                            raise ValueError("float `factor` causes non-integer charges")
                    new_qflat[:, new_i] += old_qflat_i
        # update the site with the new charges
        leg_unsorted = npc.LegCharge.from_qflat(new_chinfo, new_qflat, site.leg.qconj)
        if sort_charge:
            perm_qind, leg = leg_unsorted.sort()
            perm_flat = leg_unsorted.perm_flat_from_perm_qind(perm_qind)
            perms.append(perm_flat)
        else:
            perm_flat = None
        site.change_charge(leg, perm_flat)
        if new_charge_to_JW_parity is not None:
            site.charge_to_JW_parity = new_charge_to_JW_parity
    if sort_charge:
        return perms


def _set_common_charges_charge_to_JW_parity(sites, new_charges, new_mod):
    """Try to be clever and guess a new `charge_to_JW_parity` for `set_common_charges`.

    This will work for some cases, including an originally `new_charges='same'` or
    `new_charges='independent'`, but not in any case.

    If the user really needs `charge_to_JW_parity` and this function doesn't find it,
    they should just define it by hand...
    """
    # get new `charge_to_JW_parity` if possible
    c2JWps = [getattr(s, 'charge_to_JW_parity', None) for s in sites]
    if not all(p is not None for p in c2JWps):
        return None

    need = []
    for s, parities in enumerate(c2JWps):
        for old_i, p in enumerate(parities):
            if p != 0:
                need.append((1, s, old_i))
    if len(need) == 0:
        # no fermions at all, so trivial `charge_to_JW_parity`
        return np.array([0] * len(new_charges))

    need = set(need)   # can't have duplicates anyways; convert to set to compare without order
    new_charge_sets = []
    new_is = []
    for new_i, new_charge in enumerate(new_charges):
        m = new_mod[new_i]
        if m == 1 or m % 2 == 0:
            new_charge_set = set(new_charge)
            if new_charge_set == need:
                # got it: this new charge is just the total number of fermions
                charge_to_JW_parity = [0] * len(new_charges)
                charge_to_JW_parity[new_i] = 1
                return charge_to_JW_parity
            if new_charge_set <= need:
                new_charge_sets.append(new_charge_set)
                new_is.append(new_i)
            # else: has other quantum numbers
    # we don't have a single charge as the total fermion number
    # but maybe the charges are "independent" so we can just sum them up?
    charge_to_JW_parity = [0] * len(new_charges)
    # try to find partitioning of `need` with (subset of) new_charge_sets
    for new_i, new_charge_set in zip(new_is, new_charge_sets):
        if not new_charge_set <= need:
            continue
        charge_to_JW_parity[new_i] = 1
        need = need - new_charge_set
    if len(need) == 0:
        return np.array(charge_to_JW_parity, int)
    # else: couldn't partition at least with the greedy algorithm.
    return None


def kron(*ops, group=True):
    """Kronecker product of two or more local operators.

    Parameters
    ----------
    *ops : :class:`~tenpy.linalg.np_conserved.Array`
        Local operators with labels ``'p', 'p*'`` as defined in :class:`Site`.
    group : bool
        Whether to combine the in/outgoing legs.

    Returns
    -------
    product : :class:`~tenpy.linalg.np_conserved.Array`
        Outer product of the `ops`, with legs ``'p0', 'p0*', 'p1', 'p1*', ...`` (grouped=False)
        or combined legs ``'(p0.p1...)', '(p0*.p1*...)'`` (grouped=True).
    """
    if len(ops) <= 1:
        raise ValueError("need at least 2 ops")
    product = npc.outer(ops[0].replace_labels(['p', 'p*'], ['p0', 'p0*']),
                        ops[1].replace_labels(['p', 'p*'], ['p1', 'p1*']))
    for i in range(2, len(ops)):
        op = ops[i].replace_labels(['p', 'p*'], [f"p{i:d}", f"p{i:d}*"])
        product = npc.outer(product, op)
    if group:
        labels = [[f"p{i:d}" for i in range(len(ops))], [f"p{i:d}*" for i in range(len(ops))]]
        product = product.combine_legs(labels, qconj=[+1, -1])
    return product


# ------------------------------------------------------------------------------
# The most common local sites.


class SpinHalfSite(Site):
    r"""Spin-1/2 site.

    Local states are ``up`` (0) and ``down`` (1).
    Local operators are the usual spin-1/2 operators, e.g. ``Sz = [[0.5, 0.], [0., -0.5]]``,
    ``Sx = 0.5*sigma_x`` for the Pauli matrix `sigma_x`.

    =========================== ================================================
    operator                    description
    =========================== ================================================
    ``Id, JW``                  Identity :math:`\mathbb{1}`
    ``Sx, Sy, Sz``              Spin components :math:`S^{x,y,z}`,
                                equal to half the Pauli matrices.
    ``Sigmax, Sigmay, Sigmaz``  Pauli matrices :math:`\sigma^{x,y,z}`
    ``Sp, Sm``                  Spin flips :math:`S^{\pm} = S^{x} \pm i S^{y}`
    =========================== ================================================

    ============== ====  ============================
    `conserve`     qmod  *excluded* onsite operators
    ============== ====  ============================
    ``'Sz'``       [1]   ``Sx, Sy, Sigmax, Sigmay``
    ``'parity'``   [2]   --
    ``'None'``     []    --
    ============== ====  ============================

    Parameters
    ----------
    conserve : str | None
        Defines what is conserved, see table above.
    sort_charge : bool
        Whether :meth:`sort_charge` should be called at the end of initialization.
        This is usually a good idea to reduce potential overhead when using charge conservation.
        Note that this permutes the order of the local basis states!

    Attributes
    ----------
    conserve : str
        Defines what is conserved, see table above.
    """

    def __init__(self, conserve='Sz', sort_charge=True):
        if not conserve:
            conserve = 'None'
        if conserve not in ['Sz', 'parity', 'None']:
            raise ValueError("invalid `conserve`: " + repr(conserve))
        Sx = [[0., 0.5], [0.5, 0.]]
        Sy = [[0., -0.5j], [+0.5j, 0.]]
        Sz = [[0.5, 0.], [0., -0.5]]
        Sp = [[0., 1.], [0., 0.]]  # == Sx + i Sy
        Sm = [[0., 0.], [1., 0.]]  # == Sx - i Sy
        ops = dict(Sp=Sp, Sm=Sm, Sz=Sz)
        if conserve == 'Sz':
            chinfo = npc.ChargeInfo([1], ['2*Sz'])
            leg = npc.LegCharge.from_qflat(chinfo, [1, -1])
        else:
            # Added computational state projectors for PXP model - SAJANT
            ops.update(Sx=Sx, Sy=Sy, P0=np.eye(2)/2+Sz, P1=np.eye(2)/2-Sz)
            if conserve == 'parity':
                chinfo = npc.ChargeInfo([2], ['parity_Sz'])
                leg = npc.LegCharge.from_qflat(chinfo, [1, 0])  # ([1, -1] would need ``qmod=[4]``)
            else:
                leg = npc.LegCharge.from_trivial(2)
        self.conserve = conserve
        # Specify Hermitian conjugates
        Site.__init__(self, leg, ['up', 'down'], sort_charge=sort_charge, **ops)
        # further alias for state labels
        self.state_labels['-0.5'] = self.state_labels['down']
        self.state_labels['0.5'] = self.state_labels['up']
        # Add Pauli matrices
        if conserve != 'Sz':
            self.add_op('Sigmax', 2. * self.Sx)
            self.add_op('Sigmay', 2. * self.Sy)
        self.add_op('Sigmaz', 2. * self.Sz)
        self.charge_to_JW_parity = np.array([0] * leg.chinfo.qnumber, int)  # trivial

    def __repr__(self):
        """Debug representation of self."""
        return "SpinHalfSite({c!r})".format(c=self.conserve)


class SpinSite(Site):
    r"""General Spin S site.

    There are `2S+1` local states range from ``down`` (0)  to ``up`` (2S+1),
    corresponding to ``Sz=-S, -S+1, ..., S-1, S``.
    Local operators are the spin-S operators,
    e.g. ``Sz = [[0.5, 0.], [0., -0.5]]``,
    ``Sx = 0.5*sigma_x`` for the Pauli matrix `sigma_x`.

    ==============  ================================================
    operator        description
    ==============  ================================================
    ``Id, JW``      Identity :math:`\mathbb{1}`
    ``Sx, Sy, Sz``  Spin components :math:`S^{x,y,z}`,
                    equal to half the Pauli matrices.
    ``Sp, Sm``      Spin flips :math:`S^{\pm} = S^{x} \pm i S^{y}`
    ==============  ================================================

    ============== ====  ============================
    `conserve`     qmod  *excluded* onsite operators
    ============== ====  ============================
    ``'Sz'``       [1]   ``Sx, Sy, Sigmax, Sigmay``
    ``'parity'``   [2]   --
    ``'None'``     []    --
    ============== ====  ============================

    Parameters
    ----------
    conserve : str
        Defines what is conserved, see table above.
    sort_charge : bool
        Whether :meth:`sort_charge` should be called at the end of initialization.
        This is usually a good idea to reduce potential overhead when using charge conservation.
        Note that this permutes the order of the local basis states for ``conserve='parity'``!

    Attributes
    ----------
    S : {0.5, 1, 1.5, 2, ...}
        The 2S+1 states range from m = -S, -S+1, ... +S.
    conserve : str
        Defines what is conserved, see table above.
    """

    def __init__(self, S=0.5, conserve='Sz', sort_charge=True):
        if not conserve:
            conserve = 'None'
        if conserve not in ['Sz', 'parity', 'None']:
            raise ValueError("invalid `conserve`: " + repr(conserve))
        self.S = S = float(S)
        d = 2 * S + 1
        if d <= 1:
            raise ValueError("negative S?")
        if np.rint(d) != d:
            raise ValueError("S is not half-integer or integer")
        d = int(d)
        Sz_diag = -S + np.arange(d)
        Sz = np.diag(Sz_diag)
        Sp = np.zeros([d, d])
        for n in np.arange(d - 1):
            # Sp |m> =sqrt( S(S+1)-m(m+1)) |m+1>
            m = n - S
            Sp[n + 1, n] = np.sqrt(S * (S + 1) - m * (m + 1))
        Sm = np.transpose(Sp)
        # Sp = Sx + i Sy, Sm = Sx - i Sy
        Sx = (Sp + Sm) * 0.5
        Sy = (Sm - Sp) * 0.5j
        # Note: For S=1/2, Sy might look wrong compared to the Pauli matrix or SpinHalfSite.
        # Don't worry, I'm 99.99% sure it's correct (J. Hauschild)
        # The reason it looks wrong is simply that this class orders the states as ['down', 'up'],
        # while the usual spin-1/2 convention is ['up', 'down'], as you can also see if you look
        # at the Sz entries...
        # (The commutation relations are checked explicitly in `tests/test_site.py`)
        ops = dict(Sp=Sp, Sm=Sm, Sz=Sz)
        if conserve == 'Sz':
            chinfo = npc.ChargeInfo([1], ['2*Sz'])
            leg = npc.LegCharge.from_qflat(chinfo, np.array(2 * Sz_diag, dtype=np.int64))
        else:
            ops.update(Sx=Sx, Sy=Sy)
            if conserve == 'parity':
                chinfo = npc.ChargeInfo([2], ['parity_Sz'])
                leg = npc.LegCharge.from_qflat(chinfo, np.mod(np.arange(d), 2))
            else:
                leg = npc.LegCharge.from_trivial(d)
        self.conserve = conserve
        names = [str(i) for i in np.arange(-S, S + 1, 1.)]
        Site.__init__(self, leg, names, sort_charge=sort_charge, **ops)
        self.state_labels['down'] = self.state_labels[names[0]]
        self.state_labels['up'] = self.state_labels[names[-1]]
        self.charge_to_JW_parity = np.array([0] * leg.chinfo.qnumber, int)  # trivial

    def __repr__(self):
        """Debug representation of self."""
        return "SpinSite(S={S!s}, {c!r})".format(S=self.S, c=self.conserve)


class FermionSite(Site):
    r"""Create a :class:`Site` for spin-less fermions.

    Local states are ``empty`` and ``full``.

    .. warning ::
        Using the Jordan-Wigner string (``JW``) is crucial to get correct results,
        otherwise you just describe hardcore bosons!
        Further details in :doc:`/intro/JordanWigner`.

    ==============  ===================================================================
    operator        description
    ==============  ===================================================================
    ``Id``          Identity :math:`\mathbb{1}`
    ``JW``          Sign for the Jordan-Wigner string.
    ``C``           Annihilation operator :math:`c` (up to 'JW'-string left of it)
    ``Cd``          Creation operator :math:`c^\dagger` (up to 'JW'-string left of it)
    ``N``           Number operator :math:`n= c^\dagger c`
    ``dN``          :math:`\delta n := n - filling`
    ``dNdN``        :math:`(\delta n)^2`
    ==============  ===================================================================

    ============== ====  ===============================
    `conserve`     qmod  *excluded* onsite operators
    ============== ====  ===============================
    ``'N'``        [1]   --
    ``'parity'``   [2]   --
    ``'None'``     []    --
    ============== ====  ===============================

    Parameters
    ----------
    conserve : str
        Defines what is conserved, see table above.
    filling : float
        Average filling. Used to define ``dN``.

    Attributes
    ----------
    conserve : str
        Defines what is conserved, see table above.
    filling : float
        Average filling. Used to define ``dN``.
    """

    def __init__(self, conserve='N', filling=0.5):
        if not conserve:
            conserve = 'None'
        if conserve not in ['N', 'parity', 'None']:
            raise ValueError("invalid `conserve`: " + repr(conserve))
        JW = np.array([[1., 0.], [0., -1.]])
        C = np.array([[0., 1.], [0., 0.]])
        Cd = np.array([[0., 0.], [1., 0.]])
        N = np.array([[0., 0.], [0., 1.]])
        dN = np.array([[-filling, 0.], [0., 1. - filling]])
        dNdN = dN**2  # (element wise power is fine since dN is diagonal)
        ops = dict(JW=JW, C=C, Cd=Cd, N=N, dN=dN, dNdN=dNdN)
        if conserve == 'N':
            chinfo = npc.ChargeInfo([1], ['N'])
            leg = npc.LegCharge.from_qflat(chinfo, [0, 1])
            self.charge_to_JW_parity = np.array([1])
        elif conserve == 'parity':
            chinfo = npc.ChargeInfo([2], ['parity_N'])
            leg = npc.LegCharge.from_qflat(chinfo, [0, 1])
            self.charge_to_JW_parity = np.array([1])
        else:
            leg = npc.LegCharge.from_trivial(2)
            # no charge_to_JW_parity possible
        self.conserve = conserve
        self.filling = filling
        Site.__init__(self, leg, ['empty', 'full'], sort_charge=True, **ops)
        # specify fermionic operators
        self.need_JW_string |= set(['C', 'Cd', 'JW'])

    def __repr__(self):
        """Debug representation of self."""
        return "FermionSite({c!r}, {f:f})".format(c=self.conserve, f=self.filling)


class SpinHalfFermionSite(Site):
    r"""Create a :class:`Site` for spinful (spin-1/2) fermions.

    Local states are:
         ``empty``  (vacuum),
         ``up``     (one spin-up electron),
         ``down``   (one spin-down electron), and
         ``full``   (both electrons)

    Local operators can be built from creation operators.

    .. warning ::
        Using the Jordan-Wigner string (``JW``) in the correct way is crucial to get correct
        results, otherwise you just describe hardcore bosons!

    ==============  =============================================================================
    operator        description
    ==============  =============================================================================
    ``Id``          Identity :math:`\mathbb{1}`
    ``JW``          Sign for the Jordan-Wigner string :math:`(-1)^{n_{\uparrow}+n_{\downarrow}}`
    ``JWu``         Partial sign for the Jordan-Wigner string :math:`(-1)^{n_{\uparrow}}`
    ``JWd``         Partial sign for the Jordan-Wigner string :math:`(-1)^{n_{\downarrow}}`
    ``Cu``          Annihilation operator spin-up :math:`c_{\uparrow}`
                    (up to 'JW'-string on sites left of it).
    ``Cdu``         Creation operator spin-up :math:`c^\dagger_{\uparrow}`
                    (up to 'JW'-string on sites left of it).
    ``Cd``          Annihilation operator spin-down :math:`c_{\downarrow}`
                    (up to 'JW'-string on sites left of it).
                    Includes ``JWu`` such that it anti-commutes onsite with ``Cu, Cdu``.
    ``Cdd``         Creation operator spin-down :math:`c^\dagger_{\downarrow}`
                    (up to 'JW'-string on sites left of it).
                    Includes ``JWu`` such that it anti-commutes onsite with ``Cu, Cdu``.
    ``Nu``          Number operator :math:`n_{\uparrow}= c^\dagger_{\uparrow} c_{\uparrow}`
    ``Nd``          Number operator :math:`n_{\downarrow}= c^\dagger_{\downarrow} c_{\downarrow}`
    ``NuNd``        Dotted number operators :math:`n_{\uparrow} n_{\downarrow}`
    ``Ntot``        Total number operator :math:`n_t= n_{\uparrow} + n_{\downarrow}`
    ``dN``          Total number operator compared to the filling :math:`\Delta n = n_t-filling`
    ``Sx, Sy, Sz``  Spin operators :math:`S^{x,y,z}`, in particular
                    :math:`S^z = \frac{1}{2}( n_\uparrow - n_\downarrow )`
    ``Sp, Sm``      Spin flips :math:`S^{\pm} = S^{x} \pm i S^{y}`,
                    e.g. :math:`S^{+} = c^\dagger_\uparrow c_\downarrow`
    ==============  =============================================================================

    The spin operators are defined as :math:`S^\gamma =
    (c^\dagger_{\uparrow}, c^\dagger_{\downarrow}) \sigma^\gamma (c_{\uparrow}, c_{\downarrow})^T`,
    where :math:`\sigma^\gamma` are spin-1/2 matrices (i.e. half the pauli matrices).

    ============= ============= ======= =======================================
    `cons_N`      `cons_Sz`     qmod    *excluded* onsite operators
    ============= ============= ======= =======================================
    ``'N'``       ``'Sz'``      [1, 1]  ``Sx, Sy``
    ``'N'``       ``'parity'``  [1, 4]  --
    ``'N'``       ``None``      [1]     --
    ``'parity'``  ``'Sz'``      [2, 1]  ``Sx, Sy``
    ``'parity'``  ``'parity'``  [2, 4]  --
    ``'parity'``  ``None``      [2]     --
    ``None``      ``'Sz'``      [1]     ``Sx, Sy``
    ``None``      ``'parity'``  [4]     --
    ``None``      ``None``      []      --
    ============= ============= ======= =======================================

    Parameters
    ----------
    cons_N : ``'N' | 'parity' | None``
        Whether particle number is conserved, c.f. table above.
    cons_Sz : ``'Sz' | 'parity' | None``
        Whether spin is conserved, c.f. table above.
    filling : float
        Average filling. Used to define ``dN``.

    Attributes
    ----------
    cons_N : ``'N' | 'parity' | None``
        Whether particle number is conserved, c.f. table above.
    cons_Sz : ``'Sz' | 'parity' | None``
        Whether spin is conserved, c.f. table above.
    filling : float
        Average filling. Used to define ``dN``.
    """

    def __init__(self, cons_N='N', cons_Sz='Sz', filling=1.):
        if not cons_N:
            cons_N = 'None'
        if cons_N not in ['N', 'parity', 'None']:
            raise ValueError("invalid `cons_N`: " + repr(cons_N))
        if not cons_Sz:
            cons_Sz = 'None'
        if cons_Sz not in ['Sz', 'parity', 'None']:
            raise ValueError("invalid `cons_Sz`: " + repr(cons_Sz))
        d = 4
        states = ['empty', 'up', 'down', 'full']
        # 0) Build the operators.
        Nu_diag = np.array([0., 1., 0., 1.], dtype=np.float64)
        Nd_diag = np.array([0., 0., 1., 1.], dtype=np.float64)
        Nu = np.diag(Nu_diag)
        Nd = np.diag(Nd_diag)
        Ntot = np.diag(Nu_diag + Nd_diag)
        dN = np.diag(Nu_diag + Nd_diag - filling)
        NuNd = np.diag(Nu_diag * Nd_diag)
        JWu = np.diag(1. - 2 * Nu_diag)  # (-1)^Nu
        JWd = np.diag(1. - 2 * Nd_diag)  # (-1)^Nd
        JW = JWu * JWd  # (-1)^{Nu+Nd}

        Cu = np.zeros((d, d))
        Cu[0, 1] = Cu[2, 3] = 1
        Cdu = np.transpose(Cu)
        # For spin-down annihilation operator: include a Jordan-Wigner string JWu
        # this ensures that Cdu.Cd = - Cd.Cdu
        # c.f. the chapter on the Jordan-Wigner trafo in the userguide
        Cd_noJW = np.zeros((d, d))
        Cd_noJW[0, 2] = Cd_noJW[1, 3] = 1
        Cd = np.dot(JWu, Cd_noJW)  # (don't do this for spin-up...)
        Cdd = np.transpose(Cd)

        # spin operators are defined as  (Cdu, Cdd) S^gamma (Cu, Cd)^T,
        # where S^gamma is the 2x2 matrix for spin-half
        Sz = np.diag(0.5 * (Nu_diag - Nd_diag))
        Sp = np.dot(Cdu, Cd)
        Sm = np.dot(Cdd, Cu)
        Sx = 0.5 * (Sp + Sm)
        Sy = -0.5j * (Sp - Sm)

        ops = dict(JW=JW, JWu=JWu, JWd=JWd,
                   Cu=Cu, Cdu=Cdu, Cd=Cd, Cdd=Cdd,
                   Nu=Nu, Nd=Nd, Ntot=Ntot, NuNd=NuNd, dN=dN,
                   Sx=Sx, Sy=Sy, Sz=Sz, Sp=Sp, Sm=Sm)  # yapf: disable

        # handle charges
        qmod = []
        qnames = []
        charges = []
        if cons_N == 'N':
            qnames.append('N')
            qmod.append(1)
            charges.append([0, 1, 1, 2])
        elif cons_N == 'parity':
            qnames.append('parity_N')
            qmod.append(2)
            charges.append([0, 1, 1, 0])
        if cons_Sz == 'Sz':
            qnames.append('2*Sz')  # factor 2 s.t. Cu, Cd have well-defined charges!
            qmod.append(1)
            charges.append([0, 1, -1, 0])
            del ops['Sx']
            del ops['Sy']
        elif cons_Sz == 'parity':
            qnames.append('parity_Sz')  # the charge is (2*Sz) mod (2*2)
            qmod.append(4)
            charges.append([0, 1, 3, 0])  # == [0, 1, -1, 0] mod 4
            # e.g. terms like `Sp_i Sp_j + hc` with Sp=Cdu Cd have charges 'N', 'parity_Sz'.
            # The `parity_Sz` is non-trivial in this case!
        if len(qmod) == 0:
            leg = npc.LegCharge.from_trivial(d)
        else:
            if len(qmod) == 1:
                charges = charges[0]
            else:  # len(charges) == 2: need to transpose
                charges = [[q1, q2] for q1, q2 in zip(charges[0], charges[1])]
            chinfo = npc.ChargeInfo(qmod, qnames)
            leg = npc.LegCharge.from_qflat(chinfo, charges)
        self.cons_N = cons_N
        self.cons_Sz = cons_Sz
        self.filling = filling
        Site.__init__(self, leg, states, sort_charge=True, **ops)
        # specify fermionic operators
        self.need_JW_string |= set(['Cu', 'Cdu', 'Cd', 'Cdd', 'JWu', 'JWd', 'JW'])
        if cons_N == 'N' or cons_N == 'parity':
            self.charge_to_JW_parity = np.array([1] + [0]*(len(qnames) - 1))
        # else: can't define charge_to_JW_parity

    def __repr__(self):
        """Debug representation of self."""
        return "SpinHalfFermionSite({cN!r}, {cS!r}, {f:f})".format(cN=self.cons_N,
                                                                   cS=self.cons_Sz,
                                                                   f=self.filling)


class SpinHalfHoleSite(Site):
    r"""Create a :class:`Site` for spinful (spin-1/2) fermions, restricted to empty or singly occupied sites

    Local states are:
         ``empty``  (vacuum),
         ``up``     (one spin-up electron),
         ``down``   (one spin-down electron)

    Local operators can be built from creation operators.

    .. warning ::
        Using the Jordan-Wigner string (``JW``) in the correct way is crucial to get correct
        results, otherwise you just describe hardcore bosons!

    ==============  =============================================================================
    operator        description
    ==============  =============================================================================
    ``Id``          Identity :math:`\mathbb{1}`
    ``JW``          Sign for the Jordan-Wigner string :math:`(-1)^{n_{\uparrow}+n_{\downarrow}}`
    ``JWu``         Partial sign for the Jordan-Wigner string :math:`(-1)^{n_{\uparrow}}`
    ``JWd``         Partial sign for the Jordan-Wigner string :math:`(-1)^{n_{\downarrow}}`
    ``Cu``          Annihilation operator spin-up :math:`c_{\uparrow}`
                    (up to 'JW'-string on sites left of it).
    ``Cdu``         Creation operator spin-up :math:`c^\dagger_{\uparrow}`
                    (up to 'JW'-string on sites left of it).
    ``Cd``          Annihilation operator spin-down :math:`c_{\downarrow}`
                    (up to 'JW'-string on sites left of it).
                    Includes ``JWu`` such that it anti-commutes onsite with ``Cu, Cdu``.
    ``Cdd``         Creation operator spin-down :math:`c^\dagger_{\downarrow}`
                    (up to 'JW'-string on sites left of it).
                    Includes ``JWu`` such that it anti-commutes onsite with ``Cu, Cdu``.
    ``Nu``          Number operator :math:`n_{\uparrow}= c^\dagger_{\uparrow} c_{\uparrow}`
    ``Nd``          Number operator :math:`n_{\downarrow}= c^\dagger_{\downarrow} c_{\downarrow}`
    ``Ntot``        Total number operator :math:`n_t= n_{\uparrow} + n_{\downarrow}`
    ``dN``          Total number operator compared to the filling :math:`\Delta n = n_t-filling`
    ``Sx, Sy, Sz``  Spin operators :math:`S^{x,y,z}`, in particular
                    :math:`S^z = \frac{1}{2}( n_\uparrow - n_\downarrow )`
    ``Sp, Sm``      Spin flips :math:`S^{\pm} = S^{x} \pm i S^{y}`,
                    e.g. :math:`S^{+} = c^\dagger_\uparrow c_\downarrow`
    ==============  =============================================================================

    The spin operators are defined as :math:`S^\gamma =
    (c^\dagger_{\uparrow}, c^\dagger_{\downarrow}) \sigma^\gamma (c_{\uparrow}, c_{\downarrow})^T`,
    where :math:`\sigma^\gamma` are spin-1/2 matrices (i.e. half the pauli matrices).

    ============= ============= ======= =======================================
    `cons_N`      `cons_Sz`     qmod    *excluded* onsite operators
    ============= ============= ======= =======================================
    ``'N'``       ``'Sz'``      [1, 1]  ``Sx, Sy``
    ``'N'``       ``'parity'``  [1, 4]  --
    ``'N'``       ``None``      [1]     --
    ``'parity'``  ``'Sz'``      [2, 1]  ``Sx, Sy``
    ``'parity'``  ``'parity'``  [2, 4]  --
    ``'parity'``  ``None``      [2]     --
    ``None``      ``'Sz'``      [1]     ``Sx, Sy``
    ``None``      ``'parity'``  [4]     --
    ``None``      ``None``      []      --
    ============= ============= ======= =======================================

    Parameters
    ----------
    cons_N : ``'N' | 'parity' | None``
        Whether particle number is conserved, c.f. table above.
    cons_Sz : ``'Sz' | 'parity' | None``
        Whether spin is conserved, c.f. table above.
    filling : float
        Average filling. Used to define ``dN``.

    Attributes
    ----------
    cons_N : ``'N' | 'parity' | None``
        Whether particle number is conserved, c.f. table above.
    cons_Sz : ``'Sz' | 'parity' | None``
        Whether spin is conserved, c.f. table above.
    filling : float
        Average filling. Used to define ``dN``.
    """

    def __init__(self, cons_N='N', cons_Sz='Sz', filling=1.):
        if not cons_N:
            cons_N = 'None'
        if cons_N not in ['N', 'parity', 'None']:
            raise ValueError("invalid `cons_N`: " + repr(cons_N))
        if not cons_Sz:
            cons_Sz = 'None'
        if cons_Sz not in ['Sz', 'parity', 'None']:
            raise ValueError("invalid `cons_Sz`: " + repr(cons_Sz))
        d = 3
        states = ['empty', 'up', 'down']
        # 0) Build the operators.
        Nu_diag = np.array([0., 1., 0.], dtype=np.float64)
        Nd_diag = np.array([0., 0., 1.], dtype=np.float64)
        Nu = np.diag(Nu_diag)
        Nd = np.diag(Nd_diag)
        Ntot = np.diag(Nu_diag + Nd_diag)
        dN = np.diag(Nu_diag + Nd_diag - filling)
        JWu = np.diag(1. - 2 * Nu_diag)  # (-1)^Nu
        JWd = np.diag(1. - 2 * Nd_diag)  # (-1)^Nd
        JW = JWu * JWd  # (-1)^{Nu+Nd}

        Cu = np.zeros((d, d))
        Cu[0, 1] = 1
        Cdu = np.transpose(Cu)
        # For spin-down annihilation operator: include a Jordan-Wigner string JWu
        # this ensures that Cdu.Cd = - Cd.Cdu
        # c.f. the chapter on the Jordan-Wigner trafo in the userguide
        Cd_noJW = np.zeros((d, d))
        Cd_noJW[0, 2] = 1
        Cd = np.dot(JWu, Cd_noJW)  # (don't do this for spin-up...)
        Cdd = np.transpose(Cd)

        # spin operators are defined as  (Cdu, Cdd) S^gamma (Cu, Cd)^T,
        # where S^gamma is the 2x2 matrix for spin-half
        Sz = np.diag(0.5 * (Nu_diag - Nd_diag))
        Sp = np.dot(Cdu, Cd)
        Sm = np.dot(Cdd, Cu)
        Sx = 0.5 * (Sp + Sm)
        Sy = -0.5j * (Sp - Sm)

        ops = dict(JW=JW, JWu=JWu, JWd=JWd,
                   Cu=Cu, Cdu=Cdu, Cd=Cd, Cdd=Cdd,
                   Nu=Nu, Nd=Nd, Ntot=Ntot, dN=dN,
                   Sx=Sx, Sy=Sy, Sz=Sz, Sp=Sp, Sm=Sm)  # yapf: disable

        # handle charges
        qmod = []
        qnames = []
        charges = []
        if cons_N == 'N':
            qnames.append('N')
            qmod.append(1)
            charges.append([0, 1, 1])
        elif cons_N == 'parity':
            qnames.append('parity_N')
            qmod.append(2)
            charges.append([0, 1, 1])
        if cons_Sz == 'Sz':
            qnames.append('2*Sz')  # factor 2 s.t. Cu, Cd have well-defined charges!
            qmod.append(1)
            charges.append([0, 1, -1])
            del ops['Sx']
            del ops['Sy']
        elif cons_Sz == 'parity':
            qnames.append('parity_Sz')  # the charge is (2*Sz) mod (2*2)
            qmod.append(4)
            charges.append([0, 1, 3])  # == [0, 1, -1, 0] mod 4
            # e.g. terms like `Sp_i Sp_j + hc` with Sp=Cdu Cd have charges 'N', 'parity_Sz'.
            # The `parity_Sz` is non-trivial in this case!
        if len(qmod) == 0:
            leg = npc.LegCharge.from_trivial(d)
        else:
            if len(qmod) == 1:
                charges = charges[0]
            else:  # len(charges) == 2: need to transpose
                charges = [[q1, q2] for q1, q2 in zip(charges[0], charges[1])]
            chinfo = npc.ChargeInfo(qmod, qnames)
            leg = npc.LegCharge.from_qflat(chinfo, charges)
        self.cons_N = cons_N
        self.cons_Sz = cons_Sz
        self.filling = filling
        Site.__init__(self, leg, states, sort_charge=True, **ops)
        # specify fermionic operators
        self.need_JW_string |= set(['Cu', 'Cdu', 'Cd', 'Cdd', 'JWu', 'JWd', 'JW'])

        if cons_N == 'N' or cons_N == 'parity':
            self.charge_to_JW_parity = np.array([1] + [0]*(len(qnames) - 1))
        # else: can't define charge_to_JW_parity


    def __repr__(self):
        """Debug representation of self."""
        return "SpinHalfHoleSite({cN!r}, {cS!r}, {f:f})".format(cN=self.cons_N,
                                                                   cS=self.cons_Sz,
                                                                   f=self.filling)


class BosonSite(Site):
    r"""Create a :class:`Site` for up to `Nmax` bosons.

    Local states are ``vac, 1, 2, ... , Nmax``.
    (Exception: for parity conservation, we sort as ``vac, 2, 4, ..., 1, 3, 5, ...``.)

    ==============  ========================================
    operator        description
    ==============  ========================================
    ``Id, JW``      Identity :math:`\mathbb{1}`
    ``B``           Annihilation operator :math:`b`
    ``Bd``          Creation operator :math:`b^\dagger`
    ``N``           Number operator :math:`n= b^\dagger b`
    ``NN``          :math:`n^2`
    ``dN``          :math:`\delta n := n - filling`
    ``dNdN``        :math:`(\delta n)^2`
    ``P``           Parity :math:`Id - 2 (n \mod 2)`.
    ==============  ========================================

    ============== ====  ==================================
    `conserve`     qmod  *excluded* onsite operators
    ============== ====  ==================================
    ``'N'``        [1]   --
    ``'parity'``   [2]   --
    ``'None'``     []    --
    ============== ====  ==================================

    Parameters
    ----------
    Nmax : int
        Cutoff defining the maximum number of bosons per site.
        The default ``Nmax=1`` describes hard-core bosons.
    conserve : str
        Defines what is conserved, see table above.
    filling : float
        Average filling. Used to define ``dN``.

    Attributes
    ----------
    conserve : str
        Defines what is conserved, see table above.
    filling : float
        Average filling. Used to define ``dN``.
    """

    def __init__(self, Nmax=1, conserve='N', filling=0.):
        if not conserve:
            conserve = 'None'
        if conserve not in ['N', 'parity', 'None']:
            raise ValueError("invalid `conserve`: " + repr(conserve))
        dim = Nmax + 1
        states = [str(n) for n in range(0, dim)]
        if dim < 2:
            raise ValueError("local dimension should be larger than 1....")
        B = np.zeros([dim, dim], dtype=np.float64)  # destruction/annihilation operator
        for n in range(1, dim):
            B[n - 1, n] = np.sqrt(n)
        Bd = np.transpose(B)  # .conj() wouldn't do anything
        # Note: np.dot(Bd, B) has numerical roundoff errors of eps~=4.4e-16.
        Ndiag = np.arange(dim, dtype=np.float64)
        N = np.diag(Ndiag)
        NN = np.diag(Ndiag**2)
        dN = np.diag(Ndiag - filling)
        dNdN = np.diag((Ndiag - filling)**2)
        P = np.diag(1. - 2. * np.mod(Ndiag, 2))
        ops = dict(B=B, Bd=Bd, N=N, NN=NN, dN=dN, dNdN=dNdN, P=P)
        if conserve == 'N':
            chinfo = npc.ChargeInfo([1], ['N'])
            leg = npc.LegCharge.from_qflat(chinfo, range(dim))
        elif conserve == 'parity':
            chinfo = npc.ChargeInfo([2], ['parity_N'])
            leg = npc.LegCharge.from_qflat(chinfo, [i % 2 for i in range(dim)])
        else:
            leg = npc.LegCharge.from_trivial(dim)
        self.Nmax = Nmax
        self.conserve = conserve
        self.filling = filling
        Site.__init__(self, leg, states, sort_charge=True, **ops)
        self.state_labels['vac'] = self.state_labels['0']  # alias
        self.charge_to_JW_parity = np.array([0] * leg.chinfo.qnumber, int)  # trivial

    def __repr__(self):
        """Debug representation of self."""
        return "BosonSite({N:d}, {c!r}, {f:f})".format(N=self.Nmax,
                                                       c=self.conserve,
                                                       f=self.filling)


def spin_half_species(SpeciesSite, cons_N, cons_Sz, **kwargs):
    """Initialize two FermionSite to represent spin-1/2 species.

    You can use this directly in the :meth:`tenpy.models.model.CouplingMPOModel.init_sites`,
    e.g., as in the :meth:`tenpy.models.hubbard.FermiHubbardModel2.init_sites`::

        cons_N = model_params.get('cons_N', 'N', str)
        cons_Sz = model_params.get('cons_Sz', 'Sz', str)
        return spin_half_species(FermionSite, cons_N=cons_N, cons_Sz=cons_Sz)

    Parameters
    ----------
    SpeciesSite : :class:`Site` | str
        The (name of the) site class for the species;
        usually just :class:`FermionSite`.
    cons_N : None | ``"N", "parity", "None"``
        Whether to conserve the (parity of the) total particle number ``N_up + N_down``.
    cons_Sz : None | ``"Sz", "parity", "None"``
        Whether to conserve the (parity of the) total Sz spin ``N_up - N_down``.

    Returns
    -------
    sites : list of `SpeciesSite`
        Each one instance of the site for spin up and down.
    species_names : list of str
        Always ``['up', 'down']``. Included such that a ``return spin_half_species(...)``
        in :meth:`~tenpy.models.model.CouplingMPOModel.init_sites` triggers the use of the
        :class:`~tenpy.models.lattice.MultiSpeciesLattice`.
    """
    SpeciesSite = find_subclass(Site, SpeciesSite)
    if not cons_N:
        cons_N = 'None'
    if cons_N not in ['N', 'parity', 'None']:
        raise ValueError("invalid `cons_N`: " + repr(cons_N))
    if not cons_Sz:
        cons_Sz = 'None'
    if cons_Sz not in ['Sz', 'parity', 'None']:
        raise ValueError("invalid `cons_Sz`: " + repr(cons_Sz))

    conserve = None if cons_N == 'None' and cons_Sz == 'None' else 'N'

    up_site = SpeciesSite(conserve=conserve, **kwargs)
    down_site = SpeciesSite(conserve=conserve, **kwargs)

    new_charges = []
    new_names = []
    new_mod = []
    if cons_N == 'N':
        new_charges.append([(1, 0, 0), (1, 1, 0)])
        new_names.append('N')
        new_mod.append(1)
    elif cons_N == 'parity':
        new_charges.append([(1, 0, 0), (1, 1, 0)])
        new_names.append('parity_N')
        new_mod.append(2)
    if cons_Sz == 'Sz':
        new_charges.append([(1, 0, 0), (-1, 1, 0)])
        new_names.append('2*Sz')  # factor 2 s.t. Cu, Cd have well-defined charges!
        new_mod.append(1)
    elif cons_Sz == 'parity':
        new_charges.append([(1, 0, 0), (-1, 1, 0)])
        new_names.append('2*Sz')  # factor 2 s.t. Cu, Cd have well-defined charges!
        new_mod.append(4)
    set_common_charges([up_site, down_site], new_charges, new_names, new_mod)
    return [up_site, down_site], ['up', 'down']


class ClockSite(Site):
    r"""Quantum clock site.

    There are ``q`` local states, with labels ``['0', '1', ..., str(q-1)]``.
    Special aliases are ``up`` (0), and if q is even ``down`` (q / 2).
    Local operators are the clock operators ``Z = diag([w ** 0, w ** 1, ..., w ** (q - 1)])``
    with ``w = exp(2.j * pi / q)`` and ``X = eye(q, k=1) + eye(q, k=1-q)``, which are not hermitian (!)

    =========================== ================================================
    operator                    description
    =========================== ================================================
    ``Id, JW``                  Identity :math:`\mathbb{1}`
    ``X, Z``                    Clock operators
    ``Xhc, Zhc``                Hermitian conjugates of clock operators
    ``Xphc, Zphc``              Clock operator plus its hermitian conjugate
    =========================== ================================================

    ============== ====  ============================
    `conserve`     qmod  *excluded* onsite operators
    ============== ====  ============================
    ``'Z'``        [q]   ``Xphc, Zphc``
    ``'None'``     []    --
    ============== ====  ============================

    Parameters
    ----------
    q : int
        Number of states per site
    conserve : str | None
        Defines what is conserved, see table above.
    sort_charge : bool
        Whether :meth:`sort_charge` should be called at the end of initialization.
        This is usually a good idea to reduce potential overhead when using charge conservation.
        Note that this permutes the order of the local basis states!

    Attributes
    ----------
    q : int
        Number of states per site
    conserve : str
        Defines what is conserved, see table above.
    """
    def __init__(self, q, conserve='Z', sort_charge=True):
        if not (isinstance(q, int) and q > 1):
            raise ValueError(f'invalid q: {q}')
        self.q = q
        if not conserve:
            conserve = 'None'
        if conserve not in ['Z', 'None']:
            raise ValueError("invalid `conserve`: " + repr(conserve))
        X = np.eye(q, k=1) + np.eye(q, k=1-q)
        Z = np.diag(np.exp(2.j * np.pi * np.arange(q, dtype=np.complex128) / q))
        Xhc = X.conj().transpose()
        Zhc = Z.conj().transpose()
        Xphc = X + Xhc
        Zphc = np.diag(2. * np.cos(2. * np.pi * np.arange(q, dtype=np.complex128) / q))
        if conserve == 'Z':
            # we store n as the charge where <Z> = exp(2.j * pi * n / q)
            chinfo = npc.ChargeInfo([q], ['clock_phase'])
            leg = npc.LegCharge.from_qflat(chinfo, list(range(q)))
        else:
            leg = npc.LegCharge.from_trivial(q)
        self.conserve = conserve
        names = [str(m) for m in range(q)]
        Site.__init__(self, leg, names, sort_charge=sort_charge)
        self.add_op('X', X, hc='Xhc')
        self.add_op('Xhc', Xhc, hc='X')
        self.add_op('Z', Z, hc='Zhc')
        self.add_op('Zhc', Zhc, hc='Z')
        if conserve != 'Z':
            self.add_op('Xphc', Xphc, hc='Xphc')
            self.add_op('Zphc', Zphc, hc='Zphc')
        self.state_labels['up'] = self.state_labels['0']
        if q % 2 == 0:
            self.state_labels['down'] = self.state_labels[str(q // 2)]

    def __repr__(self):
        return f'ClockSite(q={self.q}, conserve={self.conserve})'

class DoubledSite(Site):
    r"""Doubled site for Heisenberg or density matrix evolution.

    Given some physical Hilbert space of dimenison $d$, we want to define the vectorized,
    doubled Hilbert space of dimension $d^2$. Often we might want to specify the opertor as
    a set of operators rather than ket-bra basis elements. This allows us to perform DMT
    by picking out specific operators. We allow for 3 types of basis, specified by the function
    parameters `trivial` and `hermitian`.

    1. `trivial=True`, `hermitian=False` - standard ket-bra basis |i><j|. These are not
    operators we typically consider. This is the basis one gets by trivially vectorizing
    a density matrix.
    2. `trivial=False`, `hermitian=True` - basis of Hermitian operators, such as the Pauli
    basis {I, X, Y, Z} for d=2. For d > 2, this will be generalizations of the Pauli matrices.
    Only one operator (I) will have unit trace while the rest will be traceless.
    3. `trivial=False`, `hermitian=False` - basis of operators for U(1) (or parity) charge
    conservation. For d = 2, this corresponds to {I, Z, S+, S-}. For d > 2, we have analogous
    operators that correspond to powers of raising and lowering operators.

    In each basis, we define the set of operators as {sigma_alpha}. We then vectorize these
    operators and build a matrix using these vectors as columns. Call this matrix OP
    (operators). What is the meaning of OP? OP|i> is the vectorized representation of a
    state specified by operator sigma_i, for |i> a unit basis vector. OP for the trivial basis
    is simply the identity matrix; the unit basis vectors simply represent |i><j| once vectorized.

    Suppose we have a site in the computational basis spanned by |i>. The density matrix is
    spanned by |i><j| which gives |i,j>> when vectorized with OP = I. The state of the density
    matrix is given by |rho>> = \sum_{i,j} c_{i,j} |i,j>>. Suppose we want to work in some other
    basis OP. We thus want to find the vector |s> such that |rho>> = OP |s>. |s> specifies the
    linear combination of operators of OP that give the state |rho>>. However, OP isn't
    necessarily unitary, so we can't use it as a change of basis matrix. So instead, we
    decompose OP = QR. Then Q is an orthogonal version of OP. We use Q to do the basis transforms.
    So in the orthogonal basis, |t> = R |s> = Q^\dagger |rho>> or |rho>> = Q |t>

    Suppose we have an initial vector |rho>> in the vectorized |i><j| basis and we want the
    expectation value with respect to operator P; i.e. <<rho| P |rho>>. Once we rotate this,
    we find <<rho| P |rho> = <t| Q^\dagger P Q |t>.

    ---------------- OLD COMMENTS --------------------------
    We want to define a basis of Hermitian,
    orthogonal, and mostly traceless (HOMT) operators for use in DMT and DAOE.
    What do these words mean?
        (1) Hermitian - this is self-evident.
        (2) orthogonal - given two operators $\sigma_\alpha, \sigma_\beta$, we demand that
        $\langle \langle \sigma_\alpha | \sigma_\beta \rangle \rangle = Tr (\sigma_\alpha^\dagger \sigma_\beta) = \delta_{\alpha, \beta} $.
        (3) Mostly traceless - only the Identity operator has a non-zero trace (and by orthogonality).

    How to we find such a basis? First, define a basis of $d^2$ Hermitian, mostly traceless (HMT)
    (but not orthogonal) operators. Choose op[0] = Id, op[1 <= i < d] = |0><0| - |i><i|,
    op[d <= d + d(d-1)/2] = \sum_{i<j} |i><j| + |j><i|,
    op[d + d(d-1)/2 <= d^2] = \sum_{i<j} i|i><j| - i|j><i|

    These are just a basis of operators in the standard bra-ket basis. These are not HOMT, but we
    will make them so via a rotation and rescaling. Suppose we have a vector $\vec{v}$ in the standard
    bra-ket basis, where $\vec{v}$ represents a Hermitian density matrix or Hermitian operator.
    To write $\vec{v}$ as a linear combination of our HMT operators defined above, note that
    $\vec{v} = OP \vec{s}$.

    But we want the coefficients of $\vec{v}$ in a HOMT basis, so define $OP = QR$ and gauge-fix
    the $R$ such that all of the diagonals are positive. Then $Q$ defines an HOMT basis.
    So then let $\vec{v} = Q \vec{\lambda}$ where $\vec{\lambda} = R \vec{s}$. So $\vec{\lambda}$
    defines how to make $\vec{v}$ as a linear combinations of elements of the HOMT $Q$ basis.

    Finally, given an operator $M$ that acts on the doubled Hilbert space vector $\vec{v}$, we map this
    operator to the HOMT basis $Q$ by $M \rightarrow Q^\dagger M Q$. Then $\vec{v}^\dagger M \vec{v}
    = \vec{w}^\dagger Q^\dagger M Q \vec{w}$.

    Let us give an example for $d=2$. The Pauli operators $I, Z, Y, X$ define an HOMT basis already.
    The HMT operators we typically define are `already` the Pauli operators. So $OP = Q * 1$ is already
    unitary (the hallmark of an HOMT basis). The $R$ matrix is trivial. Then, to map from the
    computation, bra-ket basis to the basis in which index 0,1,2,3 corresponds to operators I, Z, X, Y,
    one must use $Q$.

    For the general case, $R$ is non-trivial. While it seem paradoxical that we don't actually use $R$
    anywhere, we could by first finding $\vec{s} = OP^{-1} \vec{v}$ and then $\vec{\lambda} = R \vec{s}$,
    but this is equivalent to $\vec{\lambda} = Q^\dagger \vec{v}$.

    =========================== ================================================
    operator                    description
    =========================== ================================================
    We don't define operators as the usual physical operators can be used once
    we rotate back to the original, bra-ket basis.
    =========================== ================================================

    ============== ====  ============================
    `conserve`     qmod  *excluded* onsite operators
    ============== ====  ============================
    ``'None'``     []    --
    ============== ====  ============================

    Parameters
    ----------
    d : int
        Number of states per site in the physical (undoubled) Hilbert space
    conserve : str | None
        Defines what is conserved, see table above.
    sort_charge : bool
        Whether :meth:`sort_charge` should be called at the end of initialization.
        This is usually a good idea to reduce potential overhead when using charge conservation.
        Note that this permutes the order of the local basis states!
        For backwards compatibility with existing data, it is not (yet) enabled by default.

    Attributes
    ----------
    conserve : str
        Defines what is conserved, see table above.
    """

    # Generalize this to double ANY type of site. Need to take in kwargs for the desired type
    # of site and pass them on to the class initialization. This will be needed for bosons.
    def __init__(self, d, conserve=None, sort_charge=True, hermitian=True, trivial=False):
        if not conserve:
            conserve = 'None'
        if conserve not in ['None', 'Sz', 'parity']:
            raise ValueError("invalid `conserve`: " + repr(conserve))
        if hermitian:
            assert conserve == 'None'
            assert not trivial
        if trivial:
            # We can conserve charge with a trivial site
            assert not hermitian
        self.d = d # Dimension of original Hilbert space
        self.hermitian = hermitian
        self.trivial = trivial

        # sort_charge=False as the sorting should be done later
        ss_op = SpinSite(S=(d-1)/2, conserve=conserve, sort_charge=False)
        # Bunch doesn't matter if we don't sort since the neighboring charges are not the same
        # If we are to sort the charges, it should be done now so that we still have a leg pipe.
        # If we just sort at the end when calling the Site initializer, the leg pipe will get replaced
        # with a leg charge, meaning that we cannot split the leg later.
        leg1 = npc.LegPipe([ss_op.leg, ss_op.leg.conj()], qconj=+1, sort=sort_charge, bunch=True)
        # leg1._perm is the index of where the charge GOES; i.e. _perm[0] is the new index of charges[0] after sorting
        if conserve != 'None':
            # The site should record the permutation done.
            self.perm = leg1._perm
            # perm = array([0, 5, 1, 6, 2, 7, 3, 8, 4]) for d=3, 'parity'
            # We order the indices as [0, 2, 4, 6, 8, 1, 3, 5, 7]
            # charges (below) are [0, 1, 0, 1, 0, 1, 0, 1, 0]
        if trivial:
            # Don't rotate the basis at all. The basis is simply |i><j|

            OP_flat = np.eye(d**2)
            self.OP_ops = OP_ops = []
            self.charges = charges = []

            for i in range(d**2):
                OP_ops.append(npc.Array.from_ndarray(OP_flat[:,i].reshape(d,d), [ss_op.leg, ss_op.leg.conj()], dtype=np.complex128, labels=['p', 'p*']))
                if conserve != 'None':
                    charges.append(OP_ops[-1].qtotal.item())
            if conserve != 'None':
                inverted_perm = np.argsort(leg1._perm)
                # This will cause an error since there is ambiguity in ordering indices correspondong to the same charge.
                assert np.all(inverted_perm == np.argsort(charges))
            else:
                inverted_perm = np.arange(d**2)
            self.identity_ind = [j+j*d for j in range(d)]   # multiple operators with trace; d of them in fact.
            # We must reorder the OP array so that the charges work out; otherwise the charge structure is incompatible.
            self.OP = npc.Array.from_ndarray(OP_flat[inverted_perm[:,None],inverted_perm[None,:]], [leg1, leg1.conj()], dtype=np.complex128, qtotal=None, labels=['p', 'p*'])
        else:
            if not hermitian:
                # Choose a basis of I, |0><0| - |i><i|, and |i><j!=i|.
                # All of the operators will have define U(1) charge.

                def build_COB(d):
                    """
                    Each column will represent an operator once it has been vectorized.
                    For d = 1, charges_0s = [0, 3]. So the first column is I and the last is Z.
                    """
                    COB = np.eye(d**2)
                    charge_0s = np.array([j+j*d for j in range(d)])
                    COB[charge_0s,charge_0s] = 0
                    COB[charge_0s,0] = 1
                    for j in charge_0s[1:]:
                        COB[0,j] = 1
                        COB[j,j] = -1
                    return COB

                OP_flat = build_COB(d)
                self.OP_ops = OP_ops = []
                self.charges = charges = []

                for i in range(d**2):
                    OP_ops.append(npc.Array.from_ndarray(OP_flat[:,i].reshape(d,d), [ss_op.leg, ss_op.leg.conj()], dtype=np.complex128, labels=['p', 'p*']))
                    if conserve != 'None':
                        charges.append(OP_ops[-1].qtotal.item())
                if conserve != 'None':
                    inverted_perm = np.argsort(leg1._perm)
                    # This will cause an error since there is ambiguity in ordering indices correspondong to the same charge.
                    assert np.all(inverted_perm == np.argsort(charges))
                else:
                    inverted_perm = np.arange(d**2)
                self.identity_ind = 0   # this might be moved by the permutation
                # We must reorder the OP array so that the charges work out; otherwise the charge structure is incompatible.
                self.OP = npc.Array.from_ndarray(OP_flat[inverted_perm[:,None],inverted_perm[None,:]], [leg1, leg1.conj()], dtype=np.complex128, qtotal=None, labels=['p', 'p*'])

            else:
                # Choose a basis of I, |0><0| - |i><i|, |i><j!=i| + |j!=i><i|, and |i><j!=i| - 1.j |j!=i><i|.
                # All of the operators are hermitian.

                # We choose a convention that generates the (normalized) TENPY spin-1/2 matrices
                # Sz = [[-1,0],[0,1]]/2, Sx = [[0,1],[1,0]]/2, Sy = [[0,1.j],[-1.j,0]]/2

                assert conserve == 'None'

                self.OP_ops = OP_ops = []
                # Want legPipes, so let's do this with NPC.
                OP_ops.append(npc.Array.from_ndarray(np.eye(d, dtype=np.complex128), [ss_op.leg, ss_op.leg.conj()], dtype=np.complex128, labels=['p', 'p*']))
                self.identity_ind = 0


                for j in range(0, d-1):
                    for i in range(j+1, d):
                        op = np.zeros((d,d), dtype=np.complex128)
                        op[j,i] = 1
                        OP_ops.append(npc.Array.from_ndarray(op + op.conj().T, [ss_op.leg, ss_op.leg.conj()], dtype=np.complex128, labels=['p', 'p*']))

                        op *= 1.j
                        OP_ops.append(npc.Array.from_ndarray(op + op.conj().T, [ss_op.leg, ss_op.leg.conj()], dtype=np.complex128, labels=['p', 'p*']))

                for i in range(1, d):
                    OP_ops.append(npc.Array.from_ndarray(np.diag([-1+0.j] + (i-1)*[0] + [1+0.j] + [0] * (d-1-i)), [ss_op.leg, ss_op.leg.conj()], dtype=np.complex128, labels=['p', 'p*']))


                OP_ops_augmented = np.column_stack([op.combine_legs(['p', 'p*']).to_ndarray() for op in OP_ops])
                self.OP = npc.Array.from_ndarray(OP_ops_augmented, [leg1, leg1.conj()], dtype=np.complex128, qtotal=None, labels=['p', 'p*'])

        # Turn original OP basis into orthogonal Q basis
        self.Q, self.R = npc.qr(self.OP, inner_labels=['p*', 'p'], mode='complete', pos_diag_R=True)
        self.Q.legs[1] = leg1.conj() # Need second leg of Q to be a LegPipe
        self.R.legs[0] = leg1

        self.new_ops = self.Q.replace_labels(['p', 'p*'], ['(p.p*)', '(q.q*)']).split_legs()
        self.new_ops = [self.new_ops.take_slice([i, j], ['q', 'q*']) for i in range(d) for j in range(d)]
        if hermitian:
            # Check that the Q basis is Hermitian
            hermitian=False
            while not hermitian:
                self.new_ops = self.Q.replace_labels(['p', 'p*'], ['(p.p*)', '(q.q*)']).split_legs()
                self.new_ops = [self.new_ops.take_slice([i, j], ['q', 'q*']) for i in range(d) for j in range(d)]

                traces = []
                failed = False
                for i, Q in enumerate(self.new_ops):
                    try:
                        assert np.isclose(npc.norm(Q - Q.conj().transpose()), 0.0), f"{Q.to_ndarray()} is not Hermitian."
                    except AssertionError as e:
                        # For d > 9, for some reason the final operator in Q is not Hermitian. I cannot figure out why.
                        # So we explicitly make it Hermitian where needed.
                        print(f'Operator {i}')
                        print(e)
                        self.new_ops[i] = Q = (Q + Q.conj().transpose())
                        self.new_ops[i] = Q = 1 / np.sqrt(npc.trace(npc.tensordot(Q, Q, axes=(['p*'], ['p'])))) * Q
                        failed=True
                    traces.append(npc.trace(Q, leg1=0, leg2=1))
                hermitian = not failed
                if not hermitian: # Found an operator which is not Hermitian; need to reconstruct Q using the updated operator
                    new_ops_augmented = np.column_stack([op.combine_legs(['p', 'p*']).to_ndarray() for op in self.new_ops])
                    self.Q = npc.Array.from_ndarray(new_ops_augmented, [leg1, leg1.conj()], dtype=np.complex128, qtotal=None, labels=['p', 'p*'])
                    # Find new R by Q^\dagger OP; I suppose we aren't guaranteed it's upper triangular.
                    self.R = npc.tensordot(Q.conj(), self.OP, axes=(['p*', 'p']))
                    R_diag = np.sign(np.diag(self._R.to_ndarray()))
                    self.Q.iscale_axis(R_diag, axis='p*')
                    self.R.iscale_axis(R_diag, axis='p')
                    assert np.isclose(npc.norm(self.OP - npc.tensordot(self.Q, self.R, axes=(['p*', 'p']))), 0.0)
        else:
            traces = []
            for i, Q in enumerate(self.new_ops):
                traces.append(npc.trace(Q, leg1=0, leg2=1))

        # s2d maps from standard -> desired basis by left multiplication; s2d @ standard = desired
        self.s2d = self.Q.conj().transpose() # The transpose just changes how the data is stored, but as long as we
        # use labels to reference the legs, it is not strictly needed.
        self.d2s = self.Q

        # Check that the matrices are inverses as desired
        assert np.isclose(npc.norm(npc.tensordot(self.d2s, self.s2d, axes=(['p*'], ['p'])) - npc.eye_like(self.d2s)), 0.0)
        assert np.isclose(npc.norm(npc.tensordot(self.s2d, self.d2s, axes=(['p*'], ['p'])) - npc.eye_like(self.d2s)), 0.0)

        # Check orthogonality - $trace_mat[i,j] = Tr (op_i^\dagger op_j) ?= delta_{i,j}$
        self.trace_mat = trace_mat = np.zeros((d**2,d**2), dtype=np.complex128)
        for i in range(d**2):
            for j in range(d**2):
                # NOTE: We conjugate one operator! Then, for the non-Hermitian off-diagonal operators, $Tr(|i><j| |j><i|) = 1$.
                trace_mat[i,j] = npc.trace(npc.tensordot(self.new_ops[i].conj(), self.new_ops[j], axes=(['p*'], ['p'])))
        # Always should be orthogonal since we've done QR
        assert np.isclose(np.linalg.norm(trace_mat - np.eye(d**2)), 0.0)

        # Check mostly traceless - only identity should have trace
        self.traces = traces = np.array(traces)
        self.traceful_ind = np.where(traces > 1.e-13)[0]
        if not trivial:
            assert len(self.traceful_ind) == 1
            self.traceful_ind = self.traceful_ind.item()
            assert self.traceful_ind == self.identity_ind == 0, f"traceful_ind={self.traceful_ind}, identity_ind={self.identity_ind}."
        else:
            assert len(self.traceful_ind) == len(self.identity_ind)
            #self.traceful_ind = self.identity_ind = -1

        # We already checked that the basis is hermitian if `hermitian==True`.

        ops = dict()
        self.conserve = conserve
        # Specify Hermitian conjugates
        Site.__init__(self, leg1, [str(i) for i in range(self.d**2)], sort_charge=sort_charge, **ops)

    def __repr__(self):
        """Debug representation of self."""
        return f'DoubledSite(q={self.d}, conserve={self.conserve}, sort_charge={self.leg.sorted}, hermitian={self.hermitian}, trivial={self.trivial})'
