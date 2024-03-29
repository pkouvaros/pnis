from abc import ABCMeta, abstractmethod

from src.utils.constants import NOT_IMPLEMENTED


class Constraint:
    __metaclass__ = ABCMeta

    @abstractmethod
    def accept(self, visitor):
        """
        Mutable accept. See Docstring of ConstraintVisitor (mutable constraint visitor).
        :param visitor: Visitor object
        :return: None
        """
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def acceptI(self, visitor):
        """
        Immutable accept. See Docstring of ConstraintVisitorI (immutable constraint visitor).
        :param visitor: Visitor object
        :return: Any concrete return type
        """
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def to_gurobi_constr(self, all_grb_vars, gmodel):
        """
        Convert a Constraint to a Gurobi constraint.
        :param all_grb_vars: (Initialised) Gurobi variables of the gmodel also passed to function.
        :param gmodel: (Initialised) Gurobi model.
        :return: Gurobi constraint.
        """
        raise NotImplementedError(NOT_IMPLEMENTED)

    @abstractmethod
    def get_vars(self):
        """
        Return the variables occuring in the constraint.
        :return: a list of custom variables.
        """
        raise NotImplementedError(NOT_IMPLEMENTED)
