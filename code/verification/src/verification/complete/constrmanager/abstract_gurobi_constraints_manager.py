from gurobipy import *
import numpy as np

from src.verification.complete.constrmanager.aes_variable_tracker import AESVariableTracker
from src.verification.complete.constrmanager.constraints_manager_interface import ConstraintsManagerI
from operator import __le__, __ge__, __eq__


class AbstractGurobiConstraintsManager(ConstraintsManagerI):
    """
    class AbstractGurobiConstraintsManager

    implements the basic functionality of a constraint manager for building a Gurobi model for verifying
    an agent-environment system

    includes a variable tracker
    """

    def __init__(self, gmodel):
        super(AbstractGurobiConstraintsManager, self).__init__()

        self.gmodel = gmodel
        self.binvars = set()
        self.variable_tracker = AESVariableTracker()

    def create_state_variables(self, var_number, lbs=None, ubs=None):
        # :side-effects: Modifies Gurobi model.
        if lbs is None or ubs is None:
            grb_vars = [self.gmodel.addVar(lb=-GRB.INFINITY) for _ in range(var_number)]
        else:
            grb_vars = [self.gmodel.addVar(lb=lbs[i], ub=ubs[i]) for i in range(var_number)]
        return grb_vars

    def create_binary_variables(self, var_number, lbs=None, ubs=None):
        # :side-effects: Modifies Gurobi model.
        if lbs is None or ubs is None:
            grb_vars = [self.gmodel.addVar(vtype=GRB.BINARY) for _ in range(var_number)]
        else:
            grb_vars = [self.gmodel.addVar(vtype=GRB.BINARY, lb=lbs[i], ub=ubs[i]) for i in range(var_number)]

        return grb_vars

    def create_integer_variable(self, lb, ub):
        grb_var = self.gmodel.addVar(vtype=GRB.INTEGER, lb=lb, ub=ub)
        return grb_var

    @staticmethod
    def get_sum_constraint(delta, rhs):
        return quicksum(delta) == rhs

    @staticmethod
    def get_assignment_constraint(var, value):
        return var == value

    @staticmethod
    def get_equality_constraint(var1, var2):
        return var1 == var2

    @staticmethod
    def create_indicator_constraint(binvar, binval, expr):
        return (binvar == binval) >> expr

    @staticmethod
    def get_le_constraint(var, value):
        return var <= value

    @staticmethod
    def get_ge_constraint(var, value):
        return var >= value

    @staticmethod
    def get_inequality_constraint(var, value, inequality):
        return inequality(var, value)

    @staticmethod
    def get_linear_constraint(vars, coeff, rhs, sense=__eq__):
        return sense(np.array(coeff).dot(np.array(vars)), rhs)

    @staticmethod
    def get_linear_constraint(vars, coeff, rhs, sense=__eq__):
        return sense(np.array(coeff).dot(np.array(vars)), rhs)

    @staticmethod
    def get_max_constraint(var, values):
        return (var == max_(values))

    @staticmethod
    def get_min_constraint(var, values):
        return (var == min_(values))

    def update(self):
        self.gmodel.update()

    def get_vars_by_name(self, grb_vars):
        # :side-effects: Updates and reads from Gurobi model.
        self.gmodel.update()
        return [self.gmodel.getVarByName(var.varName) for var in grb_vars]

    def add_constrs(self, new_constrs):
        # :side-effects: Modifies Gurobi model.
        for c in new_constrs:
            self.gmodel.addConstr(c)

    @staticmethod
    def get_atomic_constraint(constraint, state_vars):
        if hasattr(constraint, "get_atomic_constraint"):
            return constraint.get_atomic_constraint(state_vars)
        else:
            raise Exception("Unexpected type of atomic constraint", constraint)

    def get_variable_tracker(self):
        return self.variable_tracker
