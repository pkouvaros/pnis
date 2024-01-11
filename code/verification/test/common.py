#!/usr/bin/env python
import sys
sys.path.append('../../')

import datetime
from timeit import default_timer as timer
from gurobipy import Model

from src.verification.complete.constrmanager.gurobi_constraints_manager import GurobiConstraintsManager
from src.verification.complete.verifier.monolithic_ctl_milp_encoder import MonolithicCTLMILPEncoder
from src.verification.complete.verifier.breadth_first_compositional_ctl_milp_encoder import CompositionalCTLMILPEncoder
from src.utils.formula_visitors.immutable_nnf_visitor import FormulaVisitorNNF
from src.verification.complete.verifier.depth_first_compositional_ex_milp_encoder import DepthFirstCompositionalMILPEncoder

from src.verification.complete.constrmanager.custom_constraints_manager import CustomConstraintsManager
from src.verification.complete.verifier.aesverifier import AESVerifier
from src.utils.formula import *


TO_USER_RESULT = {"True": "False", "False": "True", "Timeout": "Timeout", "Interrupted": "Interrupted"}


def mono_verify(formula, input_hyper_rectangle, agent, env, timeout=3600):
    # print("Monolithic encoding")

    # struct to store solving stats
    log_info = []

    start = timer()
    print("Formula ", formula)

    print("Start: ", datetime.datetime.now())  # Do not delete

    # Create the Gurobi constraints manager to get a single program
    gmodel = Model("AES")
    gmodel.Params.LogToConsole = 0
    gmodel.Params.TimeLimit = timeout

    constraint_manager = GurobiConstraintsManager(gmodel)

    # Create a MILP builder visitor using the variables for the initial state
    initial_state_vars, _ = env.get_constraints_for_initial_state(constraint_manager, input_hyper_rectangle)
    mono_visitor = MonolithicCTLMILPEncoder(constraint_manager, initial_state_vars, agent, env)

    # Compute the set of MILP constraints for the negation of the formula in NNF
    negated_formula = NegationFormula(formula).acceptI(FormulaVisitorNNF())
    milp_constrs = negated_formula.acceptI(mono_visitor)

    # add the constraints and check feasibility of the resulting encoding
    constraint_manager.add_constrs(milp_constrs)
    result = constraint_manager.check_feasibility()

    print("End: ", datetime.datetime.now())  # Do not delete
    end = timer()
    runtime = end - start

    # Negate the result
    result = TO_USER_RESULT[result]

    # store results and total time
    log_info.append(runtime)
    log_info.append(result)

    print("Overall result and time:", result, runtime)
    stats = constraint_manager.stats
    print("Max number of variables  ", stats.max_var_number)
    print("Max number of constraints", stats.max_constr_number)

    if result == "False":
        print("Counter-example:")
        depth = len(stats.witness_states)
        for i in range(0, depth - 1):
            print("\t", "state", i, ":", stats.witness_states[i])
            print("\t", "action", i, ":", stats.witness_actions[i])
        print("\t", "state", depth - 1, ":", stats.witness_states[depth - 1])

    return log_info


def seq_verify(formula, input_hyper_rectangle, agent, env, timeout=3600, workers_n=1):
    """
    Verify specification using multi-milp encoding in parallel or sequential manner (depending on the workers number).
    :param env: A VcasEnv.
    :param agent: A VcasAgent.
    :param formula: Temporal specification to verify.
    :param input_hyper_rectangle: Input bounds.
    :return: None
    """

    print("Sequential Compositional encoding - {} workers".format(workers_n))
    log_info = []

    start = timer()
    print("Formula ", formula)
    print("Start: ", datetime.datetime.now())  # Do not delete

    # Create the custom constraints manager to get a number of (small) programs
    constraint_manager = CustomConstraintsManager()

    # Create a MILP builder visitor using the variables for the initial state
    initial_state_vars, _ = env.get_constraints_for_initial_state(constraint_manager, input_hyper_rectangle)
    # ctlverifier_visitor = FormulaMultiMILPBuilderVisitor(constraint_manager, initial_state_vars, agent, env)
    ## For the experiments, we use the more efficient encoder that only supports formulas
    ## with EX, disjunction and conjunction of atomic formulas
    ctlverifier_visitor = DepthFirstCompositionalMILPEncoder(constraint_manager, initial_state_vars, agent, env)

    # Create a pool verifier for the MILP builder visitor and for the negation of the formula in NNF
    negated_formula = NegationFormula(formula).acceptI(FormulaVisitorNNF())
    aesverifier = AESVerifier(ctlverifier_visitor, negated_formula, workers_n)

    AESVerifier.TIME_LIMIT = timeout
    AESVerifier.PARALLEL_PROCESSES_NUMBER = workers_n

    result, job_id, extra = aesverifier.verify()

    # Negate the result
    result = TO_USER_RESULT[result]

    print("End: ", datetime.datetime.now())  # Do not delete
    end = timer()
    runtime = end - start

    # Store results and total time
    log_info.append(runtime)
    log_info.append(result)

    print("Overall result and time:", result, runtime, "job n", job_id)
    if result == "False":
        print("\t\tCounter-example:")
        print("\n".join(["{}".format(item) for item in extra]))
    print("")

    return log_info


def pool_verify(formula, input_hyper_rectangle, agent, env, timeout=3600, workers_n=8):
    """
    Verify specification using multi-milp encoding in parallel or sequential manner (depending on the workers number).
    :param env: A VcasEnv.
    :param agent: A VcasAgent.
    :param formula: Temporal specification to verify.
    :param input_hyper_rectangle: Input bounds.
    :return: None
    """

    print("Parallel Compositional encoding - {} workers".format(workers_n))
    log_info = []

    start = timer()
    print("Formula ", formula)
    print("Start: ", datetime.datetime.now())  # Do not delete

    # Create the custom constraints manager to get a number of (small) programs
    constraint_manager = CustomConstraintsManager()

    # Create a MILP builder visitor using the variables for the initial state
    initial_state_vars, _ = env.get_constraints_for_initial_state(constraint_manager, input_hyper_rectangle)
    # ctlverifier_visitor = CompositionalCTLMILPEncoder(constraint_manager, initial_state_vars, agent, env)
    ## For the experiments, we use the more efficient encoder that only supports formulas
    ## with EX, disjunction and conjunction of atomic formulas
    ctlverifier_visitor = DepthFirstCompositionalMILPEncoder(constraint_manager, initial_state_vars, agent, env)

    # Create a pool verifier for the MILP builder visitor and for the negation of the formula in NNF
    negated_formula = NegationFormula(formula).acceptI(FormulaVisitorNNF())
    aesverifier = AESVerifier(ctlverifier_visitor, negated_formula, workers_n)

    AESVerifier.TIME_LIMIT = timeout
    AESVerifier.PARALLEL_PROCESSES_NUMBER = workers_n

    result, job_id, extra = aesverifier.verify()

    # Negate the result
    result = TO_USER_RESULT[result]

    print("End: ", datetime.datetime.now())  # Do not delete
    end = timer()
    runtime = end - start

    # Store results and total time
    log_info.append(runtime)
    log_info.append(result)

    print("Overall result and time:", result, runtime, "job n", job_id)
    if result == "False":
        print("\t\tCounter-example:")
        print("\n".join(["{}".format(item) for item in extra]))
    print("")

    return log_info
