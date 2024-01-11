#!/usr/bin/env python

import argparse
import sys

from common import TO_USER_RESULT
from resources.guarding.guardagent import GuardingAgent, GuardingConstants
from resources.guarding.guardzoagent import GuardingZeroOneAgent
from resources.guarding.guardenv import GuardingEnv
from src.verification.complete.verifier.monolithic_ctl_pnis_milp_encoder import MonolithicCTLParametricNISMILPEncoder

sys.path.append('../../')

from src.verification.complete.verifier.monolithic_atl_milp_encoder import MonolithicATLMILPEncoder

import datetime
from timeit import default_timer as timer
from gurobipy import Model

from src.verification.complete.constrmanager.gurobi_constraints_manager import GurobiConstraintsManager
from src.network_parser.network_model import NetworkModel
from src.utils.formula import *
from src.verification.bounds.bounds import HyperRectangleBounds


# Mono (--method 2)
def verify_single(formula, input_hyper_rectangle, agents, env, timeout):
    """
    Verify using the monolithic approach.
    :param formula: An ATL formula
    :param input_hyper_rectangle: Hyperrectangle representing initial state.
    :param gamma: Group of agents in coalition.
    :param not_gamma: Group of agents in complement of gamma.
    :param env: Multi-agent environment.
    :param timeout: Timeout in minutes.

    :return: Void.
    """
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
    mono_visitor = MonolithicCTLParametricNISMILPEncoder(constraint_manager, initial_state_vars, agents, env)

    # Compute the set of MILP constraints for the negation of the formula in NNF
    # negated_formula = NegationFormula(formula).acceptI(FormulaVisitorNNF())
    # milp_constrs = negated_formula.acceptI(mono_visitor)
    milp_constrs = formula.acceptI(mono_visitor)

    # add the constraints and check feasibility of the resulting encoding
    constraint_manager.add_constrs(milp_constrs)
    result = constraint_manager.check_feasibility()

    print("End: ", datetime.datetime.now())  # Do not delete
    end = timer()
    runtime = end - start

    # Negate the result
    result = TO_USER_RESULT[result]

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




def main():
    parser = argparse.ArgumentParser(description="Verify a MANS")
    parser.add_argument("-m", "--method", type=int, default=2, help="Method to use for verification: 0. Parallel-poly; 1. Sequential-poly; 2. Mono; 3. Mono-hybrid. ")
    parser.add_argument("-f", "--formula", type=int, default=1, help="Formula to verify: 0. [[o,i]] X^k safety; 1. <<o>> X^k safety; 2. [[i]] X^k safety; 3. <<o,i>> X^k unsafety")
    parser.add_argument("-n", "--noise", default=2.0, type=float, help="Noise to add to initial position of pilot.")
    parser.add_argument("-a", "--agents_number", default=2, type=int, help="Number of template agents.")
    parser.add_argument("-hp", "--initial_health", default=10, type=int, help="Initial health points of a template agent.")
    parser.add_argument("-per", "--initial_percept", default=2, type=int, help="Initial percept of a template agent (one of 0-expired, 1-rest, or 2-volunteer-to-guard).")
    parser.add_argument("-k", "--max_steps", default=4, type=int, help="Maximum number of time steps to verify for.")
    parser.add_argument("-w", "--workers", default=2, type=int, help="Number of workers.")
    parser.add_argument("-to", "--timeout", default=3600, type=int, help="Timeout in minutes.")

    ARGS = parser.parse_args()
    # ** Note ** , when adding a new formula, add appropriate agent config to list.
    agents, env = initialise_and_get_agent_and_env(ARGS.agents_number)

    # Constraint specific variables of the initial state to one value by setting the upper
    # bounds equal to the lower bounds.

    # This is len([health_agent1,
    #              percept_agent1,
    #              health_agent2,
    #              percept_agent2,
    #              healths_zo,
    #              percepts_zo,
    #              env_flag]
    #            )

    unzipped = zip(*[(ARGS.initial_health,            ARGS.initial_health),
                     (ARGS.initial_percept,           ARGS.initial_percept),
                     (ARGS.initial_health,            ARGS.initial_health),
                     (ARGS.initial_percept,           ARGS.initial_percept),
                     # (ARGS.initial_health,            ARGS.initial_health),
                     # (ARGS.initial_percept,           ARGS.initial_percept),
                     (0,                              0)])

    input_hyper_rectangle = HyperRectangleBounds(*unzipped)
    print(input_hyper_rectangle)

    steps = range(1, ARGS.max_steps + 1)

    for num_steps in steps:
        print(num_steps, "steps")

        formula = get_formula_and_gamma(ARGS, agents, num_steps)

        print("Formula to verify", formula)
        # Run a method.
        verify_single(formula, input_hyper_rectangle, agents, env, ARGS.timeout)
        print("\n")


def get_formula_and_gamma(ARGS, agents, num_steps):
    colony_alive = VarConstConstraint(
            StateCoordinate(GuardingConstants.HEALTH_IDX), GE, GuardingConstants.EXPIRED_HEALTH_POINTS + 1)

    # gamma_all = agents
    # not_gamma_empty = []
    #
    # colony_survives = GammaExistentialFormula(num_steps, gamma_all, not_gamma_empty, colony_alive) # [[o,i]] X^k safety
    # return gamma_all, not_gamma_empty, colony_survives

    colony_survives = ENextFormula(num_steps, colony_alive)
    return colony_survives


def initialise_and_get_agent_and_env(agents_number):
    """
    Initialise agent and environment.
    :return: List of initialised GuardingAgent objects and a GuardingEnv object.
    """

    # Where the agent networks are.
    REL_PATH = "../resources/guarding/agent.h5"

    # Dimension of the action space; there is only one action, the advisory, returned by the agent.
    ACTION_SPACE_DIM = 1

    # Dimension of the global state space;
    # This is len([health_agent1,
    #              percept_agent1,
    #              health_agent2,
    #              percept_agent2,
    #              healths_zo,
    #              percepts_zo,
    #              env_flag]
    #             ) = 7 for 2 agents.
    STATE_SPACE_DIM = 2 *(agents_number
                          # + 1
                          ) + \
                      1 # the environment state

    network_model = NetworkModel()
    network_model.parse(REL_PATH)

    agents = [GuardingAgent(network_model) for _ in range(agents_number)] #+ [GuardingZeroOneAgent(network_model)]

    env = GuardingEnv()

    return agents, env


if __name__ == "__main__":
    main()
