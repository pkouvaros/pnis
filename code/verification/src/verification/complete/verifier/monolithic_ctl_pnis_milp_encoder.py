import math

from src.utils.utils import get_widest_bounds
from src.verification.bounds.bounds import HyperRectangleBounds
from src.verification.complete.verifier.monolithic_boolean_milp_encoder import MonolithicBooleanMILPEncoder


class MonolithicCTLParametricNISMILPEncoder(MonolithicBooleanMILPEncoder):
    def __init__(self, constrs_manager, state_vars, template_agents, zero_one_agent, env):
        """
        An immutable visitor implementation for constructing a single MILP from a CTL formula for
        a parameteric NIS.
        :param constrs_manager: Constraints manager.
        :param state_vars: The current state of the env.
        :param agents: The set of agents.
        :param env: The  environment.
        :side-effects: Modifies constrs_manager and state_vars.
        """
        super(MonolithicCTLParametricNISMILPEncoder, self).__init__(constrs_manager, state_vars)
        self.env = env
        self.next_vars = self.state_vars

        self.template_agents = template_agents
        self.zero_one_agent = zero_one_agent
        self.num_joint_template_actions = math.prod([agent.get_branching_factor() for agent in self.template_agents])
        self.num_zero_one_actions = zero_one_agent.get_branching_factor()

    def visitENextFormula(self, element):
        """
        Existential formula visitor.
        :param element:
        :return:
        """

        k = element.k
        if k == 1:
            smaller_formula = element.left
        else:
            from src.utils.formula import ENextFormula
            smaller_formula = ENextFormula(k - 1, element.left)

        binvars = set()
        constrs_to_add = []

        # Allocate binary variables for each possible combination of actions.
        delta = self.constrs_manager.create_binary_variables(self.num_joint_template_actions * self.num_zero_one_actions)
        constrs_to_add.append(self.constrs_manager.get_sum_constraint(delta, 1))

        self.constrs_manager.update()

        init_vars = self.state_vars

        template_see_vars = []
        zero_one_see_vars = []
        for agent in self.template_agents:
            see_vars, see_constrs = agent.get_constraints_for_obs(self.constrs_manager, init_vars)
            constrs_to_add.extend(see_constrs)
            template_see_vars.append(see_vars)
            self.constrs_manager.get_variable_tracker().add_action_variables(see_vars)

        see_vars, see_constrs = self.zero_one_agent.get_constraints_for_obs(self.constrs_manager, init_vars)
        constrs_to_add.extend(see_constrs)
        zero_one_see_vars.append(see_vars)
        self.constrs_manager.get_variable_tracker().add_action_variables(see_vars)

        # next_state_vars are the variables for the next state.
        next_state_vars = self.constrs_manager.create_state_variables(len(self.state_vars))
        next_state_bounds = [(float("inf"), float("-inf")) for _ in range(len(next_state_vars))]  # Widest upper and lower bounds for output vars.
        self.constrs_manager.get_variable_tracker().add_state_variables(next_state_vars)

        for template_joint_action_idx in range(self.num_joint_template_actions):
            for zero_one_action_idx in range(self.num_zero_one_actions):
                # Add and get constraints for transition function.

                # y_ij = output_state_vars
                output_state_vars, output_state_constrs, _ \
                    = self.env.get_constraints_for_transition(
                    self.constrs_manager, template_see_vars, zero_one_see_vars, init_vars)

                # Compute max possible upper and min possible lower bounds for each output var.
                get_widest_bounds(next_state_bounds, output_state_vars)

                # Constraints for identifying the next_state_vars with the output_state_vars
                # for the current action combination
                next_var_constrs = [nsv == osv for nsv, osv in zip(next_state_vars, output_state_vars)]

                action_constrs, _ = \
                    self.env.get_constraints_for_joint_protocol(
                        self.constrs_manager, self.state_vars, template_joint_action_idx, zero_one_action_idx,
                        template_see_vars, zero_one_see_vars, output_state_vars, binvars)

                global_action_idx = template_joint_action_idx*self.num_joint_template_actions + zero_one_action_idx
                for constr in output_state_constrs + next_var_constrs + action_constrs:
                    if constr._sense != 'I':  # Hack to check if indicator constraint.
                        constrs_to_add.append(self.constrs_manager
                                              .create_indicator_constraint(delta[global_action_idx], 1, constr))
                    else:
                        constrs_to_add.append(constr)

        output_lower, output_upper = zip(*next_state_bounds)  # Unzip the bounds.
        self.constrs_manager.add_variable_bounds(next_state_vars, HyperRectangleBounds(output_lower, output_upper))

        self.state_vars = next_state_vars

        left_constraints = smaller_formula.acceptI(self)
        constrs_to_add.extend(left_constraints)

        self.state_vars = init_vars

        self.constrs_manager.binvars.update(binvars)

        return constrs_to_add
