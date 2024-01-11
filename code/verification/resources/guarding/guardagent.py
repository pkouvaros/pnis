from src.actors.agents.agent import MultiAgent
from src.utils.utils import get_widest_bounds
from src.verification.bounds.bounds import HyperRectangleBounds


class GuardingConstants:
    EXPIRED_OBSERVATION = 0
    REST_OBSERVATION = 1
    VOLUNTEER_GUARD_OBSERVATION = 2

    EXPIRED_ACTION = 0
    REST_ACTION = 1
    GUARD_ACTION = 2

    EXPIRED_HEALTH_POINTS = 0

    AGENT_STATE_DIMENSIONS = 2
    # The index for the health value in a state
    HEALTH_IDX = 0
    PERCEPT_IDX = 1

    AGENT_MAX_ACTIONS = 2


class GuardingAgent(MultiAgent):

    def __init__(self, obs_network_model):
        """
        GuardingAgent represents an agent template in the guarding game.

        In the game, there is a colony of agents. The colony
        needs to be guarded by exactly one guard. Guarding duty
        costs the guard some health, Gp, while those who are resting
        improve their health by Rr, where the health is limited by a
        maximum value Mh. If no one is guarding the colony, then
        all the agents in the colony lose some health, Up. When an
        agent does not have any health left, it expires.
        :param network_models: The internal representation of the 9 models
        """
        # The dimensionality of the action space,
        # it is 1 as the action is the advisory.
        self.ACTION_SPACE_DIM = 1

        # the network representing the observation function
        self.observation = obs_network_model

        # Dimension of the output of the network.
        # 0 - rest only
        # 1 - volunteer to guard (two available actions, rest and guard)
        # There is also an expired action, but it is not part of the NN output
        # We will hardcode it for the inputs where the health points are <= 0
        self.NETWORK_OUTPUT_DIM = 2

        # Dimension of the input state of the network. Determines the number of variables from the
        # entire scenario state which make up the actual input state of the network.
        #
        # The input to the neural network is health_points
        self.NETWORK_INPUT_DIM = 1

        super(GuardingAgent, self).__init__()

    def get_constraints_for_obs(self, constrs_manager, input_state_vars):
        """
        Create Gurobi constraints for performing an observation on the env.
        Get the constraints for the network itself, set constraints to
        determine the percept output by the agent. Will need to compute the
        percept of the agent based on the Q-value with highest value.
        :param constrs_manager: Constraint manager handling all constraints.
        :param input_state_vars: Set of variables representing the state that
        is input to the agent's protocol function.
        :side-effects: May modify constrs_manager.
        :return:
            variables representing action taken,
            the constraints and
            the binary variables used in indicator constraints.
        """

        # Initialise a list of constraints to be added to allow only one advisory network to be used at a given time.
        constrs_to_add = []

        # Select the variables which correspond to the network input.
        # It is only the first component, encoding the health points
        network_input_vars = input_state_vars[:self.NETWORK_INPUT_DIM]

        # Binary variable for checking whether the agent is expired (health_points is non-positive)
        [expired] = constrs_manager.create_binary_variables(1)

        # Integer variable for the resulting observation
        # 0 - Expired
        # 1 - Rest
        # 2 - (volunteer to) Guard
        integer_obs_var = constrs_manager.create_integer_variable(lb=0, ub=2)

        # If expired, then health_points should be non-positive and the observation should be 0
        constrs_to_add.append(
            constrs_manager.create_indicator_constraint(expired, 1,
                                                        constrs_manager.get_le_constraint(
                                                            network_input_vars[0],
                                                            GuardingConstants.EXPIRED_HEALTH_POINTS))
        )
        constrs_to_add.append(
            constrs_manager.create_indicator_constraint(expired, 1,
                                                        constrs_manager.get_assignment_constraint(
                                                            integer_obs_var,
                                                            GuardingConstants.EXPIRED_OBSERVATION))
        )

        #########################################################################
        # Otherwise, health_points should be at least 1                         #
        # and we add the constraints for computing the observation using the NN #
        #########################################################################
        constrs_to_add.append(
            constrs_manager.create_indicator_constraint(expired, 0,
                                                        constrs_manager.get_ge_constraint(
                                                            network_input_vars[0],
                                                            GuardingConstants.EXPIRED_HEALTH_POINTS + 1))
        )

        # Add constraints for the observation neural network
        q_vars, network_constrs = constrs_manager.get_network_constraints(self.observation.layers, network_input_vars)
        constrs_to_add.extend([constrs_manager.create_indicator_constraint(expired, 0, constr)
                               for constr in network_constrs])

        # Add the encoding of argmax for the computed q-values.
        # argmax_constrs would typically have indicator constraints and one sum constraint.
        # We cannot nest indicator constraints, so we add them without a guard (expired=0).
        # If expired = 1, then they would be enforced, but not really used anywhere.
        # If expired = 0, then they would be enforced and used.
        one_hot_argmax_vars, argmax_constrs = constrs_manager.get_argmax_constraints(q_vars, use_q_bounds=True)
        constrs_to_add.extend(argmax_constrs)

        # Return a single integer variable as the output of argmax.
        # integer_argmax_constrs would typically have indicator constraints only
        # We cannot nest indicator constraints, so we add them without a guard (expired=0)
        integer_argmax_var, integer_argmax_constrs = constrs_manager.get_argmax_index_constraints(one_hot_argmax_vars)
        constrs_manager.add_variable_bounds([integer_argmax_var],
                                            HyperRectangleBounds([integer_argmax_var.lb], [integer_argmax_var.ub]))
        constrs_to_add.extend(integer_argmax_constrs)

        # The final observation should be integer_argmax_var + 1
        constrs_to_add.append(
            constrs_manager.create_indicator_constraint(expired, 0,
                                                        constrs_manager.get_linear_constraint(
                                                            [integer_obs_var, integer_argmax_var], [1, -1], 1))
        )

        return [integer_obs_var], constrs_to_add

    def get_constraints_for_protocol(self, constrs_manager, input_state_vars):
        """
        Create Gurobi constraints for performing an action. Constraints are only added to the model by
        the caller, to reduce side-effects in this function.
        :param constrs_manager: Manager of Gurobi constraints.
        :param input_state_vars: Gurobi variables representing the input state passed to the agent.
        :return: Singleton list of Gurobi variables.
        :side-effects: Modifies constraints manager when adding variables.
        """

        constrs_to_add = []

        percept_var = input_state_vars[GuardingConstants.PERCEPT_IDX]

        # Binary variable for checking whether the percept is expired, need rest or volunteered to guard
        [expired, need_rest, volunteered] = constrs_manager.create_binary_variables(3)

        constrs_to_add.extend([
            constrs_manager.create_indicator_constraint(
                expired, 1,
                constrs_manager.get_assignment_constraint(percept_var, GuardingConstants.EXPIRED_OBSERVATION)),
            constrs_manager.create_indicator_constraint(
                need_rest, 1,
                constrs_manager.get_assignment_constraint(percept_var, GuardingConstants.REST_OBSERVATION)),
            constrs_manager.create_indicator_constraint(
                volunteered, 1,
                constrs_manager.get_assignment_constraint(percept_var, GuardingConstants.VOLUNTEER_GUARD_OBSERVATION)),
            constrs_manager.get_sum_constraint([expired, need_rest, volunteered], 1)
        ])

        action1 = constrs_manager.create_integer_variable(lb=0, ub=2)
        action2 = constrs_manager.create_integer_variable(lb=0, ub=2)

        constrs_to_add.extend([
            constrs_manager.create_indicator_constraint(
                expired, 1,
                constrs_manager.get_assignment_constraint(action1, GuardingConstants.EXPIRED_ACTION)),
            constrs_manager.create_indicator_constraint(
                expired, 1,
                constrs_manager.get_assignment_constraint(action2, GuardingConstants.EXPIRED_ACTION)),
            constrs_manager.create_indicator_constraint(
                need_rest, 1,
                constrs_manager.get_assignment_constraint(action1, GuardingConstants.REST_ACTION)),
            constrs_manager.create_indicator_constraint(
                need_rest, 1,
                constrs_manager.get_assignment_constraint(action2, GuardingConstants.REST_ACTION)),
            constrs_manager.create_indicator_constraint(
                volunteered, 1,
                constrs_manager.get_assignment_constraint(action1, GuardingConstants.REST_ACTION)),
            constrs_manager.create_indicator_constraint(
                volunteered, 1,
                constrs_manager.get_assignment_constraint(action2, GuardingConstants.GUARD_ACTION))
        ])

        actions_vars = []

        integer_argmax_var = None

        return [integer_argmax_var], constrs_to_add

    def get_constraints_for_transition(self, constrs_manager, state_vars,
                                       own_action_vars, joint_action_vars, env_action_vars):

        constrs_to_add = []

        # Binary variable for checking whether the agent is expired, need rest or volunteered to guard
        [expired, need_rest, volunteered] = constrs_manager.create_binary_variables(3)

        constrs_to_add.append(
            constrs_manager.create_indicator_constraint(expired, 1,
                                                        constrs_manager.get_assignment_constraint(
                                                            state_vars[GuardingConstants.PERCEPT_IDX],
                                                            GuardingConstants.EXPIRED_OBSERVATION))
        )

        constrs_to_add.append(
            constrs_manager.create_indicator_constraint(need_rest, 1,
                                                        constrs_manager.get_assignment_constraint(
                                                            state_vars[GuardingConstants.PERCEPT_IDX],
                                                            GuardingConstants.REST_OBSERVATION))
        )

        constrs_to_add.append(
            constrs_manager.create_indicator_constraint(volunteered, 1,
                                                        constrs_manager.get_assignment_constraint(
                                                            state_vars[GuardingConstants.PERCEPT_IDX],
                                                            GuardingConstants.VOLUNTEER_GUARD_OBSERVATION))
        )

        constrs_to_add.append(
            constrs_manager.get_sum_constraint([expired, need_rest, volunteered], 1)
        )




    def get_branching_factor(self):
        return GuardingConstants.AGENT_MAX_ACTIONS

    def get_state_dimensions(self):
        return GuardingConstants.AGENT_STATE_DIMENSIONS
