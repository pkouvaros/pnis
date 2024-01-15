from resources.guarding.guardagent import GuardingConstants
from src.actors.agents.agent import MultiAgent
from src.utils.utils import get_widest_bounds, increment_indices
from src.verification.bounds.bounds import HyperRectangleBounds

from operator import __le__, __ge__, __eq__


class GuardingZeroOneAgent(MultiAgent):

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

        # intuitively, the private state of this zero-one agent will consist of
        # a bunch of binary variables,
        # one for every possible value of health where 0 <= h <= MAX_HEALTH
        # (0), (1),...,(MAX)
        self.private_dimensions = GuardingConstants.MAX_HEALTH_POINTS + 1
        # For the local dimensions there are also three components for the percepts,
        # encoded by 3 binary variables as well
        self.dimensions = self.private_dimensions + 3

        super(GuardingZeroOneAgent, self).__init__()

        self._precompute_observations()

    def _precompute_observations(self):
        self.observations = {}
        self.observations[0] = 0

        import numpy as np
        for health_points in range(1, self.private_dimensions):
            out = self.observation.forward(np.array([health_points]))
            self.observations[health_points] = np.argmax(out) + 1

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

        # 0 - expired, 1 - need rest, 2 - volunteer
        obs_vars = constrs_manager.create_binary_variables(3)

        # only the components corresponding to the private part (health) will be checked,
        # not the final three components for the percepts
        for health_points in range(self.private_dimensions):
            constrs_to_add.append(
                # if (h,p) is true, then the observation for h is true
                # (as precomputed NN output)
                constrs_manager.create_indicator_constraint(
                    input_state_vars[health_points], 1,
                    constrs_manager.get_assignment_constraint(obs_vars[self.observations[health_points]], 1)
                )
            )

        return obs_vars, constrs_to_add

    def get_constraints_for_protocol(self, constrs_manager, input_state_vars):
        """
        Create Gurobi constraints for performing an action. Constraints are only added to the model by
        the caller, to reduce side-effects in this function.
        :param constrs_manager: Manager of Gurobi constraints.
        :param input_state_vars: Gurobi variables representing the input state passed to the agent.
        :return:
            a list of action vars and a list of action constraints,
            where i-th element corresponds to i-th possible action.
        :side-effects: Modifies constraints manager when adding variables.
        """

        # 0 - expired, 1 - need rest, 2 - volunteered
        [expired, need_rest, volunteered] = input_state_vars[-3:]

        # one action is a set of (template state, template action) pairs
        # so we have an integer component for each possible template state, 0 - not present, 1 or 2 - present
        # for the expired state (0,0), the only available action is expired, so the first component is either 0 or 1
        # for a state (h,p), 0 is not present, 1 - rest action, 2 - guard action

        [no_guard_no_rest, only_rest] = constrs_manager.create_binary_variables(2)
        no_guard_no_rest_constraints = [
            constrs_manager.create_indicator_constraint(no_guard_no_rest, 1,
                                                        constrs_manager.get_assignment_constraint(need_rest, 0)),
            constrs_manager.create_indicator_constraint(no_guard_no_rest, 1,
                                                        constrs_manager.get_assignment_constraint(volunteered, 0)),
            constrs_manager.create_indicator_constraint(no_guard_no_rest, 0,
                                                        constrs_manager.get_linear_constraint(
                                                            [need_rest, volunteered], [1,1], 1, sense=__ge__))
        ]
        only_rest_constraints = [
            constrs_manager.create_indicator_constraint(only_rest, 1,
                                                        constrs_manager.get_assignment_constraint(need_rest, 1)),
            constrs_manager.create_indicator_constraint(only_rest, 1,
                                                        constrs_manager.get_assignment_constraint(volunteered, 0)),
            constrs_manager.create_indicator_constraint(only_rest, 0,
                                                        constrs_manager.get_assignment_constraint(volunteered, 1))
        ]

        set_action_constrs = []
        set_action_vars = []

        action_ranges = [1] + [2 for _ in range(GuardingConstants.MAX_HEALTH_POINTS)]
        action_indices = [0 for _ in range(len(action_ranges))]

        # Iterating through all possible states-action sets,
        # at maximum there are MAX_HEALTH^2, as each non-expired state can have two possible actions, rest or guard
        while(True):

            action_vars = [constrs_manager.create_integer_variable(lb=0, ub=2) for _ in range(self.private_dimensions)]
            constrs_manager.add_variable_bounds(action_vars,
                                                HyperRectangleBounds([0]*self.private_dimensions, [2]*self.private_dimensions))
            action_constrs = []

            action_constrs.extend(no_guard_no_rest_constraints)
            action_constrs.extend(only_rest_constraints)

            # if there is expired action, copy over the value of (0,0) state
            # otherwise there is no (0,0,expired) state-action-pair
            action_constrs.extend([
                constrs_manager.create_indicator_constraint(
                    expired, 1,
                    constrs_manager.get_equality_constraint(action_vars[0], input_state_vars[0])
                ),
                constrs_manager.create_indicator_constraint(
                    expired, 0,
                    constrs_manager.get_equality_constraint(action_vars[0], 0)
                )
            ])

            for hper in range(1, self.private_dimensions):
                # REST_ACTION
                if action_indices[hper] == 0:
                    action_constrs.extend([
                        constrs_manager.create_indicator_constraint(
                            volunteered, 1,
                            constrs_manager.get_equality_constraint(action_vars[hper], input_state_vars[hper] * GuardingConstants.REST_ACTION)
                        ),
                        constrs_manager.create_indicator_constraint(
                            need_rest, 1,
                            constrs_manager.get_equality_constraint(action_vars[hper], input_state_vars[hper] * GuardingConstants.REST_ACTION)
                        ),
                        constrs_manager.create_indicator_constraint(
                            no_guard_no_rest, 1,
                            constrs_manager.get_equality_constraint(action_vars[hper], 0)
                        )
                    ])

                # GUARD_ACTION
                if action_indices[hper] == 1:
                    action_constrs.extend([
                        constrs_manager.create_indicator_constraint(
                            volunteered, 1,
                            constrs_manager.get_equality_constraint(action_vars[hper], input_state_vars[hper] * GuardingConstants.GUARD_ACTION)
                        ),
                        # if there is no volunteered percept, then the second branch is rest as well
                        constrs_manager.create_indicator_constraint(
                            only_rest, 1,
                            constrs_manager.get_equality_constraint(action_vars[hper], input_state_vars[hper] * GuardingConstants.REST_ACTION)
                        ),
                        constrs_manager.create_indicator_constraint(
                            no_guard_no_rest, 1,
                            constrs_manager.get_equality_constraint(action_vars[hper], 0)
                        )
                    ])

            ###
            set_action_constrs.append(action_constrs)
            set_action_vars.append(action_vars)
            stop = increment_indices(action_indices, action_ranges)
            if stop:
                break

        return set_action_vars, set_action_constrs

    def get_constraints_for_transition(self, constrs_manager, state_vars,
                                       own_action_vars, joint_action_vars, env_action_vars):

        # state_vars is a zero-one state
        # own_action_vars is a zero-one action
        # joint_action_vars are a set of template actions

        constrs_to_add = []

        ####################################################################
        ##### Compute the complete joint actions as 3 binary variables #####
        ####################################################################
        [joint_expired, joint_rest, joint_guard] = constrs_manager.create_binary_variables(3)
        flat_joint_actions = [item for row in joint_action_vars for item in row]
        for action in flat_joint_actions:
            [e, r, g] = constrs_manager.create_binary_variables(3)

            constrs_to_add.extend([
                constrs_manager.create_indicator_constraint(
                    e, 1, constrs_manager.get_equality_constraint(action, GuardingConstants.EXPIRED_ACTION)),
                constrs_manager.create_indicator_constraint(
                    r, 1, constrs_manager.get_equality_constraint(action, GuardingConstants.REST_ACTION)),
                constrs_manager.create_indicator_constraint(
                    g, 1, constrs_manager.get_equality_constraint(action, GuardingConstants.GUARD_ACTION)),
                constrs_manager.get_sum_constraint([e,r,g], 1),
                constrs_manager.create_indicator_constraint(
                    e, 1, constrs_manager.get_equality_constraint(joint_expired, 1)),
                constrs_manager.create_indicator_constraint(
                    r, 1, constrs_manager.get_equality_constraint(joint_rest, 1)),
                constrs_manager.create_indicator_constraint(
                    g, 1, constrs_manager.get_equality_constraint(joint_guard, 1)),
            ])

        next_state_vars = constrs_manager.create_binary_variables(self.private_dimensions)

        # Add from own actions
        constrs_to_add.extend([
            constrs_manager.create_indicator_constraint(
                own_action_vars[0], 1, constrs_manager.get_equality_constraint(joint_expired, 1)),
            constrs_manager.create_indicator_constraint(
                own_action_vars[0], 1, constrs_manager.get_equality_constraint(next_state_vars[0], 1))
        ])

        for health_points in range(1, GuardingConstants.MAX_HEALTH_POINTS):

            [absent, r, g, r_guarded, r_unguarded] = constrs_manager.create_binary_variables(5)

            next_health_guarding = max(health_points + GuardingConstants.GUARDING_REWARD, GuardingConstants.EXPIRED_HEALTH_POINTS)
            next_health_regenerating = min(health_points + GuardingConstants.RESTING_REWARD, GuardingConstants.MAX_HEALTH_POINTS)
            next_health_unguarded = max(health_points + GuardingConstants.UNGUARDED_REWARD, GuardingConstants.EXPIRED_HEALTH_POINTS)

            action = own_action_vars[health_points]

            constrs_to_add.extend([
                constrs_manager.create_indicator_constraint(
                    absent, 1, constrs_manager.get_equality_constraint(action, 0)),
                constrs_manager.create_indicator_constraint(
                    r, 1, constrs_manager.get_equality_constraint(action, GuardingConstants.REST_ACTION)),
                constrs_manager.create_indicator_constraint(
                    g, 1, constrs_manager.get_equality_constraint(action, GuardingConstants.GUARD_ACTION)),

                constrs_manager.get_sum_constraint([absent,r,g], 1),

                # For computing joint action
                constrs_manager.create_indicator_constraint(
                    r, 1, constrs_manager.get_equality_constraint(joint_rest, 1)),
                constrs_manager.create_indicator_constraint(
                    g, 1, constrs_manager.get_equality_constraint(joint_guard, 1)),

                # Constraints for computing next health
                constrs_manager.create_indicator_constraint(
                    r_guarded, 1, constrs_manager.get_linear_constraint([r, joint_guard], [1,1], 2, sense=__ge__)),
                constrs_manager.create_indicator_constraint(
                    r_guarded, 0, constrs_manager.get_linear_constraint([r, joint_guard], [1,1], 1, sense=__le__)),

                constrs_manager.create_indicator_constraint(
                    r_unguarded, 1, constrs_manager.get_linear_constraint([r, joint_guard], [1, -1], 1, sense=__ge__)),
                constrs_manager.create_indicator_constraint(
                    r_unguarded, 0, constrs_manager.get_linear_constraint([r, joint_guard], [1, -1], 0, sense=__le__)),

                constrs_manager.create_indicator_constraint(
                    r_guarded, 1, constrs_manager.get_equality_constraint(next_state_vars[next_health_regenerating], 1)),
                constrs_manager.create_indicator_constraint(
                    r_unguarded, 1, constrs_manager.get_equality_constraint(next_state_vars[next_health_unguarded], 1)),

                constrs_manager.create_indicator_constraint(
                    g, 1, constrs_manager.get_equality_constraint(next_state_vars[next_health_guarding], 1)),
            ])

        return next_state_vars, constrs_to_add

    def get_branching_factor(self):
        return 2**GuardingConstants.MAX_HEALTH_POINTS

    def get_state_dimensions(self):
        return self.dimensions

    def get_private_state_dimensions(self):
        return self.private_dimensions
