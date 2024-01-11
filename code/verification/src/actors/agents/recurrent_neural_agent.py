from src.actors.agents.agent import Agent
from src.actors.agents.neural_agent import NeuralAgent
from src.network_parser.recurrent_network_model import RecurrentNetworkModel
from src.utils.constants import NeuralOutputSettings


class RecurrentNeuralAgent(Agent):
    """
    Class representing an agent whose protocol function is defined exactly by a recurrent neural network.
    Gets as a parameter an instance of NetworkModel, our internal representation of a network
    """

    def __init__(self, network_model, output_setting=NeuralOutputSettings.INTEGER_ARGMAX):
        """
        :param network_model: an instance of RecurrentNetworkModel,
            an internal representation of a recurrent neural network
        :param output_setting: a value indicating whether the actual output
            of the agent's protocol function is simply the tuple of output q_values,
            a one-hot or an integer representation of the argmax of the output layer.
            Default value is integer representation.
        """
        assert isinstance(network_model, RecurrentNetworkModel)

        super(RecurrentNeuralAgent, self).__init__()

        self.network_model = network_model
        self.output_setting = output_setting

    def get_constraints_for_action(self, constrs_manager, input_state_vars, unrolling_depth=1):
        """
        Get the constraints for the network itself, set constraints to
        determine the action taken by the agent. Will need to compute the
        action of the agent based on the Q-value with highest value.
        :param constrs_manager: Constraint manager handling all constraints.
        :param input_state_vars: Set of variables representing the state that
        is input to the agent's protocol function.
        :side-effects: May modify constrs_manager.
        :param unrolling_depth: A natural number indicating how many times the
            internal loop of the RNN should be unrolled to obtain its abstraction, a FFNN
        :return:
            variables representing action taken,
            the constraints and
            the binary variables used in indicator constraints.
        """

        ff_network = self.network_model.unroll(unrolling_depth)

        return NeuralAgent.get_network_constraints(ff_network, self.output_setting, constrs_manager, input_state_vars)

