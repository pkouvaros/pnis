from GuardSwarmDilemma import Simulation
from GuardSwarmDilemma.Strategy import DQNStrategy

sim = Simulation()

sim.train(512, savefigs=True)

dqn: DQNStrategy = sim.template_agent.strategy # type: ignore
dqn.save("qnets/policy_net.h5")

training_history = dqn.training_history
# Find average values for each episode
averages = training_history.groupby('Episode').mean()
averages = averages.rename(columns={'Loss': 'Average Loss', 'Epsilon': 'Average Epsilon', 'Reward': 'Average Reward'})
averages.to_csv('traininghistory/history.csv')