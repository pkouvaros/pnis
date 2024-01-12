from GuardSwarmDilemma import Simulation
from GuardSwarmDilemma.Strategy import DQNStrategy

sim = Simulation()

sim.train(savefig=True)

dqn: DQNStrategy = sim.template_agent.strategy # type: ignore
dqn.save("qnets/policy_net.h5")