from typing import List
from .Template import TemplateAgent
from .Strategy import AbstractStrategy, DQNStrategy
from .Agent import Agent
from .State import State, Transition
import pandas as pd
import random
from tqdm import tqdm


class Simulation:
    DEFAULT_NUMBER_OF_AGENTS: int = 3
    DEFAULT_MAX_EPISODE_LENGTH: int = 32
    DEFAULT_GUARD_VALUE: int = -8
    DEFAULT_REGEN_VALUE: int = 4
    DEFAULT_UNGUARDED_VALUE: int = -32

    def __init__(self,
                 number_of_agents: int = DEFAULT_NUMBER_OF_AGENTS,
                 max_episode_length: int = DEFAULT_MAX_EPISODE_LENGTH,
                 guard_value: int = DEFAULT_GUARD_VALUE,
                 regen_value: int = DEFAULT_REGEN_VALUE,
                 unguarded_value: int = DEFAULT_UNGUARDED_VALUE,
                 template_agent: TemplateAgent = TemplateAgent()
                 ) -> None:
        self.number_of_agents = number_of_agents
        self.max_episode_length = max_episode_length
        self.guard_value = guard_value
        self.regen_value = regen_value
        self.unguarded_value = unguarded_value
        self.template_agent = template_agent

        # initialise scores dataframe for easy access
        self.scores: pd.DataFrame = pd.DataFrame()

        # initialise agents
        self.agents: list[Agent] = [
            Agent(
                self.template_agent.max_hp,
                self.template_agent.initial_hp)
            for _ in range(self.number_of_agents)]

    def step_agent(self, agent_idx: int, action: int, unguarded: bool) -> Transition:
        def valid_hp(hp) -> int:
            if hp < 0:
                return 0
            elif hp > self.template_agent.max_hp:
                return self.template_agent.max_hp
            else:
                return hp

        agent_hp = self.agents[agent_idx].hp
        agent_state_label = self.agents[agent_idx].state_label

        if unguarded:
            reward = self.unguarded_value
            new_state_label = agent_state_label
        elif action == AbstractStrategy.GUARD_ACTION:
            reward = self.guard_value
            new_state_label = Agent.GUARD_STATE
        elif action == AbstractStrategy.REGEN_ACTION:
            reward = self.regen_value
            new_state_label = Agent.REGEN_STATE
        elif action == AbstractStrategy.EXPIRE_ACTION:
            reward = 0
            new_state_label = Agent.EXPIRE_STATE
        else:
            reward = 0
            new_state_label = agent_state_label

        new_agent_hp = valid_hp(agent_hp + reward)

        return Transition(
            State(agent_hp, agent_hp/self.template_agent.max_hp, agent_state_label),
            action,
            reward / self.template_agent.max_hp,
            State(new_agent_hp, new_agent_hp/self.template_agent.max_hp, new_state_label),
            new_agent_hp <= 0)

    def step(self, actions: List[int]) -> List[Transition]:
        # arbitrarily choose a random subset of guard actions and revert other guard actions to regen actions
        guard_actions = [idx for idx, action in enumerate(
            actions) if action == AbstractStrategy.GUARD_ACTION]
        chosen_guard = random.sample(
            guard_actions, k=random.choice(range(len(guard_actions))) + 1) if guard_actions else None
        unguarded = chosen_guard is None
        if not unguarded:
            for idx in range(len(actions)):
                if actions[idx] == AbstractStrategy.GUARD_ACTION and idx not in chosen_guard:
                    actions[idx] = AbstractStrategy.REGEN_ACTION
        # feedback new state to agents
        transitions = []
        for agent_idx in range(len(self.agents)):
            transition = self.step_agent(
                agent_idx, actions[agent_idx], unguarded)
            transitions.append(transition)

        return transitions
    
    def run(self, train: bool = False) -> None:
        for _ in range(self.max_episode_length):
            round_hps: dict[int, list[int]] = {}

            # get actions
            actions = [self.template_agent.strategy.act(self.agents[idx].hp / self.template_agent.max_hp) # TODO: change to dictionary
                       for idx in range(len(self.agents))]
            game_over = all(
                action == AbstractStrategy.EXPIRE_ACTION for action in actions)

            transitions = self.step(actions)

            # update agent states
            for idx, transition in enumerate(transitions):
                self.agents[idx].update_state(transition)
                round_hps[idx] = [transition.state.hp]
            
            # the agent will only remember one final transition, not looping in the final state
            if train:
                train_transitions = [t for t in transitions if t.state.hp > 0]
                self.template_agent.strategy.remember(train_transitions)

            # update scores dataframe
            self.scores = pd.concat(
                [self.scores, pd.DataFrame(round_hps)], ignore_index=True)
            
            if game_over:
                return

    def train(self, n_episodes: int = DQNStrategy.DEFAULT_TRAINING_EPISODES):
        def reset_agents():
            for agent in self.agents:
                agent.hp = self.template_agent.initial_hp
                agent.state_label = Agent.REGEN_STATE

        for _ in tqdm(range(n_episodes), desc='Running Episodes'):
            self.run(train=True)
            reset_agents()
