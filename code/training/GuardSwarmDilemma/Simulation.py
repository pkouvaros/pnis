import os
from typing import Dict, Tuple
from matplotlib import pyplot as plt
import numpy as np
from .Template import TemplateAgent
from .Strategy import AbstractStrategy, DQNStrategy
from .Agent import Agent
from .State import Experience, State, Transition
import pandas as pd
import random
from tqdm import tqdm


class Simulation:
    DEFAULT_NUMBER_OF_AGENTS: int = 10
    DEFAULT_MAX_EPISODE_LENGTH: int = 32
    DEFAULT_GUARD_VALUE: int = -4
    DEFAULT_REGEN_VALUE: int = 1
    DEFAULT_UNGUARDED_VALUE: int = -TemplateAgent.DEFAULT_MAX_HP

    def __init__(self,
                 number_of_agents: int = DEFAULT_NUMBER_OF_AGENTS,
                 max_episode_length: int = DEFAULT_MAX_EPISODE_LENGTH,
                 guard_value: int = DEFAULT_GUARD_VALUE,
                 regen_value: int = DEFAULT_REGEN_VALUE,
                 unguarded_value: int = DEFAULT_UNGUARDED_VALUE,
                 template_agent: TemplateAgent = TemplateAgent(),
                 agent_training_altruism: float = -1,
                 ) -> None:
        self.number_of_agents: int = number_of_agents
        self.max_episode_length: int = max_episode_length
        self.guard_value: int = guard_value
        self.regen_value: int = regen_value
        self.unguarded_value: int = unguarded_value
        self.template_agent: TemplateAgent = template_agent

        # initialise scores dataframe for easy access
        self.scores: pd.DataFrame = pd.DataFrame()

        # initialise agents
        self.agents: list[Agent] = [
            Agent(
                self.template_agent.max_hp,
                self.template_agent.initial_hp)
            for _ in range(self.number_of_agents)]
        
        self.agent_training_altruism = agent_training_altruism if agent_training_altruism != -1 else (1 - (1/number_of_agents))


    def _get_reward(self, transition: Transition, ave_health_delta: float, game_over: bool) -> float:
        individual_reward = transition.next_state.normalised_hp - transition.state.normalised_hp
        social_reward = ave_health_delta

        reward = (self.agent_training_altruism * social_reward) + ((1 - self.agent_training_altruism) * individual_reward)

        return reward if not game_over else -1

    def _get_rewards(self, transitions: Dict[int, Transition], game_over: bool) -> Dict[int, float]:
        ave_health_delta = sum([transition.next_state.normalised_hp - transition.state.normalised_hp for transition in transitions.values()]) / len(transitions)

        rewards = {}
        for agent_idx, transition in transitions.items():
            rewards[agent_idx] = self._get_reward(transition, ave_health_delta, game_over)
        return rewards

    def _get_transitions(self, actions: Dict[int, int]) -> Tuple[Dict[int, Transition], bool]:
        unguarded = not any(action == AbstractStrategy.GUARD_ACTION for action in actions.values())

        transitions: Dict[int, Transition] = {}
        for agent_idx, action in actions.items():
            transition = self._get_transition(
                self.agents[agent_idx].state, action, unguarded)
            transitions[agent_idx] = transition

        game_over = all(action == AbstractStrategy.EXPIRE_ACTION for action in actions.values())
        
        return transitions, game_over

    def _get_transition(self, state: State, action: int, unguarded: bool, arbitrated: bool = False) -> Transition:
        def valid_hp(hp) -> int:
            if hp < 0:
                return 0
            elif hp > self.template_agent.max_hp:
                return self.template_agent.max_hp
            else:
                return hp

        hp = state.hp
        state_label = state.state_label

        if unguarded:
            delta_hp = self.unguarded_value
            new_state_label = state_label
        elif arbitrated or action == AbstractStrategy.REGEN_ACTION:
            delta_hp = self.regen_value
            new_state_label = Agent.REGEN_STATE
        elif action == AbstractStrategy.GUARD_ACTION:
            delta_hp = self.guard_value
            new_state_label = Agent.GUARD_STATE
        elif action == AbstractStrategy.EXPIRE_ACTION:
            delta_hp = 0
            new_state_label = Agent.EXPIRE_STATE
        else:
            delta_hp = 0
            new_state_label = state_label

        new_hp = valid_hp(hp + delta_hp)

        return Transition(
            State(hp, hp/self.template_agent.max_hp, state_label),
            action,
            State(new_hp, new_hp /
                  self.template_agent.max_hp, new_state_label),
            new_hp <= 0)

    def _step(self, actions: Dict[int, int]) -> Tuple[Dict[int, Transition], Dict[int, float], bool]:
        transitions, game_over = self._get_transitions(actions)

        # random subset of transitions are changed to result in regenerating instead of guarding
        guard_agents = [agent_idx for agent_idx, transition in transitions.items() if transition.action == AbstractStrategy.GUARD_ACTION]
        selected_to_regen = random.sample(guard_agents, k=random.choice(range(len(guard_agents)))) if len(guard_agents) > 0 else []
        for agent_idx in selected_to_regen:
            orig_state = transitions[agent_idx].state
            new_transition = self._get_transition(orig_state, AbstractStrategy.GUARD_ACTION, unguarded=False, arbitrated=True)
            transitions[agent_idx] = new_transition

        rewards = self._get_rewards(transitions, game_over)
        
        if len(transitions) != len(rewards):
            raise ValueError("Transition and reward vectors have different lengths")

        return transitions, rewards, game_over

    def run(self, train: bool = False) -> None:
        for _ in range(self.max_episode_length):
            round_hps: dict[int, list[int]] = {}

            # get actions
            actions = {idx: self.template_agent.strategy.act(agent.state.hp / self.template_agent.max_hp)  # TODO: change to dictionary
                       for idx, agent in enumerate(self.agents)}

            # get transitions and construct experiences
            transitions, rewards, game_over = self._step(actions)
            experiences = {agent_idx: Experience(transition, reward) for 
                          (agent_idx, transition), reward in zip(transitions.items(), rewards.values())}
            
            # transition agents and record hp for visualisation
            for agent_idx, transition in transitions.items():
                self.agents[agent_idx].transition(transition)
                round_hps[agent_idx] = [transition.state.hp]
            
            # train the agents
            # the agent will only remember one final transition, not looping in the final state
            if train:
                train_experiences = [e for e in experiences.values() if e.transition.state.hp > 0]
                self.template_agent.strategy.remember(train_experiences)

            # update scores dataframe
            self.scores = pd.concat(
                [self.scores, pd.DataFrame(round_hps)], ignore_index=True)

            if game_over:
                return

    def train(self, n_episodes: int = DQNStrategy.DEFAULT_TRAINING_EPISODES, savefig: bool = False) -> None:
        def reset_agents():
            for agent in self.agents:
                agent.state = State(agent.max_hp, 1.0, Agent.REGEN_STATE)

        for i in tqdm(range(n_episodes), desc='Running Episodes'):
            if savefig:
                qfunc = pd.DataFrame(self.template_agent.strategy.policy_net.predict(np.arange(0, 1, 0.01)), index=np.arange(0, 1, 0.01)) # type: ignore
                qfunc.plot(title='Q-Function', xlabel='Normalised HP', ylabel='Q-Value')
                if os.path.exists('trainingfig') == False:
                    os.mkdir('trainingfig')
                plt.savefig(f'trainingfig/episode{i}.png')
                plt.close()
            
            self.run(train=True)
            reset_agents()

