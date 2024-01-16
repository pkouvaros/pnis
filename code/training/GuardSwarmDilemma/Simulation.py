import glob
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
    DEFAULT_NUMBER_OF_AGENTS: int = 4
    DEFAULT_MAX_EPISODE_LENGTH: int = 16
    DEFAULT_GUARD_VALUE: int = -2
    DEFAULT_REGEN_VALUE: int = 1
    DEFAULT_UNGUARDED_VALUE: int = -3

    def __init__(self,
                 number_of_agents: int = DEFAULT_NUMBER_OF_AGENTS,
                 max_episode_length: int = DEFAULT_MAX_EPISODE_LENGTH,
                 guard_value: int = DEFAULT_GUARD_VALUE,
                 regen_value: int = DEFAULT_REGEN_VALUE,
                 unguarded_value: int = DEFAULT_UNGUARDED_VALUE,
                 template_agent: TemplateAgent = TemplateAgent()
                 ) -> None:
        self.number_of_agents: int = number_of_agents
        self.max_episode_length: int = max_episode_length
        if guard_value > 0:
            raise ValueError("Guard value must be negative")
        self.guard_value: int = guard_value
        if regen_value < 0:
            raise ValueError("Regen value must be positive")
        self.regen_value: int = regen_value
        if unguarded_value > 0:
            raise ValueError("Unguarded value must be negative")
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


    def _get_reward(self, agent_idx: int, transitions: Dict[int, Transition], game_over: bool) -> float:
        def _get_social_reward(transitions: Dict[int, Transition], agent_idx: int) -> float:
            # average health delta
            ave_health_delta = sum([transition.next_state.normalised_hp - transition.state.normalised_hp for agent_idx_, transition in transitions.items() if agent_idx_ != agent_idx]) / self.number_of_agents - 1
            dead_agent_penalty = sum([-1.0 if transition.next_state.state_label == State.EXPIRE_STATE else 0.0 for agent_idx_, transition in transitions.items() if agent_idx_ != agent_idx]) / self.number_of_agents - 1
            return (ave_health_delta + dead_agent_penalty) / 2

        transition = transitions[agent_idx]
        individual_reward = transition.next_state.normalised_hp - transition.state.normalised_hp

        reward = (transition.next_state.normalised_hp * _get_social_reward(transitions, agent_idx)) + individual_reward + (-1.0 if transition.next_state.state_label == State.EXPIRE_STATE else 1.0)
        reward += -2.0 if game_over else 0

        return reward

    def _get_rewards(self, transitions: Dict[int, Transition], game_over: bool) -> Dict[int, float]:
        rewards = {}
        for agent_idx in transitions.keys():
            rewards[agent_idx] = self._get_reward(agent_idx, transitions, game_over)
        return rewards

    def _get_transitions(self, actions: Dict[int, int]) -> Dict[int, Transition]:
        unguarded = not any(action == AbstractStrategy.GUARD_ACTION for action in actions.values())
        game_over = all(action == AbstractStrategy.EXPIRE_ACTION for action in actions.values())

        transitions: Dict[int, Transition] = {}
        for agent_idx, action in actions.items():
            transition = self._get_transition(
                self.agents[agent_idx].state, action, unguarded)
            transitions[agent_idx] = transition
        
        return transitions

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
            new_state_label = State.REGEN_STATE
        elif action == AbstractStrategy.GUARD_ACTION:
            delta_hp = self.guard_value
            new_state_label = State.GUARD_STATE
        elif action == AbstractStrategy.EXPIRE_ACTION:
            delta_hp = 0
            new_state_label = State.EXPIRE_STATE
        else:
            delta_hp = 0
            new_state_label = state_label

        new_hp = valid_hp(hp + delta_hp)
        new_state_label = new_state_label if new_hp > 0 else State.EXPIRE_STATE

        return Transition(
            State(hp, hp/self.template_agent.max_hp, state_label),
            action,
            State(new_hp, new_hp /
                  self.template_agent.max_hp, new_state_label),
            new_state_label == State.EXPIRE_STATE)

    def _step(self, actions: Dict[int, int]) -> Tuple[Dict[int, Transition], Dict[int, float], bool]:
        transitions = self._get_transitions(actions)

        # random subset of transitions are changed to result in regenerating instead of guarding
        guard_agents = [agent_idx for agent_idx, transition in transitions.items() if transition.action == AbstractStrategy.GUARD_ACTION]
        mask = [random.choice([0, 1]) for _ in range(len(guard_agents))]
        if len(mask) > 0 and all(flag == 1 for flag in mask):
            mask[random.choice(range(len(mask)))] = 0
        selected_to_regen = [agent_idx for agent_idx, flag in enumerate(mask) if flag == 1]
        for agent_idx in selected_to_regen:
            orig_state = transitions[agent_idx].state
            new_transition = self._get_transition(orig_state, AbstractStrategy.GUARD_ACTION, unguarded=False, arbitrated=True)
            transitions[agent_idx] = new_transition

        game_over = all(transition.final for transition in transitions.values())

        rewards = self._get_rewards(transitions, game_over)
        
        if len(transitions) != len(rewards):
            raise ValueError("Transition and reward vectors have different lengths")

        return transitions, rewards, game_over

    def run(self, train: bool = False, episode: int = 0) -> int:
        for i in range(self.max_episode_length):
            round_hps: dict[int, list[int]] = {}

            # get actions
            actions = {idx: self.template_agent.strategy.act(agent.state.hp / self.template_agent.max_hp)
                       for idx, agent in enumerate(self.agents)}
            # actions = {idx: AbstractStrategy.REGEN_ACTION for idx in range(self.number_of_agents)}

            # get transitions and construct experiences
            transitions, rewards, game_over = self._step(actions)
            experiences = {agent_idx: Experience(transition, reward) for 
                          (agent_idx, transition), reward in zip(transitions.items(), rewards.values())}
            
            # transition agents and record hp for visualisation
            for agent_idx, transition in transitions.items():
                self.agents[agent_idx].transition(transition)
                round_hps[agent_idx] = [transition.state.hp]
            
            # train the agents
            if train:
                self.template_agent.strategy.remember(experiences.values(), episode) # TODO: EXPIRATION?

            # update scores dataframe
            self.scores = pd.concat(
                [self.scores, pd.DataFrame(round_hps)], ignore_index=True)

            if game_over:
                return i
        return self.max_episode_length

    def train(self, n_episodes: int = DQNStrategy.DEFAULT_TRAINING_EPISODES, savefigs: bool = False) -> None:
        def reset_agents():
            for agent in self.agents:
                agent.state = State(agent.max_hp, 1.0, State.REGEN_STATE)

        if savefigs:
            if os.path.exists('traininghistory') == False:
                os.mkdir('traininghistory')

            qfunc = pd.DataFrame(self.template_agent.strategy.policy_net.predict(np.arange(0, 1, 0.01)), index=np.arange(0, 1, 0.01)) # type: ignore
            qfunc = qfunc.rename({0: 'Regen', 1: 'Guard'}, axis=1)
            qfunc.plot(title='Q-Function', xlabel='Normalised HP', ylabel='Q-Value')
            plt.savefig('traininghistory/episode0_0.png')
            plt.close()

        for i in tqdm(range(1, n_episodes + 1), desc='Running Episodes'):            
            length = self.run(train=True, episode=i)
            reset_agents()

            if savefigs:
                qfunc = pd.DataFrame(self.template_agent.strategy.policy_net.predict(np.arange(0, 1, 0.01)), index=np.arange(0, 1, 0.01)) # type: ignore
                qfunc = qfunc.rename({0: 'Regen', 1: 'Guard'}, axis=1)
                qfunc.plot(title='Q-Function Episode ' + str(i), xlabel='Normalised HP', ylabel='Q-Value')
                # save one frame for every turn of the episode
                # they are all the same, but this tells us how long the episode was in the final animation
                for j in range(length):
                    plt.savefig(f'traininghistory/episode{i}_{j}.png')
                plt.close()

        if savefigs:
            savedfigs = glob.glob('traininghistory/*.png')
            savedfigs.sort(key = os.path.getmtime)
            for i, fig_path in enumerate(savedfigs):
                new_path = f'traininghistory/episode{i:d}.png'
                os.rename(fig_path, new_path)

        print('Training complete')