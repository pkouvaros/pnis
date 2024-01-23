from abc import ABC, abstractmethod
from email.policy import Policy
import numpy as np
import random as rand
import pandas as pd
from .State import Experience, State
from collections import deque
from tensorflow.keras.layers import Dense, InputLayer
from typing import Any, List, Tuple, Union
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import History


class AbstractStrategy(ABC):
    GUARD_ACTION: int = 2
    REGEN_ACTION: int = 1
    EXPIRE_ACTION: int = 0

    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name

    @abstractmethod
    def act(self, normalised_hp: float, *args: Any, **kwargs: Any) -> int:
        pass

    @abstractmethod
    def remember(self, *args: Any, **kwargs: Any) -> None:
        pass


class RandomStrategy(AbstractStrategy):
    def __init__(self) -> None:
        super().__init__('Random')

    def act(self, normalised_hp: float) -> int:
        if normalised_hp <= 0:
            return AbstractStrategy.EXPIRE_ACTION
        else:
            return rand.choice([AbstractStrategy.GUARD_ACTION, AbstractStrategy.REGEN_ACTION])

    def remember(self, *args: Any, **kwargs: Any) -> None:
        pass


class CooperatorStrategy(AbstractStrategy):
    def __init__(self) -> None:
        super().__init__('Cooperator')

    def act(self, normalised_hp: float) -> int:
        if normalised_hp <= 0:
            return AbstractStrategy.EXPIRE_ACTION
        else:
            return AbstractStrategy.GUARD_ACTION

    def remember(self, *args: Any, **kwargs: Any) -> None:
        pass


class DefectorStrategy(AbstractStrategy):
    def __init__(self) -> None:
        super().__init__('Defector')

    def act(self, normalised_hp: int) -> int:
        if normalised_hp <= 0:
            return AbstractStrategy.EXPIRE_ACTION
        else:
            return AbstractStrategy.REGEN_ACTION

    def remember(self, *args: Any, **kwargs: Any) -> None:
        pass


class DQNStrategy(AbstractStrategy):
    DEFAULT_TRAINING_EPISODES: int = 1024

    DEFAULT_POLICY_TRAIN_FREQ: int = 1
    DEFAULT_TARGET_NET_UPDATE_FREQ: int = 32
    DEFAULT_REPLAY_BUFFER_SIZE: int = 4096
    DEFAULT_BATCH_SIZE: int = 16
    DEFAULT_LEARNING_RATE: float = 0.0025
    DEFAULT_DENSE_LAYERS_WIDTH: int = 4
    DEFAULT_DENSE_LAYERS_DEPTH: int = 2
    DEFAULT_DISCOUNT_FACTOR: float = 0.99
    DEFAULT_EPSILON: float = 1.0
    DEFAULT_EPSILON_MIN: float = 0.01
    DEFAULT_EPSILON_DECAY: float = 0.999

    def __init__(self,
                 policy_train_freq: int = DEFAULT_POLICY_TRAIN_FREQ,
                 target_net_update_freq: int = DEFAULT_TARGET_NET_UPDATE_FREQ,
                 replay_buffer_size: int = DEFAULT_REPLAY_BUFFER_SIZE,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 learning_rate: float = DEFAULT_LEARNING_RATE,
                 dense_layers_width: int = DEFAULT_DENSE_LAYERS_WIDTH,
                 dense_layers_depth: int = DEFAULT_DENSE_LAYERS_DEPTH,
                 discount_factor: float = DEFAULT_DISCOUNT_FACTOR,
                 epsilon: float = DEFAULT_EPSILON,
                 epsilon_min: float = DEFAULT_EPSILON_MIN,
                 epsilon_decay: float = DEFAULT_EPSILON_DECAY
                 ) -> None:
        super().__init__('DQN')
        self.state_size: int = 2
        self.policy_net_train_freq: int = policy_train_freq
        self.target_net_update_freq: int = target_net_update_freq
        self.memory: deque[Experience] = deque(maxlen=replay_buffer_size)
        self.batch_size: int = batch_size
        self.learning_rate: float = learning_rate
        self.dense_layer_width: int = dense_layers_width
        self.dense_layer_depth: int = dense_layers_depth
        self.discount_factor: float = discount_factor
        self.epsilon: float = epsilon
        self.epsilon_min: float = epsilon_min
        self.epsilon_decay: float = epsilon_decay

        self.policy_net: Sequential = self._build_model()
        self.target_net: Sequential = self._build_model()

        self.training_counter: int = 0

        self.training_history: pd.DataFrame = pd.DataFrame()

    def _build_model(self) -> Sequential:
        model = Sequential()
        model.add(InputLayer(input_shape=(1,)))  # type: ignore
        for _ in range(self.dense_layer_depth):
            model.add(Dense(self.dense_layer_width, activation='relu')) # type: ignore
        model.add(Dense(self.state_size, activation='linear'))  # type: ignore
        model.compile(loss='huber', optimizer=RMSprop(
            learning_rate=self.learning_rate))  # type: ignore
        # Paper recommends RMSprop for DQNs and Huber loss seems an unspoken standard for DQNs
        return model # type: ignore

    def act(self, normalised_hp: float) -> int:
        if normalised_hp > 0:
            if np.random.rand() <= self.epsilon:
                return rand.choice([AbstractStrategy.REGEN_ACTION, AbstractStrategy.GUARD_ACTION])
            actions = {0: AbstractStrategy.REGEN_ACTION,
                       1: AbstractStrategy.GUARD_ACTION}
            q_values = self.policy_net.predict( # type: ignore
                np.array([normalised_hp]))
            action = actions[np.argmax(q_values[0])]
        else:
            action = AbstractStrategy.EXPIRE_ACTION
        return action

    def remember(self, experiences: List[Experience], episode: int) -> None:
        ave_reward = sum(e.reward for e in experiences) / len(experiences) if len(experiences) > 0 else -2

        # the agent will only remember one final transition, not looping in the final state
        experiences = [e for e in experiences if e.transition.state != State.EXPIRE_STATE] # probably should never have to remove anything here?

        for experience in experiences:
            self.memory.append(experience)

        if self.training_counter % self.policy_net_train_freq == 0:
            trained, loss, epsilon = self.train()
            if trained:
                self.training_history = pd.concat([self.training_history, pd.DataFrame({"Episode": [episode], "Loss": [loss], "Epsilon": [epsilon], "Reward": [ave_reward]})], ignore_index=True)
                print(f"\n\rEpisode: {episode}, Loss: {loss:.4f}, Epsilon: {self.epsilon:.4f}, Ave Reward: {ave_reward:.4f}", flush=True)
        if self.training_counter % self.target_net_update_freq == 0:
            self.update_target_network()
        if self.training_counter % self.policy_net_train_freq == 0 and self.training_counter % self.target_net_update_freq == 0:
            self.training_counter = 0

        self.training_counter += 1

    def train(self, *args: Any, **kwargs: Any) -> Tuple[bool, float, float]:
        return self.replay(*args, **kwargs)

    def update_target_network(self):

        self.target_net.set_weights(
            self.policy_net.get_weights())

    def replay(self) -> Tuple[bool, float, float]:
        if len(self.memory) < self.batch_size:
            return False, 0, 0
        minibatch = rand.sample(self.memory, self.batch_size)

        states = np.array([e.transition.state.normalised_hp for e in minibatch])
        targets = self.policy_net.predict(np.array(states))

        for i, ((_, action, next_state, final), reward) in enumerate(minibatch):
            action_target = reward
            if not final:
                action_target += self.discount_factor * \
                    np.amax(self.target_net.predict(
                        np.array([next_state.normalised_hp]))[0])  
            # action - 1 conforms action code AbstractStrategy constants to prediction array subindices
            targets[i][action - 1] = action_target
        
        history: History = self.policy_net.fit(states, targets, batch_size=self.batch_size, verbose=0) # type: ignore
        loss: float = history.history['loss'][0]
        epsilon: float = self.epsilon

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        return True, loss, epsilon

    def save(self, path: str) -> None:
        self.policy_net.save(path)

    def load(self, path: str) -> None:
        model = load_model(path)
        self.policy_net = model if model is not None and isinstance(model, Sequential) else self.policy_net
        self.target_net = self.policy_net
