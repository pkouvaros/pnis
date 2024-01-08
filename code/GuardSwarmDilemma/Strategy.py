from abc import ABC, abstractmethod
import numpy as np
import random as rand
from .State import Transition
from collections import deque
from tensorflow.keras.layers import Dense
from typing import Any
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import RMSprop


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

    # @abstractmethod
    # def train(self, *args: Any, **kwargs: Any) -> None:
    #     pass

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

    # def train(self, *args: Any, **kwargs: Any) -> None:
    #     pass
        
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

    # def train(self, *args: Any, **kwargs: Any) -> None:
    #     pass

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

    # def train(self, *args: Any, **kwargs: Any) -> None:
    #     pass

class DQNStrategy(AbstractStrategy):
    DEFAULT_TRAINING_EPISODES: int = 10000

    DEFAULT_POLICY_TRAIN_FREQ: int = 10
    DEFAULT_TARGET_NET_UPDATE_FREQ: int = 100
    DEFAULT_REPLAY_BUFFER_SIZE: int = 10000
    DEFAULT_BATCH_SIZE: int = 32
    DEFAULT_LEARNING_RATE: float = 0.001
    DEFAULT_DENSE_LAYER_WIDTH: int = 10
    DEFAULT_DENSE_LAYER_DEPTH: int = 2
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
                 dense_layer_width: int = DEFAULT_DENSE_LAYER_WIDTH,
                 dense_layer_depth: int = DEFAULT_DENSE_LAYER_DEPTH,
                 discount_factor: float = DEFAULT_DISCOUNT_FACTOR,
                 epsilon: float = DEFAULT_EPSILON,
                 epsilon_min: float = DEFAULT_EPSILON_MIN,
                 epsilon_decay: float = DEFAULT_EPSILON_DECAY
                 ) -> None:
        self.state_size = 2
        self.policy_net_train_freq = policy_train_freq
        self.target_net_update_freq = target_net_update_freq
        self.memory: deque[Transition] = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dense_layer_width = dense_layer_width
        self.dense_layer_depth = dense_layer_depth
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.policy_net = self._build_model()
        self.target_net = self._build_model()
        self.update_target_network()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(self.dense_layer_width, input_dim=1, activation='relu')) # type: ignore
        for _ in range(self.dense_layer_depth - 1):
            model.add(Dense(self.dense_layer_width, activation='relu')) # type: ignore
        model.add(Dense(2, activation='softmax')) # type: ignore
        model.compile(loss='huber', optimizer=RMSprop(learning_rate=self.learning_rate)) # type: ignore # Paper recommends RMSprop for DQN and seen Huber loss used a lot
        return model
    
    def act(self, normalised_hp: float) -> int:
        if np.random.rand() <= self.epsilon:
            return rand.choice([AbstractStrategy.GUARD_ACTION, AbstractStrategy.REGEN_ACTION])
        act_values = self.policy_net.predict(normalised_hp) # type: ignore
        return np.argmax(act_values[0])

    def remember(self, transition: Transition, *, iteration: int):
        self.memory.append(Transition(transition.state, transition.action, transition.reward, transition.next_state, transition.final))

        if iteration != 0 and iteration % self.policy_net_train_freq == 0:
            self.train(iteration=iteration)

    def train(self, *args: Any, **kwargs: Any) -> None:
        self.replay(*args, **kwargs)
    
    def update_target_network(self):
        self.target_net.set_weights(self.policy_net.get_weights()) # type: ignore

    def replay(self, *, iteration: int):
        if len(self.memory) < self.batch_size:
            return
        minibatch = rand.sample(self.memory, self.batch_size)
        
        # a dead DQN agent would not learn anyway because it does not recieve rewards
        minibatch = [t for t in minibatch if t.final == False]

        states = np.array([t.state for t in minibatch])
        targets = self.policy_net.predict(states) # type: ignore

        for i, (_, action, reward, next_state, _) in enumerate(minibatch):
            action_target = reward + self.discount_factor * np.amax(self.target_net.predict(next_state)[0]) # type: ignore
            targets[i][action] = action_target

        self.policy_net.fit(states, targets, batch_size=len(minibatch)) # type: ignore

        # for state, action, reward, next_state, _ in minibatch:
        #     action_target = (reward + self.discount_factor * np.amax(self.target_net.predict(next_state)[0]))
        #     target = self.policy_net.predict(state) # type: ignore
        #     target[0][action] = action_target
        #     self.policy_net.fit(state, target) # type: ignore

        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
        
        if iteration != 0 and iteration % self.target_net_update_freq == 0:
            self.update_target_network()

    def save(self, path: str) -> None:
        self.policy_net.save(path) # type: ignore

    def load(self, path: str) -> None:
        model = load_model(path)
        self.policy_net = model if model is not None else self.policy_net