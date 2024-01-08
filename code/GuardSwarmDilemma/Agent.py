from .Strategy import RandomStrategy, AbstractStrategy
from .State import Transition

class Agent:
    GUARD_STATE: int = 2
    REGEN_STATE: int = 1
    EXPIRE_STATE: int = 0

    def __init__(self, max_hp, initial_hp, strategy: AbstractStrategy = RandomStrategy()) -> None:
        self.max_hp: int = max_hp
        self.hp: float = initial_hp
        self.strategy: AbstractStrategy = strategy
        self.state_label: int = Agent.REGEN_STATE

    def act(self) -> int:
        return self.strategy.act(normalised_hp=self.hp/self.max_hp)
    
    def remember(self, transition: Transition, iteration: int) -> None:
        self.strategy.remember(transition, iteration=iteration)

    def update_state(self, transition: Transition) -> None:
        self.hp = transition.next_state.hp
        self.state_label = transition.next_state.state_label
