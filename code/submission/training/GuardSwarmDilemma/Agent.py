from .Strategy import AbstractStrategy
from .State import State, Transition


class Agent:
    def __init__(self, max_hp: int, initial_hp: int) -> None:
        self.max_hp: int = max_hp
        self.state = State(initial_hp, initial_hp / max_hp,
                           AbstractStrategy.REGEN_ACTION)

    def transition(self, transition: Transition) -> None:
        self.state = State(transition.next_state.hp,
                           transition.next_state.normalised_hp, transition.next_state.state_label)
