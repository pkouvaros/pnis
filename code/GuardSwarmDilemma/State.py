from typing import NamedTuple

class State:

    def __init__(self, hp: int, normalised_hp: float, state: int) -> None:
        self.hp: int = hp
        self.normalised_hp = normalised_hp
        self.state_label: int = state

class Transition(NamedTuple):
    state: State
    action: int
    reward: float
    next_state: State
    final: bool