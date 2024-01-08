from typing import NamedTuple
class State:

    def __init__(self, normalised_hp: float, state: int) -> None:
        self.hp: float = normalised_hp
        self.state_label: int = state

class Transition(NamedTuple):
    state: State
    action: int
    reward: float
    next_state: State
    final: bool