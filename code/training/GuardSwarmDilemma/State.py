from typing import NamedTuple


class State:
    GUARD_STATE: int = 2
    REGEN_STATE: int = 1
    EXPIRE_STATE: int = 0

    def __init__(self, hp: int, normalised_hp: float, state_label: int) -> None:
        self.hp: int = hp
        self.normalised_hp = normalised_hp
        self.state_label: int = state_label


class Transition(NamedTuple):
    state: State
    action: int
    next_state: State
    final: bool


class Experience(NamedTuple):
    transition: Transition
    reward: float