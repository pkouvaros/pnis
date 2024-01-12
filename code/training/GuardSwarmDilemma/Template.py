from .Strategy import DQNStrategy, AbstractStrategy

class TemplateAgent:
    DEFAULT_INITIAL_HP: int = 8
    DEFAULT_MAX_HP: int = 8

    def __init__(self, max_hp: int = DEFAULT_MAX_HP, initial_hp: int = DEFAULT_INITIAL_HP, strategy: AbstractStrategy = DQNStrategy()) -> None:
        if initial_hp > max_hp:
            raise ValueError('initial_hp must be less than or equal to max_hp')
        self.max_hp: int = max_hp
        self.initial_hp: int = initial_hp
        self.strategy: AbstractStrategy = strategy