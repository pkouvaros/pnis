from .Strategy import RandomStrategy, AbstractStrategy

class TemplateAgent:
    DEFAULT_INITIAL_HP: int = 10
    DEFAULT_MAX_HP: int = 10

    def __init__(self, max_hp: int = DEFAULT_MAX_HP, initial_hp: int = DEFAULT_INITIAL_HP, strategy: AbstractStrategy = RandomStrategy()) -> None:
        self.max_hp: int = max_hp
        self.initial_hp: int = initial_hp
        self.strategy: AbstractStrategy = strategy