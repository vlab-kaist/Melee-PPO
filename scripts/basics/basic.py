import code
from abc import ABC, abstractmethod

import numpy as np
from melee import enums
from basics.util import *
import gymnasium as gym

class Agent(ABC):
    def __init__(self):
        self.agent_type = "AI"
        self.controller = None
        self.port = None  # this is also in controller, maybe redundant?
        self.action = 0
        self.press_start = False
        self.self_observation = None
        self.current_frame = 0

    @abstractmethod
    def act(self):
        pass

class AgentChooseCharacter(Agent):
    def __init__(self, character):
        super().__init__()
        self.character = character

class MyAgent(AgentChooseCharacter):
    def __init__(self, character):
        super().__init__(character)
        self.action_space = ActionSpace()
    
    def act(self, gamestate):
        pass

class Human(Agent):
    def __init__(self):
        super().__init__()
        self.agent_type = "HMN"

    def act(self, gamestate):
        pass

class CPU(AgentChooseCharacter):
    def __init__(self, character, lvl):
        super().__init__(character)
        self.agent_type = "CPU"
        if not 1 <= lvl <= 9:
            raise ValueError(f"CPU Level must be 1-9. Got {lvl}")
        self.lvl = lvl

    def act(self, gamestate):
        pass