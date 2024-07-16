import argparse
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.wrappers import TransformObservation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

# Import the skrl components to build the RL system
from skrl.models.torch import Model, CategoricalMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer, ParallelTrainer
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.envs.torch import wrap_env

import melee
from melee import enums
from melee_env.myenv import MeleeEnv, MyMeleeEnv
from melee_env.agents.basic import *
from melee_env.agents.util import (
    ObservationSpace,
    MyActionSpace
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--iso", default="/home/tgkang/ssbm.iso", type=str, help="Path to your NTSC 1.02/PAL SSBM Melee ISO"
)

args = parser.parse_args()

class PPOAgent(PPO):
    def __init__(self, agent_id=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_id = agent_id
        self.action = torch.tensor([[0]], device=self.device)
        self.action_cnt = 3
        
    def act(self, states, timestep: int, timesteps: int) -> torch.Tensor:
        obs = states
        # apply same action for 3 frames (Daboy style)
        if self.action_cnt >= 3:   
            # sample stochastic actions
            actions, log_prob, outputs = self.policy.act({"states": self._state_preprocessor(obs)}, role="policy")
            self._current_log_prob = log_prob
            self.action = actions
            self.action_cnt = 1
            # need to apply return macro
            # if sum(obs[0][36 + 29: 36 + 37 + 1]) and obs[0][16]:
            #     print(":::EMERGENCY")
            #     isleft = False if obs[0][0] > 0 else True 
            #     if isleft:
            #         self.action = torch.tensor([[17]], device=self.device)
            #     else:
            #         self.action = torch.tensor([[16]], device=self.device)
        else:
            self.action_cnt += 1
        return self.action, self._current_log_prob# torch.tensor([[0]], device=self.device)
        
    
class Policy(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)

        self.linear_layer_1 = nn.Linear(self.num_observations, 256)
        self.linear_layer_2 = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, self.num_actions)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(inputs["states"]))
        x = F.relu(self.linear_layer_2(x))
        return self.output_layer(x), {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.net.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


iso_path = args.iso

def make_env(cpu_lvl):
    players = [YoshiAgent(enums.Character.YOSHI), CPU(enums.Character.FOX, cpu_lvl)]

    # register the environment
    register(
        id='MyMeleeEnv',
        entry_point='melee_env.myenv:MyMeleeEnv',
        kwargs={'config': {
            "iso_path": iso_path,
            "players": players,
            "agent_id": 1, # for 1p,
            "n_states": 808,
            "n_actions": 25,
            "save_replay": True
        }},
    )
    return gym.make("MyMeleeEnv")
    
# env = gym.vector.AsyncVectorEnv([
#     lambda: make_env(1),
#     lambda: make_env(2),
#     lambda: make_env(3),
#     lambda: make_env(4),
#     lambda: make_env(5),
#     lambda: make_env(6),
#     lambda: make_env(7),
#     lambda: make_env(8),
#     lambda: make_env(9)
# ])
env = make_env(cpu_lvl=9)
env = wrap_env(env, wrapper='gymnasium')
device = env.device


# Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=8192, num_envs=env.num_envs, device=device)

models_ppo = {}
models_ppo["policy"] = Policy(env.observation_space, env.action_space, device)
models_ppo["value"] = Value(env.observation_space, env.action_space, device)

cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["state_preprocessor"] = RunningStandardScaler
cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg_ppo["value_preprocessor"] = RunningStandardScaler
cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}

agent_ppo = PPOAgent(models=models_ppo,
                memory=memory,
                cfg=cfg_ppo,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device, 
                agent_id = 1)

#model_path = "/home/tgkang/multi-env/skrl-ssbm/NewActionSpace/24-07-13_03-31-26-444904_PPOAgent/checkpoints/agent_15564800.pt"
model_path =  "/home/tgkang/multi-env/skrl-ssbm/NewActionSpace/24-07-16_16-11-23-221821_PPOAgent/checkpoints/agent_1228800.pt"
agent_ppo.load(model_path)
agent_ppo.set_running_mode("eval")

state, info = env.reset()
done = False
while not done:
    with torch.no_grad():
        action, _ = agent_ppo.act(state, 1, 0)
    next_state, reward, done, truncated, info = env.step(action[0])
    state = next_state
    