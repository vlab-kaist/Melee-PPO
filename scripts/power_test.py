import argparse
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.wrappers import TransformObservation
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

# Import the skrl components to build the RL system
from skrl.models.torch import Model, CategoricalMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_RNN, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer, ParallelTrainer
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.envs.torch import wrap_env

import sys
import os

import melee
from melee import enums
from basics.env import *
from basics.basic import *
from basics.util import *
from basics.ppo_agent import PPOAgent, PPOGRUAgent
from basics.model import Policy, Value, GRUPolicy, GRUValue

parser = argparse.ArgumentParser()
parser.add_argument(
    "--iso", default="/home/tgkang/ssbm.iso", type=str, help="Path to your NTSC 1.02/PAL SSBM Melee ISO"
)
args = parser.parse_args()


def make_env(id, cpu_lvl):
    players = [MyAgent(enums.Character.YOSHI), CPU(enums.Character.PIKACHU, cpu_lvl)]
    register(
        id=id,
        entry_point=f'basics.env:{id}',
        kwargs={'config': {
            "iso_path": args.iso,
            "players": players,
            "agent_id": 1, # for 1p,
            "n_states": 885, #869
            "n_actions": 29,
            "save_replay": True,
            "stage": stage,
        }},
    )
    return gym.make(id)

def match(model_path, lvl):
    env = make_env(id="MultiMeleeEnv", cpu_lvl=lvl)
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    models_ppo = {}
    models_ppo["policy"] = GRUPolicy(env.observation_space, env.action_space, device, num_envs=1,
                                    num_layers=4, hidden_size=512, ffn_size=512, sequence_length=64)
    models_ppo["value"] = GRUValue(env.observation_space, env.action_space, device, num_envs=1,
                                    num_layers=4, hidden_size=512, ffn_size=512, sequence_length=64) 

    agent_ppo = PPOGRUAgent(models=models_ppo,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    device=device, 
                    agent_id = 1,
                    platform=True)

    agent_ppo.load(model_path)
    agent_ppo.set_mode("eval")
    agent_ppo.init()
    state, info = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            action, _ = agent_ppo.act(state, 1, 0)
        next_state, reward, done, truncated, info = env.step((action, 0))
        state = next_state
    env.close()
    if state.player[1].stock > state.player[2].stock:
        return 1
    elif state.player[1].stock < state.player[2].stock:
        return -1

PARALLEL_NUM = 5
model_path = "/home/tgkang3/Melee-PPO/scripts/PlatformYOSHI/checkpoints/recent_model.pt"
wins = [0] * 9
loses = [0] * 9

for lvl in range(1, 10):
    futures = []
    for _ in range(PARALLEL_NUM):
        futures.append((match, model_path, lvl))
    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(*x) for x in futures]
    for future in futures:
        try:
            if future.result() == 1:        
                wins[lvl - 1] += 1
            elif future.result() == -1:
                loses[lvl - 1] += 1
        except Exception as e:
            print(f"error ouccured: {e}")
    futures = []
    for _ in range(PARALLEL_NUM):
        futures.append((match, model_path, lvl))
    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(*x) for x in futures]
    for future in futures:
        try:
            if future.result() == 1:        
                wins[lvl - 1] += 1
            elif future.result() == -1:
                loses[lvl - 1] += 1
        except Exception as e:
            print(f"error ouccured: {e}")
print("wins: ", wins)
print("loses: ", loses)