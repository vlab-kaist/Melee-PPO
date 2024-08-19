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
import csv
import psutil
import matplotlib.pyplot as plt  # Uncomment if plotting is required
import seaborn as sns

import melee
from melee import enums
from basics.env import *
from basics.basic import *
from basics.util import *
from basics.ppo_agent import PPOGRUAgent
from basics.model import Policy, Value, GRUPolicy, GRUValue
from power_test import Powertest

parser = argparse.ArgumentParser()
parser.add_argument(
    "--iso", default="/home/tgkang2/ssbm.iso", type=str, help="Path to your NTSC 1.02/PAL SSBM Melee ISO"
)
args = parser.parse_args()

class Player:
    def __init__(self, char=None, model_path=None, cpu_lvl=None, id=None):
        self.id = id
        self.char = char
        self.agent_type = "AI" if model_path is not None else "CPU"
        self.model_path = model_path
        self.cpu_lvl = cpu_lvl
        self.wins = {} # dict

def make_env(players, stage):
    config = {
            "iso_path": args.iso,
            "stage": getattr(enums.Stage, stage),
            "players": None,
            "n_states": 864 if stage == "FINAL_DESTINATION" else 880,
            "n_actions": 36, # 25
            "save_replay": False
        }
    config["players"] = players
    register(
        id='myenv',
        entry_point='basics.env:MultiMeleeEnv',
        kwargs={'config': config},
    )
    return gym.make('myenv')

def kill_dolphin():
    current_user = os.getlogin()
    for proc in psutil.process_iter(['pid', 'username', 'name']):
        try:
            if proc.info['username'] == current_user and proc.name() == "dolphin-emu":
                parent_pid = proc.pid
                parent = psutil.Process(parent_pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
def match(p1, p2, stage):
    players = [MyAgent(getattr(enums.Character, p1.char)), 
            MyAgent(getattr(enums.Character, p2.char)) if p2.agent_type == "AI" else CPU(getattr(enums.Character, p2.char), p2.cpu_lvl)]
    env = make_env(players=players, stage=stage)
    device = torch.device("cpu")

    models_ppo = {}
    models_ppo["policy"] = GRUPolicy(env.observation_space, env.action_space, device, num_envs=1,
                                    num_layers=4, hidden_size=512, ffn_size=512, sequence_length=64)
    models_ppo["value"] = GRUValue(env.observation_space, env.action_space, device, num_envs=1,
                                    num_layers=4, hidden_size=512, ffn_size=512, sequence_length=64) 

    agent_ppo = PPOGRUAgent(models=models_ppo,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    device=device, 
                    agent_id=1,
                    platform=False if stage == "FINAL_DESTINATION" else True)
    agent_ppo.load(p1.model_path)
    agent_ppo.set_mode("eval")
    agent_ppo.set_running_mode("eval")
    agent_ppo.init()
    
    if p2.agent_type == "AI":
        op_models_ppo = {}
        op_models_ppo["policy"] = GRUPolicy(env.observation_space, env.action_space, device, num_envs=1,
                                        num_layers=4, hidden_size=512, ffn_size=512, sequence_length=64)
        op_models_ppo["value"] = GRUValue(env.observation_space, env.action_space, device, num_envs=1,
                                        num_layers=4, hidden_size=512, ffn_size=512, sequence_length=64)
        op_ppo = PPOGRUAgent(models=op_models_ppo,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    device=device, 
                    agent_id=2, 
                    platform=False if stage == "FINAL_DESTINATION" else True)
        op_ppo.load(p2.model_path)
        op_ppo.set_mode("eval")
        op_ppo.set_running_mode("eval")
        op_ppo.init()
    
    state, info = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            action, _ = agent_ppo.act(state, 1, 0)
            if p2.agent_type == "AI":
                op_action, _ = op_ppo.act(state, 1, 0)
            else:
                op_action = 0
        next_state, reward, done, truncated, info = env.step((action, op_action))
        state = next_state
    env.close()

    if state.player[1].stock > state.player[2].stock:
        return 1
    elif state.player[1].stock < state.player[2].stock:
        return -1
    else:
        return 0

def parallel_match(p1, p2, stage, parallel_num=1):
    futures = []
    for _ in range(10):
        futures.append((match, p1, p2, stage))
    with ProcessPoolExecutor(max_workers=parallel_num) as executor:
        futures = [executor.submit(*x) for x in futures]
    p1.wins[p2.id] = 0
    p2.wins[p1.id] = 0
    for future in futures:
        try:
            if future.result() == 1:        
                p1.wins[p2.id] += 1
            elif future.result() == -1:
                p2.wins[p1.id] += 1
        except Exception as e:
            print(f"error ouccured: {e}")
    print(f"{p1.id} vs {p2.id} {p1.wins[p2.id]}:{p2.wins[p1.id]}")
        
#characters = ["DOC", "LINK", "LUIGI", "MARIO", "PIKACHU", "YOSHI"]
agents = [
    Player(char="LINK", cpu_lvl=9, id=0),
    Player(char="LINK", model_path="/home/tgkang/Melee-PPO/scripts/Selfplay2/LINK/checkpoints/agent_0.pt", id=1),
    Player(char="LINK", model_path="/home/tgkang/Melee-PPO/scripts/Selfplay2/LINK/checkpoints/agent_5400000.pt", id=2),
    Player(char="LINK", model_path="/home/tgkang/Melee-PPO/scripts/Selfplay2/LINK/checkpoints/agent_11700000.pt", id=3),
]
for i in range(len(agents)):
    for j in range(i + 1, len(agents)):
        if agents[i].agent_type == "CPU":
            if agents[j].agent_type == "CPU":
                continue 
            parallel_match(agents[j], agents[i], stage="FINAL_DESTINATION")
        else:
            parallel_match(agents[i], agents[j], stage="FINAL_DESTINATION")
        kill_dolphin()

win_matrix = np.zeros((len(agents), len(agents)), dtype=int)
for agent in agents:
    for opp_id, win in agent.wins.items():
        win_matrix[agent.id, opp_id] = win
print("Win Matrix:")
print(win_matrix)

for agent in agents:
    if agent.agent_type == "AI":
        print(f"{agent.id}: {agent.char}, {agent.model_path}")
    else:
        print(f"{agent.id}: {agent.char}, CPU Level {agent.cpu_lvl}")
# visuallize win martix
plt.figure(figsize=(10, 8))
sns.heatmap(win_matrix, annot=True, fmt="d", cmap="coolwarm", xticklabels=[f'{id}' for id in range(len(agents))], yticklabels=[f'{id}' for id in range(len(agents))], cbar=True)
plt.title("Win Matrix")
plt.xlabel("Opponent")
plt.ylabel("Player")
plt.savefig('win_matrix.png', bbox_inches='tight')
plt.close()
