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

import melee
from melee import enums
from basics.env import *
from basics.basic import *
from basics.util import *
from basics.ppo_agent import PPOGRUAgent
from basics.model import Policy, Value, GRUPolicy, GRUValue

parser = argparse.ArgumentParser()
parser.add_argument(
    "--iso", default="E:/Projects/ssbm.iso", type=str, help="Path to your NTSC 1.02/PAL SSBM Melee ISO"
)
parser.add_argument(
    "--char", default=None, type=str, help="my character"
)
parser.add_argument(
    "--model_path", default=None, type=str, help="model for evaluate"
)
parser.add_argument(
    "--stage", default="FINAL_DESTINATION", type=str, help="stages to play"
)
args = parser.parse_args()

class Powertest:
    def __init__(self, save_replay=False):
        self.config = {
            "iso_path": args.iso,
            "stage": getattr(enums.Stage, args.stage),
            "players": None,
            "agent_id": 1,
            "n_states": 864 if args.stage == "FINAL_DESTINATION" else 880,
            "n_actions": 36, # 25
            "save_replay": save_replay
        }
    
    def make_env(self, id, players):
        self.config["players"] = players
        register(
            id=id,
            entry_point=f'basics.env:{id}',
            kwargs={'config': self.config},
        )
        return gym.make(id)
    
    def kill_dolphin(self):
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

    def match(self, lvl=None):
        players = [MyAgent(getattr(enums.Character, args.char)), 
                Human()]
        env = self.make_env(id="MultiMeleeEnv", players=players)
        device = torch.device("cpu") #torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

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
                        platform=False if args.stage == "FINAL_DESTINATION" else True)
        agent_ppo.load(args.model_path)
        agent_ppo.set_mode("eval")
        agent_ppo.set_running_mode("eval")
        agent_ppo.init()
        
        state, info = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                action, _ = agent_ppo.act(state, 1, 0)
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
                
if __name__ == "__main__":
    P = Powertest(save_replay=True)
    P.match()