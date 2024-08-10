import argparse
import os
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.wrappers import TransformObservation
import psutil
import shutil
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import random
from collections import deque

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
from basics.env import *
from basics.basic import *
from basics.util import *
from basics.ppo_agent import PPOGRUAgent
from basics.model import GRUPolicy, GRUValue

parser = argparse.ArgumentParser()
parser.add_argument(
    "--iso", default="/home/tgkang/ssbm.iso", type=str, help="Path to your NTSC 1.02/PAL SSBM Melee ISO"
)
args = parser.parse_args()

class ModelContainer:
    def __init__(self):
        self.models = {}
        chars = ["DOC", "MARIO", "YOSHI", "LUIGI", "PIKACHU", "LINK"]
        for char in chars:
            self.models[char] = deque(maxlen=10)
    def push(self, char, model_path):
        self.models[char].append(model_path)
    def get(self, char):
        return self.models[char]
    
class Selfplay:
    def __init__(self, model_path, exp_name, char, models):
        self.config = {
            "iso_path": args.iso,
            "stage": enums.Stage.FINAL_DESTINATION,
            "players": None,
            "agent_id": 1,
            "n_states": 864,
            "n_actions": 30,
            "save_replay": False,
            "timesteps": 18000 # total timesteps for training
        }
        self.save_freq = self.config["timesteps"] * 50
        self.char = char
        self.exp_name = exp_name
        self.save_dir = os.path.abspath(os.path.join('.', exp_name, "checkpoints"))
        self.recent_model = os.path.join(self.save_dir, "recent_model.pt")
        if not os.path.exists(os.path.join('.', exp_name)):
            os.makedirs(os.path.join('.', exp_name))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        shutil.copy2(model_path, self.recent_model)
        shutil.copy2(model_path, os.path.join(self.save_dir, "agent_0.pt"))
        self.timesteps = 0
        self.models = models
        self.models.push(self.char, os.path.join(self.save_dir, "agent_0.pt"))
        
    def train(self, cpu=None, actor_model=None, actor_char=None):
        self.config["players"] = [MyAgent(getattr(enums.Character, self.char)), MyAgent(getattr(enums.Character, actor_char))]
        self.config["actor_model"] = actor_model
        env = gym.vector.AsyncVectorEnv([
            lambda: self.make_selfplay_env(self.config),
            # lambda: self.make_cpu_env(self.config, 5),
            # lambda: self.make_cpu_env(self.config, 8),
            ])
        #env = self.make_env("SelfPlayMeleeEnv", self.config)
        env = wrap_env(env, wrapper="gymnasium")
        device = env.device
        memory = RandomMemory(memory_size=1024, num_envs=env.num_envs, device=device)
        cfg_ppo = PPO_DEFAULT_CONFIG.copy()
        cfg_ppo["rollouts"] = 1024  # memory_size
        cfg_ppo["learning_epochs"] = 10
        cfg_ppo["mini_batches"] = 8
        cfg_ppo["discount_factor"] = 0.99
        cfg_ppo["lambda"] = 0.95
        cfg_ppo["learning_rate"] = 1e-5  # Lower the learning rate
        cfg_ppo["learning_rate_scheduler"] = KLAdaptiveRL
        cfg_ppo["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
        cfg_ppo["grad_norm_clip"] = 1.0
        cfg_ppo["ratio_clip"] = 0.2
        cfg_ppo["value_clip"] = 0.2
        cfg_ppo["clip_predicted_values"] = False
        cfg_ppo["entropy_loss_scale"] = 0.01
        cfg_ppo["value_loss_scale"] = 0.5
        cfg_ppo["kl_threshold"] = 0
        cfg_ppo["value_preprocessor"] = RunningStandardScaler
        cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}
        cfg_ppo["value_preprocessor"] = RunningStandardScaler
        cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}
        cfg_ppo["experiment"]["write_interval"] = 1024
        cfg_ppo["experiment"]["checkpoint_interval"] = self.config["timesteps"]
        
        learner_model = {}
        learner_model["policy"] = GRUPolicy(env.observation_space, env.action_space, device, num_envs=env.num_envs, 
                                        num_layers=4, hidden_size=512, ffn_size=512, sequence_length=64)
        learner_model["value"] = GRUValue(env.observation_space, env.action_space, device, num_envs=env.num_envs,
                                        num_layers=4, hidden_size=512, ffn_size=512, sequence_length=64)
        
        learner_agent = PPOGRUAgent(
                models=learner_model,
                memory=memory,
                cfg=cfg_ppo,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device, 
                agent_id = 1,
            )
        learner_agent.load(self.recent_model)
        
        learner_agent.experiment_dir = os.path.join('.',self.exp_name)
        cfg_trainer = {"timesteps": self.config["timesteps"], "headless": True, "disable_progressbar": True}
        trainer = ParallelTrainer(cfg=cfg_trainer, env=env, agents=learner_agent)
        trainer.initial_timestep = self.timesteps
        trainer.timesteps += self.timesteps
        trainer.train()
            
    def make_selfplay_env(self, config):
        register(
            id='SelfPlayMeleeEnv',
            entry_point=f'basics.env:SelfPlayMeleeEnv',
            kwargs={'config': config},
        )
        return gym.make('SelfPlayMeleeEnv')
    
    def make_cpu_env(self, config, lvl):
        candidates = ["DOC", "MARIO", "YOSHI", "LUIGI", "PIKACHU", "LINK"]
        config["players"][1] = CPU(getattr(enums.Character, random.choice(candidates)), lvl)
        register(
            id="CPUMeleeEnv",
            entry_point=f'basics.env:CPUMeleeEnv',
            kwargs={'config': config},
        )
        return gym.make("CPUMeleeEnv")
    
    def kill_dolphin(self):
        # clear dolphin
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
    
    def run(self):
        self.kill_dolphin()
        op_char = random.choice(["DOC", "MARIO", "YOSHI", "LUIGI", "PIKACHU", "LINK"])
        op_model = random.choice(self.models.get(op_char))
        self.train(actor_model=op_model, actor_char=op_char)
        self.kill_dolphin()

if __name__ == "__main__":
    models = ModelContainer()
    MAX_NUMS = 10000
    trainers = {}
    chars = ["DOC", "MARIO", "YOSHI", "LUIGI", "PIKACHU", "LINK"]
    for char in chars:    
        model_path = f"/home/tgkang/saved_model/aginst_cpu_FD/{char}_cpu.pt"
        trainers[char] = Selfplay(model_path=model_path, exp_name=char, char=char, models=models)
    for i in range(MAX_NUMS):
        print("Iter: ", i)
        futures = []
        for char in chars:
            futures.append(trainers[char].run)
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(x) for x in futures]
        for char in chars:
            s = trainers[char]
            s.timesteps += s.config["timesteps"]
            exp_dir = os.path.join('.',s.exp_name)
            new_model = os.path.join(exp_dir,"checkpoints",f"agent_{s.timesteps}.pt")
            if os.path.exists(new_model):
                shutil.copy2(new_model, s.recent_model)
                if not s.timesteps % s.save_freq == 0:
                    os.remove(new_model)
                else:
                    s.models.push(new_model)