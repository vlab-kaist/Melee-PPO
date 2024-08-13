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
import subprocess
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
        
    def push(self, char, model_path):
        if not char in self.models:
            self.models[char] = deque(maxlen=10)
            self.models[char].append(model_path)
        else:
            self.models[char].append(model_path)
            
    def get(self, char):
        return self.models[char]
    
class Selfplay:
    def __init__(self, model_path, exp_name, char, models):
        self.timesteps = 18000
        self.save_freq = self.timesteps * 50
        self.char = char
        self.stage = "FINAL_DESTINATION"
        self.script_path = "./self_train.py"
        
        self.exp_name = exp_name
        self.save_dir = os.path.abspath(os.path.join('.', exp_name, "checkpoints"))
        self.recent_model = os.path.join(self.save_dir, "recent_model.pt")
        if not os.path.exists(os.path.join('.', exp_name)):
            os.makedirs(os.path.join('.', exp_name))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        shutil.copy2(model_path, self.recent_model)
        shutil.copy2(model_path, os.path.join(self.save_dir, "agent_0.pt"))
        self.init_timestep = 0
        self.models = models
        self.models.push(self.char, os.path.join(self.save_dir, "agent_0.pt"))
        
    def run(self):
        op_char = random.choice(list(self.models.models.keys()))
        op_model = random.choice(self.models.get(op_char))
        cmd = (
            f"python {self.script_path} "
            f"--iso {args.iso} "
            f"--exp_name {self.exp_name} "
            f"--init_timestep {self.init_timestep} "
            f"--timesteps {self.timesteps} "
            f"--model_path {self.recent_model} "
            f"--op_model_path {op_model} "
            f"--char {self.char} "
            f"--op_char {op_char} "
            f"--stage {self.stage} "
        )
        self.run_command(cmd)
    
    def run_command(self, cmd):
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate(timeout=60 * 6)
            return_code = process.returncode

            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd, output=stdout, stderr=stderr)
        except subprocess.CalledProcessError as e:
            print('The command failed with exit code', e.returncode)
            print('Error output:', e.stderr)
        except subprocess.TimeoutExpired as e:
            print('The command timed out and was terminated.')
            process.kill()
        except Exception as e:
            print('An unexpected error occurred:', e)
        finally:
            print('Command finished.')

def kill_dolphin():
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

if __name__ == "__main__":
    models = ModelContainer()
    MAX_NUMS = 10000
    trainers = {}
    chars = ["DOC", "MARIO", "YOSHI", "LUIGI", "PIKACHU", "LINK"]
    for char in chars:    
        model_path = f"/home/tgkang/saved_model/against_cpu_FD/{char}_FD.pt"
        trainers[char] = Selfplay(model_path=model_path, exp_name=char, char=char, models=models)
    for i in range(MAX_NUMS):
        print("Iter: ", i)
        kill_dolphin()
        futures = []
        for char in chars[:3]:
            futures.append(trainers[char].run)
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(x) for x in futures]
        
        kill_dolphin()
        futures = []
        for char in chars[3:]:
            futures.append(trainers[char].run)
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(x) for x in futures]
        for char in chars:
            s = trainers[char]
            s.init_timestep += s.timesteps
            exp_dir = os.path.join('.',s.exp_name)
            new_model = os.path.join(exp_dir,"checkpoints",f"agent_{s.timesteps}.pt")
            if os.path.exists(new_model):
                shutil.copy2(new_model, s.recent_model)
                if not s.init_timestep % s.save_freq == 0:
                    os.remove(new_model)
                    print("YEAH")
                else:
                    s.models.push(new_model)
        kill_dolphin()