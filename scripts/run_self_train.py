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
import time

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
parser.add_argument("--stage", default=None, type=str, help="BATTLEFIELD, FINAL_DESTINATION, POKEMON_STADIUM")
args = parser.parse_args()

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
            
class ModelContainer:
    def __init__(self):
        self.models = {}
        
    def push(self, char, model_path):
        if not char in self.models:
            self.models[char] = deque(maxlen=3)
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
        self.stage = args.stage #"BATTLEFIELD"
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
        self.init_timestep = self.timesteps * 500
        self.models = models
        self.models.push(self.char, os.path.join(self.save_dir, "agent_0.pt"))
        # for i in range(500 // 50 + 1):
        #     path = os.path.join(self.save_dir, f"agent_{i * self.save_freq}.pt")
        #     if os.path.exists(path):
        #         self.models.push(self.char, path)
        #         print(path)
        
    def run(self):
        op1_char = self.char
        op1_model = random.choice(self.models.get(op1_char))
        op2_char = random.choice(list(self.models.models.keys()))
        op2_model = random.choice(self.models.get(op2_char))
        cmd = (
            f"python {self.script_path} "
            f"--iso {args.iso} "
            f"--exp_name {self.exp_name} "
            f"--init_timestep {self.init_timestep} "
            f"--timesteps {self.timesteps} "
            f"--model_path {self.recent_model} "
            f"--op1_model_path {op1_model} "
            f"--op2_model_path {op2_model} "
            f"--char {self.char} "
            f"--op1_char {op1_char} "
            f"--op2_char {op2_char} "
            f"--stage {self.stage} "
        )
        self.run_command(cmd)
    
    def can_release(self, new_model):
        #prev_model_path = os.path.join(self.save_dir, "agent_0.pt") #random.choice(self.models.get(self.char))
        prev_model_path = self.models.get(self.char)[-1]
        wins, loses = 0, 0
        for _ in range(5):
            kill_dolphin()
            w, l = self.parallel_match(self.char, new_model, self.char, prev_model_path, self.stage, parallel_num=2)
            wins += w; loses += l
        #wins, loses = self.parallel_match(self.char, new_model, self.char, prev_model_path, self.stage, parallel_num=10)
        kill_dolphin()
        print(f"{self.char} wins: {wins} , loses {loses}")
        if wins >= 6:
            return True
        else:
            return False
        
    def run_command(self, cmd):
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate(timeout=60 * 5)
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
    
    def make_env(self, players, stage):
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

    def match(self, p1_char, p1_model_path, p2_char, p2_model_path, stage):
        players = [MyAgent(getattr(enums.Character, p1_char)), 
                MyAgent(getattr(enums.Character, p2_char))]
        env = self.make_env(players=players, stage=stage)
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
        agent_ppo.load(p1_model_path)
        agent_ppo.set_mode("eval")
        agent_ppo.set_running_mode("eval")
        agent_ppo.init()
        
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
        op_ppo.load(p2_model_path)
        op_ppo.set_mode("eval")
        op_ppo.set_running_mode("eval")
        op_ppo.init()
        
        state, info = env.reset()
        done = False
        steps = 0
        while not done and steps < 28500:
            with torch.no_grad():
                action, _ = agent_ppo.act(state, 1, 0)
                op_action, _ = op_ppo.act(state, 1, 0)
            next_state, reward, done, truncated, info = env.step((action, op_action))
            state = next_state
            steps += 1
        env.close()
        if state.player[1].stock > state.player[2].stock:
            return 1
        elif state.player[1].stock < state.player[2].stock:
            return -1
        else:
            return 0
        
    def parallel_match(self, p1_char, p1_model_path, p2_char, p2_model_path, stage, parallel_num=3):
        futures = []
        for _ in range(parallel_num):
            futures.append((self.match, p1_char, p1_model_path, p2_char, p2_model_path, stage))
        with ProcessPoolExecutor(max_workers=parallel_num) as executor:
            futures = [executor.submit(*x) for x in futures]
        p1_wins = 0
        p2_wins = 0
        for future in futures:
            try:
                if future.result() == 1:        
                    p1_wins += 1
                elif future.result() == -1:
                    p2_wins += 1
            except Exception as e:
                print(f"error ouccured: {e}")
        return p1_wins, p2_wins

if __name__ == "__main__":
    models = ModelContainer()
    MAX_NUMS = 10000
    trainers = {}
    chars =["LUIGI", "LINK", "PIKACHU", "MARIO", "YOSHI"]
    for char in chars:    
        model_path = f"/home/tgkang/saved_model/selfplay_BF_0/{char}_BF.pt"
        trainers[char] = Selfplay(model_path=model_path, exp_name=f"../SelfplayBF/{char}", char=char, models=models)
    trainers[char] = Selfplay(model_path=model_path, exp_name=f"../SelfplayBF/{char}", char=char, models=models)
    
    for i in range(MAX_NUMS):
        print("Iter: ", i)
        kill_dolphin()
        os.system("kill $(ps aux | grep 'python ./self_train' | awk '{print $2}')")
        os.system('python kill.py')
        futures = []
        for char in chars:
            futures.append(trainers[char].run)
        with ProcessPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(x) for x in futures]
        kill_dolphin()
        for char in chars:
            s = trainers[char]
            s.init_timestep += s.timesteps
            exp_dir = os.path.join('.', s.exp_name)
            new_model = os.path.join(exp_dir,"checkpoints",f"agent_{s.init_timestep}.pt")
            if os.path.exists(new_model):
                shutil.copy2(new_model, s.recent_model)
                if not s.init_timestep % s.save_freq == 0:
                    os.remove(new_model)
                else:
                    if not s.can_release(new_model):
                        os.remove(new_model)
                        continue
                    print(f":::Release new agent: {new_model}")
                    s.models.push(char, new_model)
            # when process is terminated before evaluation
            elif s.init_timestep % s.save_freq == 0:
                shutil.copy2(s.recent_model, new_model)
                if not s.can_release(new_model):
                    os.remove(new_model)
                    continue
                print(f":::Release new agent: {new_model}")
                s.models.push(char, new_model)
        kill_dolphin()