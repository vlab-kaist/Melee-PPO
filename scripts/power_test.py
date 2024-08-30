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
    "--iso", default="/home/tgkang/ssbm.iso", type=str, help="Path to your NTSC 1.02/PAL SSBM Melee ISO"
)
parser.add_argument(
    "--char", default=None, type=str, help="my character"
)
parser.add_argument(
    "--model_path", default=None, type=str, help="model for evaluate"
)
parser.add_argument(
    "--op_char", default=None, type=str, help="opponent character"
)
parser.add_argument(
    "--op_model_path", default=None, type=str, help="opponent model path"
)
parser.add_argument(
    "--stage", default="FINAL_DESTINATION", type=str, help="stages to play"
)
args = parser.parse_args()
# ex) python power_test.py --char DOC --model_path /home/tgkang/saved_model/aginst_cpu_FD/DOC_cpu.pt 
# --op_char LUIGI --op_model_path /home/tgkang/saved_model/aginst_cpu_FD/LUIGI_cpu.pt
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
        if args.op_model_path is None:      
            players = [MyAgent(getattr(enums.Character, args.char)), 
                    CPU(getattr(enums.Character, args.op_char), lvl)]
        else:
            players = [MyAgent(getattr(enums.Character, args.char)), 
                    MyAgent(getattr(enums.Character, args.op_char))]
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
        if args.op_model_path is not None:
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
                        platform=False if args.stage == "FINAL_DESTINATION" else True)
            op_ppo.load(args.op_model_path)
            op_ppo.set_mode("eval")
            op_ppo.set_running_mode("eval")
            op_ppo.init()
        
        state, info = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                action, _ = agent_ppo.act(state, 1, 0)
                op_action = 0
                if args.op_model_path is not None:
                    op_action, _ = op_ppo.act(state, 1, 0)
            next_state, reward, done, truncated, info = env.step((action, op_action))
            state = next_state
        env.close()
        if state.player[1].stock > state.player[2].stock:
            return 1
        elif state.player[1].stock < state.player[2].stock:
            return -1
        else:
            return 0
    
    def run_test(self, lvl=None):
        """
        match 10 times and return wins, loses
        """
        self.kill_dolphin()
        wins = 0
        loses = 0
        if lvl is not None:
            futures = []
            for _ in range(10):
                futures.append((self.match, lvl))
            with ProcessPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(*x) for x in futures]
            for future in futures:
                try:
                    if future.result() == 1:        
                        wins += 1
                    elif future.result() == -1:
                        loses += 1
                except Exception as e:
                    print(f"error ouccured: {e}")
        else:
            futures = []
            for _ in range(10):
                futures.append((self.match, ))
            with ProcessPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(*x) for x in futures]
            for future in futures:
                try:
                    if future.result() == 1:        
                        wins += 1
                    elif future.result() == -1:
                        loses += 1
                except Exception as e:
                    print(f"error ouccured: {e}")
        self.kill_dolphin()
        return wins, loses
            
    def run_cpu_test(self):
        """
        match with all cpus 10 times and return wins, loses
        """
        self.kill_dolphin()
        wins = [0] * 3
        loses = [0] * 3
        for lvl in range(7, 10):
            w, l = self.run_test(lvl)
            wins[lvl - 7] = w
            loses[lvl - 7] = l
            print(wins)
            print(loses)
        directory_path = args.model_path[:-3]
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        output_file = directory_path + f"/{args.op_char}.csv"
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Level", "Wins", "Loses"])
            for level in range(7, 10):
                writer.writerow([level, wins[level - 7], loses[level - 7]])
                
if __name__ == "__main__":
    P = Powertest(save_replay=True)
    #P.match(lvl=9) for cpu eval
    #P.match() # for multi eval
    win = 0
    lose = 0
    tie = 0
    for i in range(1):
        res = P.match()
        if res == 1:
            win += 1
        elif res == -1:
            lose += 1
        else:
            tie += 1
    print("win: ", win, "lose: ", lose,"tie: ", tie)
    #P.run_test() # for specific power test
    #P.run_cpu_test() # for all level power test
