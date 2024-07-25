import argparse
import os
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.wrappers import TransformObservation
import psutil
from enum import Enum
from concurrent.futures import ProcessPoolExecutor
import csv
import random
import shutil

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
from melee_env.myenv import *
from melee_env.agents.basic import *
from melee_env.agents.util import (
    ObservationSpace,
    MyActionSpace
)
from ppo_agent import PPOAgent, StackedPPOAgent, PPOGRUAgent
from model import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--iso", default="/home/tgkang/ssbm.iso", type=str, help="Path to your NTSC 1.02/PAL SSBM Melee ISO"
)
args = parser.parse_args()

class ELOAgent:
    def __init__(self, agent_type, model_path=None, ID=None, level=None):
        self.wins = 0
        self.loses = 0
        self.agent_type = agent_type
        self.elo = 1500
        self.id = ID
        self.model_path = model_path
        self.level = level
        self.csv_path = f"./LeagueSelfPlay/agent{self.id}.csv"
    
class League:
    def __init__(self, players, csv_path, exp_dir):
        self.config = {
            "iso_path": args.iso,
            "stage": enums.Stage.FINAL_DESTINATION,
            "players": None,
            "agent_id": 1,
            "n_states": 808,
            "n_actions": 27, # 25
            "save_replay": False,
            "n_stack": None,
            "timesteps": 3000 # total timesteps for training
        }
        self.players = players
        self.csv_path = csv_path
        self.exp_dir = exp_dir
        self.iter = 0
        
        cfg_ppo = PPO_DEFAULT_CONFIG.copy()
        cfg_ppo["rollouts"] = 8192
        cfg_ppo["learning_epochs"] = 10
        cfg_ppo["mini_batches"] = 8
        cfg_ppo["discount_factor"] = 0.99
        cfg_ppo["lambda"] = 0.95
        cfg_ppo["learning_rate"] = 1e-5  # Lower the learning rate
        cfg_ppo["grad_norm_clip"] = 1.0
        cfg_ppo["ratio_clip"] = 0.2
        cfg_ppo["value_clip"] = 0.2
        cfg_ppo["clip_predicted_values"] = False
        cfg_ppo["entropy_loss_scale"] = 0.02  # Increase entropy loss scale for more exploration
        cfg_ppo["value_loss_scale"] = 0.5
        cfg_ppo["kl_threshold"] = 0
        self.cfg = cfg_ppo
    
    def match(self, actor, learner):
        if actor.agent_type == AgentType.CPU:
            # learner first, actor last
            self.config["players"] = [MyAgent(enums.Character.DOC), CPU(enums.Character.DOC, actor.level)]
            
            if learner.agent_type == AgentType.STACK:
                self.config["n_stack"] = 16
                env = self.make_env("StackedCPUMeleeEnv", self.config)
            else:
                env = self.make_env("CPUMeleeEnv", self.config)
            
        else:
            self.config["players"] = [MyAgent(enums.Character.DOC), MyAgent(enums.Character.DOC)]
            self.config["actor"] = actor
            if learner.agent_type == AgentType.STACK:
                self.config["n_stack"] = 16
                env = self.make_env("StackedSelfPlayMeleeEnv", self.config)
            else:
                env = self.make_env("SelfPlayMeleeEnv", self.config)
        env = wrap_env(env, wrapper="gymnasium")
        device = env.device
        memory = RandomMemory(memory_size=8192, num_envs=env.num_envs, device=device)
        
        self.cfg["state_preprocessor"] = RunningStandardScaler
        self.cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
        self.cfg["value_preprocessor"] = RunningStandardScaler
        self.cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
        self.cfg["experiment"]["write_interval"] = 8192
        self.cfg["experiment"]["checkpoint_interval"] = self.config["timesteps"]
        
        if learner.agent_type == AgentType.STACK:
            self.config["n_stack"] = 16
            leaner_model = {}
            leaner_model["policy"] = Policy(env.observation_space, env.action_space, device)
            leaner_model["value"] = Value(env.observation_space, env.action_space, device)
            
            learner_agent = StackedPPOAgent(
                    models=leaner_model,
                    memory=memory,
                    cfg=self.cfg,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    stack_size=self.config["n_stack"],
                    device=device, 
                    agent_id=1,
                    players={'p1':learner, 'p2':actor},
                    csv_path=self.csv_path,
                    is_selfplay=True
                )
            
        elif learner.agent_type == AgentType.GRU:
            learner_model = {}
            learner_model["policy"] = GRUPolicy(env.observation_space, env.action_space, device)
            learner_model["value"] = GRUValue(env.observation_space, env.action_space, device)
            
            learner_agent = PPOGRUAgent(
                    models=learner_model,
                    memory=memory,
                    cfg=self.cfg,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    device=device, 
                    agent_id = 1,
                    players={'p1':learner, 'p2':actor},
                    csv_path = self.csv_path,
                    is_selfplay=True
                )
        
        learner_agent.load(learner.model_path)
        learner_agent.experiment_dir = os.path.join(self.exp_dir, f"Agent{learner.id}")
        cfg_trainer = {"timesteps": self.config["timesteps"], "headless": True, "disable_progressbar": False}
        trainer = ParallelTrainer(cfg=cfg_trainer, env=env, agents=learner_agent)
        trainer.train()
        
        model_idx = self.config["timesteps"]
        new_model = os.path.join(learner_agent.experiment_dir,"checkpoints",f"agent_{model_idx}.pt")
        if os.path.exists(new_model):
            shutil.copy2(new_model, learner.model_path)
            os.remove(new_model)
            
    def make_env(self, id, config):
        register(
            id=id,
            entry_point=f'melee_env.myenv:{id}',
            kwargs={'config': config},
        )
        return gym.make(id)
    
    def pick_opp(self, learner):
        probs = torch.tensor([1 / (1 + 10 ** ((learner.elo - player.elo) / 4)) for player in self.players])
        probs[learner.id] = float('-inf')
        softmax = torch.nn.Softmax(dim=0)
        prob = softmax(probs)
        actor_id = torch.multinomial(prob, 1).item()
        return actor_id
    
    def change(self):
        win_rates = [x.wins / (x.wins + x.loses + 0.01) for x in self.players[:4]]
        if min(win_rates) <= 0.45:
            print("::POOL CHANGED::")
            worst_id = win_rates.index(min(win_rates))
            copy_id = random.choice([x for x in range(0, 4) if x != worst_id])
            shutil.copy2(self.players[copy_id].model_path, self.players[worst_id].model_path)
            
            with open(self.csv_path, 'r', newline='') as file:
                reader = csv.reader(file)
                rows = list(reader)
            rows[worst_id + 1] = [worst_id, 0, 0, self.players[worst_id].elo]
            with open(self.csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(rows)
            
    
    def update_info(self):
        with open(self.csv_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_id = int(row["ID"])
                self.players[player_id].elo = float(row["elo"])
                self.players[player_id].wins = int(row["wins"])
                self.players[player_id].loses = int(row["loses"])
                
        for player in self.players:
            data = [self.iter, player.elo]
            with open(player.csv_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data)
        
    def clear(self):
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
        # TODO : need to modify this part
        for _ in range(10000):
            self.clear()
            futures = []
            for learner_id in range(4):
                actor_id = self.pick_opp(self.players[learner_id])
                futures.append((self.match, self.players[actor_id], self.players[learner_id]))
            with ProcessPoolExecutor(max_workers=32) as executor:
                futures = [executor.submit(*x) for x in futures]
            self.update_info()
            self.change()
            self.iter += 1

if __name__ == "__main__":
    pre_trained_model = "/home/tgkang/Melee-PPO/skrl-ssbm/trained_model/CPUGRUDOC.pt"
    exp_dir = "./LeagueSelfPlay"
    agent_dirs = [f"Agent{i}" for i in range(5)]
    os.makedirs(exp_dir, exist_ok=True)
    for agent_dir in agent_dirs:
        path = os.path.join(exp_dir, agent_dir)
        os.makedirs(path, exist_ok=True)
    players = [
                ELOAgent(agent_type=AgentType.GRU, ID=0, model_path=os.path.join(exp_dir, "Agent0","recent.pt")),
                ELOAgent(agent_type=AgentType.GRU, ID=1, model_path=os.path.join(exp_dir, "Agent1","recent.pt")),
                ELOAgent(agent_type=AgentType.GRU, ID=2, model_path=os.path.join(exp_dir, "Agent2","recent.pt")),
                ELOAgent(agent_type=AgentType.GRU, ID=3, model_path=os.path.join(exp_dir, "Agent3", "recent.pt")),
                ELOAgent(agent_type=AgentType.CPU, ID=4, level=3), 
                ELOAgent(agent_type=AgentType.CPU, ID=5, level=5),
                ELOAgent(agent_type=AgentType.CPU, ID=6, level=7),
                ]
    # players = [
    #             ELOAgent(agent_type=AgentType.STACK, ID=0, model_path=os.path.join(exp_dir, "Agent0","recent.pt")),
    #             ELOAgent(agent_type=AgentType.STACK, ID=1, model_path=os.path.join(exp_dir, "Agent1","recent.pt")),
    #             ELOAgent(agent_type=AgentType.STACK, ID=2, model_path=os.path.join(exp_dir, "Agent2","recent.pt")),
    #             ELOAgent(agent_type=AgentType.STACK, ID=3, model_path=os.path.join(exp_dir, "Agent3", "recent.pt")),
    #             ELOAgent(agent_type=AgentType.STACK, ID=4, model_path=os.path.join(exp_dir, "Agent4", "recent.pt")),
    #             ELOAgent(agent_type=AgentType.CPU, ID=5, level=5), 
    #             ELOAgent(agent_type=AgentType.CPU, ID=6, level=7),
    #             ELOAgent(agent_type=AgentType.CPU, ID=7, level=9),
    #             ]
    for i in range(4):
        shutil.copy2(pre_trained_model, players[i].model_path)
    csv_path = os.path.join(exp_dir, "playerinfo.csv")
    shutil.copy2("./playerinfo_init.csv", csv_path)

    league = League(players,csv_path, exp_dir)
    league.clear()
    league.run()
    #league.match(league.players[0], league.players[1])
    