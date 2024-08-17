import argparse
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.wrappers import TransformObservation
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# Import the skrl components to build the RL system
from skrl.models.torch import Model, CategoricalMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_RNN, PPO_DEFAULT_CONFIG
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
from basics.model import Policy, Value, GRUPolicy, GRUValue

parser = argparse.ArgumentParser()
parser.add_argument(
    "--iso", default="/home/tgkang/ssbm.iso", type=str, help="Path to your NTSC 1.02/PAL SSBM Melee ISO"
)
parser.add_argument(
    "--exp_name", default=None, type=str, help="Name of experiment"
)
parser.add_argument(
    "--init_timestep", default=0, type=int, help="Timestep to start logging and save"
)
parser.add_argument(
    "--timesteps", default=18000, type=int, help="Total timesteps to train"
)
parser.add_argument(
    "--model_path", default=None, type=str, help="Path to the saved model to be loaded for further training"
)
parser.add_argument(
    "--op1_model_path", default=None, type=str, help="opponent model path"
)
parser.add_argument(
    "--op2_model_path", default=None, type=str, help="opponent model path"
)
parser.add_argument(
    "--char", default=None, type=str, help="Character to train"
)
parser.add_argument(
    "--op1_char", default=None, type=str, help="Opponent character"
)
parser.add_argument(
    "--op2_char", default=None, type=str, help="Opponent character"
)
parser.add_argument(
    "--stage", default="FINAL_DESTINATION", type=str, help="stages to play"
)

args = parser.parse_args()

iso_path = args.iso

def make_selfplay_env(op_char, op_model_path):
    players = [MyAgent(getattr(enums.Character, args.char)), 
               MyAgent(getattr(enums.Character, op_char))]
    register(
        id="SelfPlayMeleeEnv",
        entry_point='basics.env:SelfPlayMeleeEnv',
        kwargs={'config': {
            "iso_path": iso_path,
            "players": players,
            "actor_model": op_model_path,
            "agent_id": 1, # for 1p,
            "n_states": 864 if args.stage == "FINAL_DESTINATION" else 880,
            "n_actions": 36,
            "save_replay": False,
            "stage": getattr(enums.Stage, args.stage),
        }},
    )
    return gym.make("SelfPlayMeleeEnv")

def make_cpu_env(cpu_lvl):
    candidates = ["DOC", "MARIO", "YOSHI", "LUIGI", "PIKACHU", "LINK"]
    players = [MyAgent(getattr(enums.Character, args.char)), 
               CPU(getattr(enums.Character, random.choice(candidates)), cpu_lvl)]
    register(
        id="CPUMeleeEnv",
        entry_point='basics.env:CPUMeleeEnv',
        kwargs={'config': {
            "iso_path": iso_path,
            "players": players,
            "agent_id": 1, # for 1p,
            "n_states": 864 if args.stage == "FINAL_DESTINATION" else 880,
            "n_actions": 36,
            "save_replay": False,
            "stage": getattr(enums.Stage, args.stage),
        }},
    )
    return gym.make("CPUMeleeEnv")

env = gym.vector.AsyncVectorEnv([
    lambda: make_cpu_env(cpu_lvl=9),
    lambda: make_selfplay_env(args.op1_char, args.op1_model_path),
    lambda: make_selfplay_env(args.op2_char, args.op2_model_path),
])
env = wrap_env(env, wrapper="gymnasium")
device = env.device
# Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=1024, num_envs=env.num_envs, device=device)

models_ppo = {}
models_ppo["policy"] = GRUPolicy(env.observation_space, env.action_space, device, num_envs=env.num_envs,
                                num_layers=4, hidden_size=512, ffn_size=512, sequence_length=64)
models_ppo["value"] = GRUValue(env.observation_space, env.action_space, device, num_envs=env.num_envs,
                                num_layers=4, hidden_size=512, ffn_size=512, sequence_length=64) #Value(env.observation_space, env.action_space, device)

cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["rollouts"] = 1024  # memory_size
cfg_ppo["learning_epochs"] = 8 # 10
cfg_ppo["mini_batches"] = 8
cfg_ppo["discount_factor"] = 0.99
cfg_ppo["lambda"] = 0.95
cfg_ppo["learning_rate"] = 1e-5
# cfg_ppo["learning_rate_scheduler"] = KLAdaptiveRL
# cfg_ppo["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg_ppo["grad_norm_clip"] = 0.5 # 1.0
cfg_ppo["ratio_clip"] = 0.2
cfg_ppo["value_clip"] = 0.2
cfg_ppo["clip_predicted_values"] = False
cfg_ppo["entropy_loss_scale"] = 0.005 # 0.01
cfg_ppo["value_loss_scale"] = 0.5
cfg_ppo["kl_threshold"] = 0
cfg_ppo["value_preprocessor"] = RunningStandardScaler
cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}
cfg_ppo["experiment"]["write_interval"] = 1024
cfg_ppo["experiment"]["checkpoint_interval"] = args.timesteps

agent_ppo = PPOGRUAgent(models=models_ppo,
                memory=memory,
                cfg=cfg_ppo,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device, 
                agent_id=1,
                platform=False if args.stage == "FINAL_DESTINATION" else True
                )

if args.model_path is not None:
    agent_ppo.load(args.model_path)

agent_ppo.experiment_dir = os.path.join('.', args.exp_name)
cfg_trainer = {"timesteps": args.timesteps, "headless": True, "disable_progressbar": True}
trainer = ParallelTrainer(cfg=cfg_trainer, env=env, agents=agent_ppo)
trainer.initial_timestep = args.init_timestep
trainer.timesteps += args.init_timestep

trainer.train() 