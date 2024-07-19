import argparse
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
from skrl.agents.torch.ppo import PPO, PPO_RNN, PPO_DEFAULT_CONFIG
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
from ppo_agent import PPOAgent, PPOGRUAgent, StackedPPOAgent
from model import Policy, Value, GRUPolicy

parser = argparse.ArgumentParser()
parser.add_argument(
    "--iso", default="/home/tgkang/ssbm.iso", type=str, help="Path to your NTSC 1.02/PAL SSBM Melee ISO"
)
parser.add_argument(
    "--exp_name", default="STACK_PPO", type=str, help="Experiment Name"
)
parser.add_argument(
    "--model_path", default=None, type=str, help="Path to the saved model to be loaded for further training"
)
parser.add_argument(
    "--n_stack", default=16, type=int, help="Number of observations to stack"
)

args = parser.parse_args()
iso_path = args.iso
stack_size = args.n_stack

def make_env(id, cpu_lvl):
    players = [MyAgent(enums.Character.DOC), CPU(enums.Character.MARIO, cpu_lvl)]
    register(
        id=id,
        entry_point=f'melee_env.myenv:{id}',
        kwargs={'config': {
            "iso_path": iso_path,
            "players": players,
            "agent_id": 1, # for 1p,
            "n_states": 808,
            "n_actions": 27,
            "save_replay": False,
            "n_stack": stack_size
        }},
    )
    return gym.make(id)

id = "StackedCPUMeleeEnv"
env = gym.vector.AsyncVectorEnv([
    #lambda: make_env(id, 1),
    #lambda: make_env(id, 2),
    lambda: make_env(id, 3),
    #lambda: make_env(id, 4),
    lambda: make_env(id, 5),
    #lambda: make_env(id, 6),
    lambda: make_env(id, 7),
    #lambda: make_env(id, 8),
    lambda: make_env(id, 9)
])
env = wrap_env(env, wrapper="gymnasium")
device = env.device
# Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=8192, num_envs=env.num_envs, device=device)

models_ppo = {}
models_ppo["policy"] = Policy(env.observation_space, env.action_space, device)
models_ppo["value"] = Value(env.observation_space, env.action_space, device)

cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["rollouts"] = 8192  # memory_size
cfg_ppo["learning_epochs"] = 10
cfg_ppo["mini_batches"] = 8
cfg_ppo["discount_factor"] = 0.99
cfg_ppo["lambda"] = 0.95
cfg_ppo["learning_rate"] = 1e-5  # Lower the learning rate
# cfg_ppo["learning_rate_scheduler"] = KLAdaptiveRL
# cfg_ppo["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg_ppo["grad_norm_clip"] = 1.0
cfg_ppo["ratio_clip"] = 0.2
cfg_ppo["value_clip"] = 0.2
cfg_ppo["clip_predicted_values"] = False
cfg_ppo["entropy_loss_scale"] = 0.02  # Increase entropy loss scale for more exploration
cfg_ppo["value_loss_scale"] = 0.5
cfg_ppo["kl_threshold"] = 0
cfg_ppo["state_preprocessor"] = RunningStandardScaler
cfg_ppo["state_preprocessor_kwargs"] = {"size":env.observation_space, "device": device}
cfg_ppo["value_preprocessor"] = RunningStandardScaler
cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints 
cfg_ppo["experiment"]["write_interval"] = 8192
cfg_ppo["experiment"]["checkpoint_interval"] = 10000
cfg_ppo["experiment"]["directory"] = args.exp_name

agent_ppo = StackedPPOAgent(models=models_ppo,
                memory=memory,
                cfg=cfg_ppo,
                observation_space=env.observation_space,
                action_space=env.action_space,
                stack_size = stack_size,
                device=device, 
                agent_id = 1)

if args.model_path is not None:
    agent_ppo.load(args.model_path)
cfg_trainer = {"timesteps": 20000000, "headless": True}
trainer = ParallelTrainer(cfg=cfg_trainer, env=env, agents=agent_ppo)
trainer.train() 