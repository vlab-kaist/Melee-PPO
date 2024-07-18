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
from ppo_agent import PPOAgent, PPOGRUAgent
from model import Policy, Value, GRUPolicy, GRUValue

parser = argparse.ArgumentParser()
parser.add_argument(
    "--iso", default="/home/tgkang/ssbm.iso", type=str, help="Path to your NTSC 1.02/PAL SSBM Melee ISO"
)

args = parser.parse_args()


def make_env(id, cpu_lvl):
    players = [MyAgent(enums.Character.DOC), CPU(enums.Character.MARIO, cpu_lvl)]
    register(
        id=id,
        entry_point=f'melee_env.myenv:{id}',
        kwargs={'config': {
            "iso_path": args.iso,
            "players": players,
            "agent_id": 1, # for 1p,
            "n_states": 808,
            "n_actions": 27, # 25
            "save_replay": True
        }},
    )
    return gym.make(id)
    
env = make_env(id="MultiMeleeEnv", cpu_lvl=5)
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

models_ppo = {}
models_ppo["policy"] = GRUPolicy(env.observation_space, env.action_space, device, num_envs=1,
                                 hidden_size=512, sequence_length=64)
models_ppo["value"] = GRUValue(env.observation_space, env.action_space, device, num_envs=1,
                               hidden_size=512, sequence_length=64)

cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["state_preprocessor"] = RunningStandardScaler
cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg_ppo["value_preprocessor"] = RunningStandardScaler
cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}

agent_ppo = PPOGRUAgent(models=models_ppo,
                cfg=cfg_ppo,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device, 
                agent_id = 1)

#model_path = "/home/tgkang/multi-env/skrl-ssbm/NewActionSpace/24-07-13_03-31-26-444904_PPOAgent/checkpoints/agent_15564800.pt"
agent_model_path = "/home/tgkang/multi-env/skrl-ssbm/GRU_PPO/24-07-18_22-54-12-871436_PPOGRUAgent/checkpoints/agent_900000.pt"
agent_ppo.load(agent_model_path)
agent_ppo.set_running_mode("eval")
agent_ppo.init()

state, info = env.reset()
done = False
while not done:
    with torch.no_grad():
        action, _ = agent_ppo.act(state, 1, 0)
    next_state, reward, done, truncated, info = env.step((action, 0))
    state = next_state
    