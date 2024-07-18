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
from ppo_agent import PPOAgent
from model import Policy, Value

parser = argparse.ArgumentParser()
parser.add_argument(
    "--iso", default="/home/tgkang/ssbm.iso", type=str, help="Path to your NTSC 1.02/PAL SSBM Melee ISO"
)

args = parser.parse_args()


def make_env(id):
    players = [MyAgent(enums.Character.YOSHI), MyAgent(enums.Character.YOSHI)]
    register(
        id=id,
        entry_point=f'melee_env.myenv:{id}',
        kwargs={'config': {
            "iso_path": args.iso,
            "players": players,
            "agent_id": 1, # for 1p,
            "n_states": 808,
            "n_actions": 25,
            "save_replay": True
        }},
    )
    return gym.make(id)
    
env = make_env(id="MultiMeleeEnv")
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

models_ppo = {}
models_ppo["policy"] = Policy(env.observation_space, env.action_space, device)
models_ppo["value"] = Value(env.observation_space, env.action_space, device)

cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["state_preprocessor"] = RunningStandardScaler
cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg_ppo["value_preprocessor"] = RunningStandardScaler
cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}

agent_ppo = PPOAgent(models=models_ppo,
                cfg=cfg_ppo,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device, 
                agent_id = 1)

op_ppo = PPOAgent(models=models_ppo,
                cfg=cfg_ppo,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device, 
                agent_id = 2)

#model_path = "/home/tgkang/multi-env/skrl-ssbm/NewActionSpace/24-07-13_03-31-26-444904_PPOAgent/checkpoints/agent_15564800.pt"
agent_model_path = "/home/tgkang/multi-env/skrl-ssbm/SelfPlayTest/24-07-17_06-12-55-698471_PPOAgent/checkpoints/agent_18432000.pt"
agent_ppo.load(agent_model_path)
agent_ppo.set_running_mode("eval")

op_model_path = "/home/tgkang/multi-env/skrl-ssbm/NewActionSpace/24-07-16_16-11-23-221821_PPOAgent/checkpoints/agent_1228800.pt"
op_ppo.load(op_model_path)
op_ppo.set_running_mode("eval")

state, info = env.reset()
done = False
while not done:
    with torch.no_grad():
        action, _ = agent_ppo.act(state, 1, 0)
        op_action, _ = op_ppo.act(state, 1, 0)
    next_state, reward, done, truncated, info = env.step((action, op_action))
    state = next_state
    