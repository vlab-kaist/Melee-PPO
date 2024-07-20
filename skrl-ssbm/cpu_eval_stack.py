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
from skrl.resources.preprocessors.torch import RunningStandardScaler
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
from model import Policy, Value

parser = argparse.ArgumentParser()
parser.add_argument(
    "--iso", default="/home/tgkang/ssbm.iso", type=str, help="Path to your NTSC 1.02/PAL SSBM Melee ISO"
)

args = parser.parse_args()
config= {
    "iso_path": args.iso,
    "players": [MyAgent(enums.Character.DOC), CPU(enums.Character.MARIO, 9)],
    "agent_id": 1, # for 1p,
    "n_states": 808,
    "n_actions": 27, # 25
    "save_replay": True,
    "n_stack": 4
}

def make_env(id):
    register(
        id=id,
        entry_point=f'melee_env.myenv:{id}',
        kwargs={'config': config},
    )
    return gym.make(id)
    
env = make_env(id="MultiMeleeEnv")
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

low = np.array([-10000]*config["n_states"]*config["n_stack"], dtype=np.float32).reshape(-1)
high = np.array([10000]*config["n_states"]*config["n_stack"], dtype=np.float32).reshape(-1)
observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
action_space = gym.spaces.Discrete(config["n_actions"])

models_ppo = {}
models_ppo["policy"] = Policy(observation_space, action_space, device)
models_ppo["value"] = Value(observation_space, action_space, device)

cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["state_preprocessor"] = RunningStandardScaler
cfg_ppo["state_preprocessor_kwargs"] = {"size": observation_space, "device": device}
cfg_ppo["value_preprocessor"] = RunningStandardScaler
cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}
        
agent_ppo = StackedPPOAgent(models=models_ppo,
                cfg=cfg_ppo,
                observation_space=observation_space,
                action_space=action_space,
                device=device, 
                agent_id=1, 
                stack_size=config["n_stack"])

#model_path = "/home/tgkang/multi-env/skrl-ssbm/NewActionSpace/24-07-13_03-31-26-444904_PPOAgent/checkpoints/agent_15564800.pt"
agent_model_path = "/home/tgkang/Melee-PPO/skrl-ssbm/runs/24-07-20_02-04-26-112082_StackedPPOAgent/checkpoints/agent_10000.pt"
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
    