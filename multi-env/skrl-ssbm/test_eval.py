import gymnasium as gym
from gymnasium.envs.registration import register
import torch.nn as nn
import torch.nn.functional as F

# Import the skrl components to build the RL system
from skrl.models.torch import Model, CategoricalMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.envs.torch import wrap_env

import torch
import torch.nn as nn
import melee
from melee import enums
from melee_env.myenv import MeleeEnv, MyMeleeEnv
from melee_env.agents.basic import *

class Policy(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)

        self.linear_layer_1 = nn.Linear(self.num_observations, 64)
        self.linear_layer_2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, self.num_actions)

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(inputs["states"]))
        x = F.relu(self.linear_layer_2(x))
        return self.output_layer(x), {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

iso_path = "/home/coder/ssbm.iso"
players = [MyAgent(enums.Character.CPTFALCON), CPU(enums.Character.FOX, 9)]

# register the environment
register(
    id='MyMeleeEnv',
    entry_point='melee_env.myenv:MyMeleeEnv',
    kwargs={'config': {
        "iso_path": iso_path,
        "players": players,
        "agent_id": 1, # for 1p,
        "n_states": 37,
        "n_actions": 45, # need to apply my actionspace
    }},
)

env = gym.vector.make("MyMeleeEnv", num_envs=1, asynchronous=False)
env = wrap_env(env, wrapper='gymnasium')

device = env.device

models_ppo = {}
models_ppo["policy"] = Policy(env.observation_space, env.action_space, device)
models_ppo["value"] = Value(env.observation_space, env.action_space, device)

cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["rollouts"] = 1024  # memory_size
cfg_ppo["learning_epochs"] = 10
cfg_ppo["mini_batches"] = 32
cfg_ppo["discount_factor"] = 0.9
cfg_ppo["lambda"] = 0.95
cfg_ppo["learning_rate"] = 1e-3
cfg_ppo["learning_rate_scheduler"] = KLAdaptiveRL
cfg_ppo["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg_ppo["grad_norm_clip"] = 0.5
cfg_ppo["ratio_clip"] = 0.2
cfg_ppo["value_clip"] = 0.2
cfg_ppo["clip_predicted_values"] = False
cfg_ppo["entropy_loss_scale"] = 0.0
cfg_ppo["value_loss_scale"] = 0.5
cfg_ppo["kl_threshold"] = 0
cfg_ppo["state_preprocessor"] = RunningStandardScaler
cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg_ppo["value_preprocessor"] = RunningStandardScaler
cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}
cfg_ppo["experiment"]["write_interval"] = 1000
cfg_ppo["experiment"]["checkpoint_interval"] = 5000
cfg_ppo["experiment"]["directory"] = "SSBM-test"

agent_ppo = PPO(models=models_ppo,
                memory=None,
                cfg=cfg_ppo,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)

agent_ppo.load("/home/coder/Stable-Baseline3-SSBM/skrl-ssbm/SSBM-test/24-07-09_17-12-47-746301_PPO/checkpoints/agent_5000.pt")

state, info = env.reset()
done = False
while not done:
    action, log_prob, output = agent_ppo.policy.act({"states": agent_ppo._state_preprocessor(state)}, role="policy")
    print(action)
    next_state, reward, done, truncated, info = env.step(action[0])
    state = next_state
    