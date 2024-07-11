import gymnasium as gym
from gymnasium.envs.registration import register

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

import torch
import torch.nn as nn
import melee
from melee import enums
from melee_env.myenv import MeleeEnv, MyMeleeEnv
from melee_env.agents.basic import *

class PPOAgent(PPO):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        if timestep < self._random_timesteps:
            return self.policy.random_act({"states": self._state_preprocessor(states)}, role="policy")

        # sample stochastic actions
        actions, log_prob, outputs = self.policy.act({"states": self._state_preprocessor(states)}, role="policy")
        self._current_log_prob = log_prob
        # print(actions)
        return actions, log_prob, outputs
        
    
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


iso_path = "/home/tgkang/ssbm.iso"
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
env = gym.make("MyMeleeEnv")
#env = gym.vector.make("LunarLander-v2", num_envs=1)
env = wrap_env(env, wrapper='gymnasium')
device = env.device


# Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=1024, num_envs=env.num_envs, device=device)

models_ppo = {}
models_ppo["policy"] = Policy(env.observation_space, env.action_space, device)
models_ppo["value"] = Value(env.observation_space, env.action_space, device)

cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["rollouts"] = 1024  # memory_size
cfg_ppo["learning_epochs"] = 10
cfg_ppo["mini_batches"] = 32
cfg_ppo["discount_factor"] = 0.99
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
# logging to TensorBoard and write checkpoints each 500 and 5000 timesteps respectively
cfg_ppo["experiment"]["write_interval"] = 1000
cfg_ppo["experiment"]["checkpoint_interval"] = 5000
cfg_ppo["experiment"]["directory"] = "SSBM-test"

agent_ppo = PPOAgent(models=models_ppo,
                memory=memory,
                cfg=cfg_ppo,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 100000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent_ppo)
#trainer =  ParallelTrainer(cfg=cfg_trainer, env=env, agents=agent_ppo)
# start training
trainer.train()
