import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import SubprocVecEnv

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
        self.action = torch.tensor([[0]], device=self.device)
        self.action_cnt = 100
        
    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        if timestep < self._random_timesteps:
            print(self.policy.random_act({"states": self._state_preprocessor(states)}, role="policy"))
            return self.policy.random_act({"states": self._state_preprocessor(states)}, role="policy")
        # apply same action for 3 frames (Daboy style)
        if self.action_cnt >= 3:   
            # sample stochastic actions
            actions, log_prob, outputs = self.policy.act({"states": self._state_preprocessor(states)}, role="policy")
            self._current_log_prob = log_prob
            self.action = actions
            self.action_cnt = 1
        else:
            self.action_cnt += 1
        return self.action, self._current_log_prob
        
    
class Policy(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)

        self.linear_layer_1 = nn.Linear(self.num_observations, 256)
        self.linear_layer_2 = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, self.num_actions)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(inputs["states"]))
        x = F.relu(self.linear_layer_2(x))
        return self.output_layer(x), {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.net.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


iso_path = "/home/tgkang/ssbm.iso"
players = [MyAgent(enums.Character.YOSHI), CPU(enums.Character.FOX, 5)]

# register the environment
register(
    id='MyMeleeEnv',
    entry_point='melee_env.myenv:MyMeleeEnv',
    kwargs={'config': {
        "iso_path": iso_path,
        "players": players,
        "agent_id": 1, # for 1p,
        "n_states": 808,
        "n_actions": 45, # need to apply my actionspace
    }},
)
env = gym.make("MyMeleeEnv")
#env = gym.vector.make("MyMeleeEnv", num_envs=4) # async
env = wrap_env(env, wrapper='gymnasium')
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
cfg_ppo["experiment"]["directory"] = "PPO_lv5_Yoshi"

agent_ppo = PPOAgent(models=models_ppo,
                memory=memory,
                cfg=cfg_ppo,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 10000000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent_ppo)
trainer.train()
