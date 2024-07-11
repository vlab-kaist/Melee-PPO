import gymnasium as gym
import os, sys
import signal
import argparse
from distutils.util import strtobool
import psutil

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed, get_latest_run_id, configure_logger
#from stable_baselines3.common.env_checker import check_env
from gymnasium.envs.registration import register
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

import melee
from melee import enums
from melee_env.myenv import MeleeEnv, MyMeleeEnv
from melee_env.agents.basic import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.policies import ActorCriticPolicy
import torch as th

sys.setrecursionlimit(100000)

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description="Run Melee RL training script.")
parser.add_argument("--continue_train", type=str, default="False", help="whether to continue training with existing models")
parser.add_argument("--save_dir", type=str, default="./models/", help="Directory which model saved")
parser.add_argument("--log_dir", type=str, default="./result_logs/", help="Directory which logs are saved")
parser.add_argument("--log_name", type=str, default="PPO", help="Name of experiment")
parser.add_argument("--iso", type=str, default="/home/tgkang/ssbm.iso", help="Path to your NTSC 1.02/PAL SSBM Melee ISO")
parser.add_argument("--cpu_diff", type=int, default=9, help="Difficulty of CPU agent (1-9)")
parser.add_argument("--total_timesteps", type=int, default=int(1e8), help="Number of maximum timesteps to train")
parser.add_argument("--total_episodes", type=int, default=10, help="Number of maximum episodes to train")

parser.add_argument("--n_stack", type=int, default=10, help="number of observations to stack")
parser.add_argument("--n_envs", type=int, default=1, help="number of envs for multiprocessing")
parser.add_argument('--n_steps', type=int, default=8192, help="num of steps are stored in buffer")
parser.add_argument('--batch_size', type=int, default=1024)

args = parser.parse_args()

players = [MyAgent(enums.Character.CPTFALCON), CPU(enums.Character.FOX, args.cpu_diff)]

# register the environment
register(
    id='MyMeleeEnv',
    entry_point='melee_env.myenv:MyMeleeEnv',
    kwargs={'config': {
        "iso_path": args.iso,
        "players": players,
        "agent_id": 1, # for 1p,
        "n_states": 37,
        "n_actions": 45, # need to apply my actionspace
    }},
)

class EpisodicTrainingCallback(BaseCallback):
    def __init__(self, total_episodes, verbose=0):
        super(EpisodicTrainingCallback, self).__init__(verbose)
        self.total_episodes = total_episodes
        self.current_episodes = 0

    def _on_step(self) -> bool:
        done_array = self.locals.get("dones")
        if done_array is not None and any(done_array):
            self.current_episodes += sum(done_array)
        
        if self.current_episodes >= self.total_episodes:
            return False
        return True

callback = EpisodicTrainingCallback(total_episodes=args.total_episodes)

if __name__ == "__main__":
    vec_env = make_vec_env("MyMeleeEnv", n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    vec_env = VecFrameStack(vec_env, n_stack=args.n_stack)
    
    if strtobool(args.continue_train):
        model_path = os.path.join(args.save_dir,"last_saved_model")
        model = PPO.load(model_path)
        model.set_env(vec_env)
    else:
        policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
        model = PPO(
            "MlpPolicy", 
            vec_env,  
            policy_kwargs=policy_kwargs,
            n_steps=args.n_steps, 
            batch_size=args.batch_size,
            verbose=1,
        )
    logger = configure(os.path.join(args.log_dir, args.log_name), ["stdout", "tensorboard"])    
    model.set_logger(logger)
    model.learn(total_timesteps=args.total_timesteps, progress_bar=True, callback=callback, reset_num_timesteps=False)
    model.save(os.path.join(args.save_dir,f"{args.log_name}_{model.num_timesteps}"))
    model.save(os.path.join(args.save_dir,"last_saved_model"))
    vec_env.close()
