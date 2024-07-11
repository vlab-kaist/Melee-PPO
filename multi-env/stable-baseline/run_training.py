import os
import sys
import argparse

script_path = "./train_ppo.py"

continue_train="False"
log_name="PPO-4"
save_dir="./models/"
log_dir="./result_logs/"
iso="/root/ssbm.iso"
cpu_diff=9
total_timesteps=int(1e8)
total_episodes=3
n_envs=4

cmd = (
    f"python {script_path} "
    f"--continue_train {continue_train} "
    f"--save_dir {save_dir} "
    f"--log_dir {log_dir} "
    f"--log_name {log_name} "
    f"--iso {iso} "
    f"--cpu_diff {cpu_diff} "
    f"--total_timesteps {total_timesteps} "
    f"--total_episodes {total_episodes} "
    f"--n_envs {n_envs}"
)

os.system(cmd)

continue_train="True"
cmd = (
    f"python {script_path} "
    f"--continue_train {continue_train} "
    f"--save_dir {save_dir} "
    f"--log_dir {log_dir} "
    f"--log_name {log_name} "
    f"--iso {iso} "
    f"--cpu_diff {cpu_diff} "
    f"--total_timesteps {total_timesteps} "
    f"--total_episodes {total_episodes} "
    f"--n_envs {n_envs}"
)

for i in range(100): # 5000 iter
    os.system(cmd)