import argparse
import os
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.wrappers import TransformObservation
import psutil
import shutil
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import random
import subprocess
from collections import deque

parser = argparse.ArgumentParser()
parser.add_argument(
    "--iso", default="/home/tgkang/ssbm.iso", type=str, help="Path to your NTSC 1.02/PAL SSBM Melee ISO"
)
args = parser.parse_args()
    
class AgainstCPU:
    def __init__(self, exp_name, char, stage="BATTLEFIELD"):
        self.timesteps = 18000
        self.save_freq = self.timesteps * 50
        self.char = char
        self.stage = stage
        self.script_path = "./cpu_train.py"
        
        self.exp_name = exp_name
        self.save_dir = os.path.abspath(os.path.join('.', exp_name, "checkpoints"))
        self.recent_model = os.path.join(self.save_dir, "recent_model.pt")
        self.init_timestep = 0
        
    def run(self, first=True):
        if first:
            cmd = (
                f"python {self.script_path} "
                f"--iso {args.iso} "
                f"--save_dir {self.exp_name} "
                f"--init_timestep {self.init_timestep} "
                f"--timesteps {self.timesteps} "
                f"--save_freq {self.timesteps} "
                f"--character {self.char} "
                f"--stage {self.stage} "
            )
        else:
            cmd = (
                f"python {self.script_path} "
                f"--iso {args.iso} "
                f"--save_dir {self.exp_name} "
                f"--init_timestep {self.init_timestep} "
                f"--timesteps {self.timesteps} "
                f"--save_freq {self.timesteps} "
                f"--character {self.char} "
                f"--model_path {self.recent_model} "
                f"--stage {self.stage} "
            )
        self.run_command(cmd)
        
    def run_command(self, cmd):
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate(timeout=60 * 5)
            return_code = process.returncode

            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd, output=stdout, stderr=stderr)
        except subprocess.CalledProcessError as e:
            print('The command failed with exit code', e.returncode)
            print('Error output:', e.stderr)
        except subprocess.TimeoutExpired as e:
            print('The command timed out and was terminated.')
            process.terminate()
            process.kill()
        except Exception as e:
            print('An unexpected error occurred:', e)
        finally:
            print('Command finished.')
        
def kill_dolphin():
    # clear dolphin
    current_user = os.getlogin()
    for proc in psutil.process_iter(['pid', 'username', 'name']):
        try:
            if proc.info['username'] == current_user and proc.name() == "dolphin-emu":
                parent_pid = proc.pid
                parent = psutil.Process(parent_pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

if __name__ == "__main__":
    MAX_NUMS = 10000
    trainers = {}
    stage = "BATTLEFIELD"
    chars = ["MARIO", "YOSHI", "LUIGI"] # "PIKACHU", "LINK"]
    for char in chars:    
        trainers[char] = AgainstCPU(exp_name=f"./AgainstCPU/{char}", char=char, stage=stage)
    
    # for the first time
    kill_dolphin()
    futures = []
    for char in chars:
        futures.append((trainers[char].run, True))
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(*x) for x in futures]
    kill_dolphin()
    
    # after first time
    for i in range(1, MAX_NUMS):
        print("Iter: ", i)
        kill_dolphin()
        futures = []
        for char in chars:
            futures.append(trainers[char].run)
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(x) for x in futures]
        kill_dolphin()
        for char in chars:
            s = trainers[char]
            s.init_timestep += s.timesteps
            exp_dir = os.path.join('.', s.exp_name)
            new_model = os.path.join(exp_dir,"checkpoints",f"agent_{s.init_timestep}.pt")
            if os.path.exists(new_model):
                shutil.copy2(new_model, s.recent_model)
                if not s.init_timestep % s.save_freq == 0:
                    os.remove(new_model)
        kill_dolphin()