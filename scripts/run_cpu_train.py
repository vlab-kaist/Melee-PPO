import os
import sys
import time
import psutil
import subprocess
import shutil
import signal
from tqdm import tqdm

# characters = ["DOC", "MARIO", "YOSHI", "LUIGI", "PIKACHU", "LINK"]
character = "LINK"
script_path = "./cpu_train.py"
iso = "/home/tgkang/ssbm.iso"
save_dir = "./TransformerGRU"
init_timestep = 0
timesteps = 18000
save_freq = timesteps
model_path = None

real_freq = timesteps * 100

recent_model = os.path.join(save_dir, "checkpoints", "recent_model.pt")

def run_command(cmd):
    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(timeout=60 * 6)
        return_code = process.returncode

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd, output=stdout, stderr=stderr)
    except subprocess.CalledProcessError as e:
        print('The command failed with exit code', e.returncode)
        print('Error output:', e.stderr)
    except subprocess.TimeoutExpired as e:
        print('The command timed out and was terminated.')
        process.kill()
        stdout, stderr = process.communicate()
        print('Terminated command output:', stdout)
        print('Terminated command error:', stderr)
    except Exception as e:
        print('An unexpected error occurred:', e)
    finally:
        print('Command finished.')
        
# first try
cmd = (
        f"python {script_path} "
        f"--iso {iso} "
        f"--save_dir {save_dir} "
        f"--init_timestep {init_timestep} "
        f"--timesteps {timesteps} "
        f"--save_freq {save_freq} "
        f"--character {character} "
    )

run_command(cmd)
new_model = os.path.join(save_dir,"checkpoints",f"agent_{save_freq}.pt")
shutil.copy2(new_model, recent_model)

for i in tqdm(range(1, 10001), ncols=50):
    init_timestep = i * timesteps + 1

    cmd = (
        f"python {script_path} "
        f"--iso {iso} "
        f"--save_dir {save_dir} "
        f"--init_timestep {init_timestep} "
        f"--timesteps {timesteps} "
        f"--save_freq {save_freq} "
        f"--model_path {recent_model} "
        f"--character {character} "
    )

    run_command(cmd)
    
    model_idx = (i + 1) * timesteps
    new_model = os.path.join(save_dir,"checkpoints",f"agent_{model_idx}.pt")
    
    if os.path.exists(new_model):
        shutil.copy2(new_model, recent_model)
        if not model_idx % real_freq == 0:
            os.remove(new_model)
            
    time.sleep(0.1)
    
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