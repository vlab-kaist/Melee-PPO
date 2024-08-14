import os
import sys
import time
import psutil
import subprocess
import shutil
import signal
from tqdm import tqdm

# characters = ["DOC", "MARIO", "YOSHI", "LUIGI", "PIKACHU", "LINK"]
# stages = ["BATTLEFIELD", "FINAL_DESTINATION", "POKEMON_STADIUM"]
character = "LINK"
stage = "BATTLEFIELD"
script_path = "./cpu_train.py"
iso = "/home/tgkang/ssbm.iso"
save_dir = "./LINK_BF"
init_timestep = 0
timesteps = 18000
save_freq = timesteps
model_path = None

real_freq = timesteps * 50

recent_model = os.path.join(save_dir, "checkpoints", "recent_model.pt")

def kill_dolphin():
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
            
def run_command(cmd):
    try:
        current_user = os.getlogin()  
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
        process.terminate()
        process.kill()
        kill_dolphin()
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
        f"--stage {stage} "
    )

run_command(cmd)
new_model = os.path.join(save_dir,"checkpoints",f"agent_{save_freq}.pt")
shutil.copy2(new_model, recent_model)

kill_dolphin()
for i in tqdm(range(1, 10001), ncols=50):
    kill_dolphin()
    
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
        f"--stage {stage} "
    )

    run_command(cmd)
    
    model_idx = (i + 1) * timesteps
    new_model = os.path.join(save_dir,"checkpoints",f"agent_{model_idx}.pt")
    
    if os.path.exists(new_model):
        shutil.copy2(new_model, recent_model)
        if not model_idx % real_freq == 0:
            os.remove(new_model)
            
    time.sleep(0.1)
    kill_dolphin()