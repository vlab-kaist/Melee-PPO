import os
import sys
import time
import psutil
import subprocess
import shutil
from tqdm import tqdm

script_path = "./cpu_train.py"
iso = "/home/tgkang/ssbm.iso"
save_dir = "./TransformerGRU"
init_timestep = 0
timesteps = 24600
save_freq = timesteps
model_path = None

real_freq = timesteps * 20

recent_model = os.path.join(save_dir, "checkpoints", "recent_model.pt")

def run_command(cmd):
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            print(f"error occur: {stderr}. kill the program", file=sys.stderr)
            proc.kill()
            raise subprocess.CalledProcessError(proc.returncode, cmd, output=stdout, stderr=stderr)
        print(stderr, file=sys.stderr)
        return stdout
    except subprocess.CalledProcessError as e:
        print(f"error occur: {e.stderr}. kill the program")
        proc.kill()
    except Exception as e:
        print(f"unexpected error occur: {str(e)}. kill the program.")
        proc.kill()
        
# first try
cmd = (
        f"python {script_path} "
        f"--iso {iso} "
        f"--save_dir {save_dir} "
        f"--init_timestep {init_timestep} "
        f"--timesteps {timesteps} "
        f"--save_freq {save_freq} "
    )

run_command(cmd)
new_model = os.path.join(save_dir,"checkpoints",f"agent_{save_freq}.pt")
shutil.copy2(new_model, recent_model)

if os.path.exists(new_model):
    shutil.copy2(new_model, recent_model)
    os.remove(new_model)

for i in tqdm(range(1, 10000), ncols=50):
    init_timestep = i * timesteps + 1

    cmd = (
        f"python {script_path} "
        f"--iso {iso} "
        f"--save_dir {save_dir} "
        f"--init_timestep {init_timestep} "
        f"--timesteps {timesteps} "
        f"--save_freq {save_freq} "
        f"--model_path {recent_model} "
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