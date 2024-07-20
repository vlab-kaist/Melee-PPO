import os
import sys
import time
import psutil
import subprocess
from tqdm import tqdm

script_path = "./cpu_train.py"
iso = "/home/tgkang/ssbm.iso"
save_dir = "./TransformerGRU"
init_timestep = 0
timesteps = 24600
save_freq = 8200
model_path = None
recent_model = None

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

for i in tqdm(range(1, 10000), dynamic_ncols=True):
    init_timestep = i * timesteps + 1
    model_idx = init_timestep - 1
    model_path = os.path.join(save_dir,"checkpoints",f"agent_{model_idx}.pt")
    
    if not os.path.exists(model_path):
        model_path = recent_model
    recent_model = model_path
    cmd = (
        f"python {script_path} "
        f"--iso {iso} "
        f"--save_dir {save_dir} "
        f"--init_timestep {init_timestep} "
        f"--timesteps {timesteps} "
        f"--save_freq {save_freq} "
        f"--model_path {model_path} "
    )

    run_command(cmd)
    
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