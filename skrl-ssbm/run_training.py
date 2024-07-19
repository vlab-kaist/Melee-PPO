import os
import sys
import time
import psutil
from tqdm import tqdm

script_path = "./cpu_train.py"
iso = "/home/tgkang/ssbm.iso"
save_dir = "./TransformerGRU"
init_timestep = 0
timesteps = 24600
save_freq = 8200
model_path = None
recent_model = None

# first try
cmd = (
        f"python {script_path} "
        f"--iso {iso} "
        f"--save_dir {save_dir} "
        f"--init_timestep {init_timestep} "
        f"--timesteps {timesteps} "
        f"--save_freq {save_freq} "
    )
os.system(cmd)

for i in tqdm(range(1, 10000)):
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

    os.system(cmd)
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