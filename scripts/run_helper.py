import glob
import subprocess

script_path = ["power_test.py", "human_test.py"][0] #"human_test.py"
iso = "/home/tgkang/ssbm.iso"
chars = {0: "DOC", 
         1: "MARIO", 
         2: "YOSHI", 
         3: "LUIGI", 
         4: "PIKACHU", 
         5: "LINK"}
char = chars[2]
op_char = chars[2]
stage = ["BATTLEFIELD", "FINAL_DESTINATION", "POKEMON_STADIUM"][0]
model_path = "/home/tgkang/saved_model/Selfplay_BF_1/YOSHI/checkpoints/agent_12600000.pt"
op_model_path = "/home/tgkang/saved_model/Selfplay_BF_1/YOSHI/checkpoints/best_agent.pt"
cmd = (
    f"python {script_path} "
    f"--iso {iso} "
    f"--char {char} "
    f"--model_path {model_path} "
    f"--stage {stage} "
    f"--op_char {op_char} "
    f"--op_model_path {op_model_path} "
)
subprocess.run(cmd, shell=True)