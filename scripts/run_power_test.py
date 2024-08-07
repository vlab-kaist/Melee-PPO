import glob
import subprocess

script_path = "power_test.py"
opp_characters = ["DOC", "LINK", "LUIGI", "MARIO", "PIKACHU", "YOSHI"]
characters = ["DOC", "LINK", "LUIGI", "MARIO", "PIKACHU", "YOSHI"]
iso = "/home/tgkang/ssbm.iso"
for char in characters:
    for opp_char in opp_characters:
        model_path = f"/home/tgkang/saved_model/aginst_cpu_FD/{char}_cpu.pt"
        cmd = (
            f"python {script_path} "
            f"--iso {iso} "
            f"--char {char} "
            f"--model_path {model_path} "
            f"--op_char {opp_char} "
        )
        subprocess.run(cmd, shell=True)
