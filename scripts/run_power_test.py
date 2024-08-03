import glob
import subprocess

script_path = "./power_test.py"
opp_characters = ["DOC", "LINK", "LUIGI", "MARIO", "PIKACHU", "YOSHI"]
characters = ["DOC", "MARIO"]
iso = "/home/tgkang/ssbm.iso"
for char in characters:
    for opp_char in opp_characters:
        model_path = '/home/tgkang/saved_model/against_cpu_FD' + f'/{char.lower()}_best.pt'
        cmd = (
            f"python {script_path} "
            f"--iso {iso} "
            f"--model_path {model_path} "
            f"--char {char} "
            f"--opp_char {opp_char}"
        )
        subprocess.run(cmd, shell=True)
        model_path = '/home/tgkang/saved_model/against_cpu_FD' + f'/{char.lower()}_recent.pt'
        cmd = (
            f"python {script_path} "
            f"--iso {iso} "
            f"--model_path {model_path} "
            f"--char {char} "
            f"--opp_char {opp_char}"
        )
        subprocess.run(cmd, shell=True)
        
