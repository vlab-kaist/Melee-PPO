import argparse

from melee import enums
from melee_env.agents.basic import *
from melee_env.env import MeleeEnv
from melee_env.agents.util import *

parser = argparse.ArgumentParser(description="Example melee-env demonstration.")
parser.add_argument(
    "--iso", default='/home/tgkang/ssbm.iso', type=str, help="Path to your NTSC 1.02/PAL SSBM Melee ISO"
)

args = parser.parse_args()

players = [MyAgent(enums.Character.FOX), CPU(enums.Character.FOX, 9)]
env = MeleeEnv(args.iso, players, fast_forward=True)

episodes = 10
reward = 0
env.start()

for episode in range(episodes):
    gamestate, done = env.setup(enums.Stage.BATTLEFIELD)
    while not done:
        for i in range(len(players)):
            players[i].act(gamestate)
        gamestate, done = env.step()
env.close()
