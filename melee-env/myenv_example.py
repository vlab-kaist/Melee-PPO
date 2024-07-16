from melee import enums
from melee_env.myenv import MeleeEnv
from melee_env.agents.basic import *
import argparse
from melee_env.agents.util import *

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import tqdm

parser = argparse.ArgumentParser(description="Example melee-env demonstration.")
parser.add_argument("--iso", default='/home/vlab/ssbm.iso', type=str, 
    help="Path to your NTSC 1.02/PAL SSBM Melee ISO")

args = parser.parse_args()

players = [MyAgent(enums.Character.FOX), CPU(enums.Character.FOX, 1)] #CPU(enums.Character.KIRBY, 5)]

env = MeleeEnv(args.iso, players, fast_forward=True, ai_starts_game=True)

episodes = 10; reward = 0
env.start()

for episode in tqdm.tqdm(range(episodes)):
    # gamestate, done = env.setup(enums.Stage.BATTLEFIELD)
    obs, done = env.reset(enums.Stage.FINAL_DESTINATION)
    # env.close()
    while not done:
        a1 = players[0].act(obs)
        a2 = players[1].act(obs)
        obs, reward, done, info = env.step(a1, a2) #if agent is p2 env.step(0, action)
        if done:
            print(info)
        