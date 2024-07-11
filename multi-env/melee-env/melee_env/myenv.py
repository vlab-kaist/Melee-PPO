import socket
import sys
import time
from filelock import Timeout, FileLock
import gymnasium as gym

import melee
import numpy as np
from melee import enums
from melee_env.agents.util import *
from melee_env.dconfig import DolphinConfig
import psutil


def find_available_udp_port(start_port: int = 1024, end_port: int = 65535) -> int:
    for port in range(start_port, end_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("", port))
                return port
        except OSError:
            continue
    raise OSError("no availiable port")

class MeleeEnv:
    def __init__(self, 
        iso_path,
        players,
        fast_forward=False, 
        blocking_input=True,
        ai_starts_game=True,
        save_replays=False):

        self.d = DolphinConfig()
        self.d.set_ff(fast_forward)

        self.iso_path = iso_path
        self.players = players
        self.save_replays = save_replays

        self.blocking_input = blocking_input
        self.ai_starts_game = ai_starts_game
        self.observation_space = ObservationSpace()
        
        self.controllers = []
        self.gamestate = None
        self.console = None
        self.port = None

    def start(self):
        if sys.platform == "linux":
            dolphin_home_path = str(self.d.slippi_home) + "/"
        elif sys.platform == "win32":
            dolphin_home_path = None

        self.console = melee.Console(
            path=str(self.d.slippi_bin_path),
            dolphin_home_path=dolphin_home_path,
            blocking_input=self.blocking_input,
            tmp_home_directory=True,
            slippi_port=find_available_udp_port(),
            fullscreen=False,
            gfx_backend="Null",
            setup_gecko_codes=True,
            disable_audio=True,
            save_replays=self.save_replays
        )

        # print(self.console.dolphin_home_path)  # add to logging later
        # Configure Dolphin for the correct controller setup, add controllers
        human_detected = False

        for i in range(len(self.players)):
            curr_player = self.players[i]
            if curr_player.agent_type == "HMN":
                self.d.set_controller_type(
                    i + 1, enums.ControllerType.GCN_ADAPTER)
                curr_player.controller = melee.Controller(
                    console=self.console,
                    port=i + 1,
                    type=melee.ControllerType.GCN_ADAPTER,
                )
                curr_player.port = i + 1
                human_detected = True
            elif curr_player.agent_type in ["AI", "CPU"]:
                self.d.set_controller_type(
                    i + 1, enums.ControllerType.GCN_ADAPTER)
                curr_player.controller = melee.Controller(
                    console=self.console, port=i + 1
                )
                self.menu_control_agent = i
                curr_player.port = i + 1
            else:  # no player
                self.d.set_controller_type(
                    i + 1, enums.ControllerType.UNPLUGGED)

            self.controllers.append(curr_player.controller)
        if self.ai_starts_game and not human_detected:
            self.ai_press_start = True

        else:
            self.ai_press_start = (
                # don't let ai press start without the human player joining in.
                False
            )

        if self.ai_starts_game and self.ai_press_start:
            self.players[self.menu_control_agent].press_start = True

        self.console.run(iso_path=self.iso_path)
        self.console.connect()

        [player.controller.connect()
        for player in self.players if player is not None]

        self.gamestate = self.console.step()
    
    def step(self, *actions):
        for i, player in enumerate(self.players):
            if player.agent_type == "CPU":
                continue
            # controller = player.controller
            action = actions[i]
            control = player.action_space(action)
            control(player.controller)

        if self.gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            self.gamestate = self.console.step()
        
        return self.observation_space(self.gamestate)  #return self.observation_space(self.gamestate)
 
    def setup(self, stage):
        # self.observation_space._reset()
        for player in self.players:
            player.defeated = False
            
        while True:
            self.gamestate = self.console.step()
            if self.gamestate.menu_state is melee.Menu.CHARACTER_SELECT:
                for i in range(len(self.players)):
                    if self.players[i].agent_type == "AI":
                        melee.MenuHelper.choose_character(
                            character=self.players[i].character,
                            gamestate=self.gamestate,
                            controller=self.players[i].controller,
                            costume=i,
                            swag=False,
                            start=self.players[i].press_start)
                    if self.players[i].agent_type == "CPU":
                        melee.MenuHelper.choose_character(
                            character=self.players[i].character,
                            gamestate=self.gamestate,
                            controller=self.players[i].controller,
                            costume=i,
                            swag=False,
                            cpu_level=self.players[i].lvl,
                            start=self.players[i].press_start)  
            elif self.gamestate.menu_state is melee.Menu.STAGE_SELECT:
                # time.sleep(0.1)
                melee.MenuHelper.choose_stage(
                    stage=stage,
                    gamestate=self.gamestate,
                    controller=self.players[self.menu_control_agent].controller)

            elif self.gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                obs, reward, done, info = self.observation_space(self.gamestate)
                return obs, False  # game is not done on start
                
            else:
                melee.MenuHelper.choose_versus_mode(self.gamestate, self.players[self.menu_control_agent].controller)


    def close(self):
        try:
            [player.controller.disconnect() for player in self.players if player is not None]
        except Exception as e:
            print(f"Failed to disconnect controller {e}")
        # for t, c in self.controllers.items():
        #     c.disconnect()

        # self.observation_space._reset()
        self.gamestate = None
        self.console.stop()
        # time.sleep(2)

"""Gym Style Environment"""
class MyMeleeEnv(gym.Env):
    def __init__(self, config={}):
        self.env = MeleeEnv(config["iso_path"], config["players"], fast_forward=True, save_replay=False)
        # with FileLock("thisislock.lock"): self.env.start()
        self.run = False
        self.agent_id = config["agent_id"]
        self.action_space = gym.spaces.Discrete(config["n_actions"])
        low = np.array([[-10000]*config["n_states"]], dtype=np.float32).reshape(-1)
        high = np.array([10000]*config["n_states"], dtype=np.float32).reshape(-1)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        
    def reset(self, *, seed=None, options=None):
        if self.run:
            with FileLock("thisislock.lock"): self.env.close()
        with FileLock("thisislock.lock"):
            self.env.start()
        self.run = True
        obs, done = self.env.setup(enums.Stage.FINAL_DESTINATION)
        return obs[self.agent_id - 1], {}
    
    def step(self, action):
        truncated = False
        obs, reward, done, info = self.env.step(action)
        return obs[self.agent_id - 1], reward[self.agent_id - 1], bool(done), truncated, {}
    
    def close(self):
        if self.run:
            with FileLock("thisislock.lock"): self.env.close()
        