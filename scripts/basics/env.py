import socket
import sys
import time
import random
from filelock import Timeout, FileLock
import gymnasium as gym
from gymnasium import spaces

import melee
import numpy as np
from melee import enums
from basics.util import *
from basics.dconfig import DolphinConfig
import psutil
import torch
from enum import Enum

from skrl.models.torch import Model, CategoricalMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer, ParallelTrainer
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.envs.torch import wrap_env

from basics.ppo_agent import PPOGRUAgent
from basics.model import *

def find_available_udp_port(start_port: int = 1024, end_port: int = 65535) -> int:
    x = list(range(start_port, end_port + 1))
    random.shuffle(x)
    for port in x:
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
            save_replays=self.save_replays,
            overclock=0.2
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
        
        return self.observation_space(self.gamestate)
 
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
                obs, reward, done, gamestate = self.observation_space(self.gamestate)
                return obs, gamestate  # game is not done on start
                
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

# oponent agent is CPU
class CPUMeleeEnv(gym.Env):
    def __init__(self, config={}):
        self.env = MeleeEnv(config["iso_path"], config["players"], fast_forward=True, save_replays=config["save_replay"])
        # with FileLock("thisislock.lock"): self.env.start()
        self.run = False
        self.agent_id = config["agent_id"]
        self.action_space = gym.spaces.Discrete(config["n_actions"])
        low = np.array([-10000]*config["n_states"], dtype=np.float32).reshape(-1)
        high = np.array([10000]*config["n_states"], dtype=np.float32).reshape(-1)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.config = config
        self.platform = config["stage"] != getattr(enums.Stage, "FINAL_DESTINATION")
        if self.platform:
            self.env.observation_space = PlatformObservationSpace()
            

    def reset(self, *, seed=None, options=None):
        if self.run:
            with FileLock("thisislock.lock"): self.env.close()
        with FileLock("thisislock.lock"):
            self.env.start()
        self.run = True
        obs, gamestate = self.env.setup(self.config["stage"])
        return obs[self.agent_id - 1], {'gamestate': gamestate}
    
    def step(self, action):
        truncated = False
        obs, reward, done, gamestate = self.env.step(action)
        return obs[self.agent_id - 1], reward[self.agent_id - 1], bool(done), truncated, {'gamestate': gamestate}
    
    def close(self):
        if self.run:
            with FileLock("thisislock.lock"): self.env.close()

# this is only for evaluation of two agents
class MultiMeleeEnv(gym.Env):
    def __init__(self, config={}):
        self.env = MeleeEnv(config["iso_path"], config["players"], fast_forward=True, save_replays= config["save_replay"])
        # with FileLock("thisislock.lock"): self.env.start()
        self.run = False
        self.agent_id = config["agent_id"]
        self.action_space = gym.spaces.Discrete(config["n_actions"])
        low = np.array([-10000]*config["n_states"], dtype=np.float32).reshape(-1)
        high = np.array([10000]*config["n_states"], dtype=np.float32).reshape(-1)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.config = config
        self.platform = config["stage"] != getattr(enums.Stage, "FINAL_DESTINATION")
        if self.platform:
            self.env.observation_space = PlatformObservationSpace()
        
    def reset(self, *, seed=None, options=None):
        if self.run:
            with FileLock("thisislock.lock"): self.env.close()
        with FileLock("thisislock.lock"):
            self.env.start()
        self.run = True
        obs, gamestate = self.env.setup(self.config["stage"])
        return gamestate, {}
    
    def step(self, actions):
        truncated = False
        obs, reward, done, gamestate = self.env.step(actions[0], actions[1])
        return gamestate, reward, bool(done), truncated, {}
    
    def close(self):
        if self.run:
            with FileLock("thisislock.lock"): self.env.close()

# When our agent is training while op agent is not training
class SelfPlayMeleeEnv(gym.Env):
    def __init__(self, config={}):
        self.env = MeleeEnv(config["iso_path"], config["players"], fast_forward=True, save_replays=config["save_replay"])
        # with FileLock("thisislock.lock"): self.env.start()
        self.run = False
        self.agent_id = config["agent_id"]
        self.action_space = gym.spaces.Discrete(config["n_actions"])
        low = np.array([-10000]*config["n_states"], dtype=np.float32).reshape(-1)
        high = np.array([10000]*config["n_states"], dtype=np.float32).reshape(-1)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.config = config
        self.platform = config["stage"] != getattr(enums.Stage, "FINAL_DESTINATION")
        if self.platform:
            self.env.observation_space = PlatformObservationSpace()
        
        models_ppo = {}
        device = torch.device('cpu')
        
        if config["actor"].agent_type == AgentType.STACK:
            low = np.array([-10000]*config["n_states"]*16, dtype=np.float32).reshape(-1)
            high = np.array([10000]*config["n_states"]*16, dtype=np.float32).reshape(-1)
            observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
            
            models_ppo["policy"] = Policy(observation_space, self.action_space, device)
            models_ppo["value"] = Value(observation_space, self.action_space, device)
            
            self.op_agent = StackedPPOAgent(models=models_ppo,
                    observation_space=observation_space,
                    action_space=self.action_space,
                    device=device, 
                    agent_id = 1 if self.agent_id == 2 else 2,
                    stack_size=16,
                    platform=self.platform)
            
        elif config["actor"].agent_type == AgentType.GRU:
            low = np.array([-10000]*config["n_states"], dtype=np.float32).reshape(-1)
            high = np.array([10000]*config["n_states"], dtype=np.float32).reshape(-1)
            observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
            
            models_ppo["policy"] = GRUPolicy(env.observation_space, env.action_space, device, num_envs=1,
                                num_layers=4, hidden_size=512, ffn_size=512, sequence_length=64)
            models_ppo["value"] = GRUValue(env.observation_space, env.action_space, device, num_envs=1,
                                num_layers=4, hidden_size=512, ffn_size=512, sequence_length=64)
            
            self.op_agent = PPOGRUAgent(models=models_ppo,
                    observation_space=observation_space,
                    action_space=self.action_space,
                    device=device, 
                    agent_id = 1 if self.agent_id == 2 else 2,
                    platform=self.platform)
        
        self.op_agent.load(config["actor"].model_path)
        self.op_agent.set_mode("eval")
        self.op_agent.init()
        
        self.gamestate = None

    def reset(self, *, seed=None, options=None):
        if self.run:
            with FileLock("thisislock.lock"): self.env.close()
        with FileLock("thisislock.lock"):
            self.env.start()
        self.run = True
        obs, self.gamestate = self.env.setup(self.config["stage"])
        #self.stacked_obs = np.tile(obs[self.agent_id - 1], (self.stack_size, 1))
        return obs[self.agent_id - 1], {'gamestate': self.gamestate}
    
    def step(self, action):
        truncated = False
        if self.gamestate is not None:
            op_action, _ = self.op_agent.act(self.gamestate, 1, 0)
        else:
            op_action = 0
        obs, reward, done, self.gamestate = self.env.step(action, op_action)
        return obs[self.agent_id - 1], reward[self.agent_id - 1], bool(done), truncated, {'gamestate': self.gamestate}
    
    def close(self):
        if self.run:
            with FileLock("thisislock.lock"): self.env.close()
