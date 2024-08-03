import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from skrl.agents.torch.ppo import PPO, PPO_RNN, PPO_DEFAULT_CONFIG

import melee
from melee import enums
from melee.enums import Action, Character, Button
import numpy as np
from enum import Enum       
        
class PPOGRUAgent(PPO_RNN):
    def __init__(self, agent_id=1, players=None, csv_path=None,
                 is_selfplay=False, platform=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_id = agent_id
        self.action = torch.tensor([[0]], device=self.device)
        self.action_cnt = 3
        self.players = players
        self.gamestate = None
        self.csv_path = csv_path
        self.is_selfplay = is_selfplay
        self.platform = platform
        
        self.macro_mode = False
        self.macro_queue = []
        self.macro_idx = 0
        self.side_b = False # for mario
        self.cyclone = False # for mario and luigi
        
    def act(self, states, timestep: int, timesteps: int):
        #if env is not myenv, env gives gamestate itself to an agent
        if not isinstance(states, torch.Tensor):
            self.gamestate = states
            states = self.state_preprocess(states)
            states = torch.tensor(states, device=self.device, dtype=torch.float32).view(1, -1)
        ai = self.gamestate.players[self.agent_id]
        if ai.on_ground and self.action == 11: # cyclone is charged when down b occurs on ground
            self.cyclone = False
        if ai.on_ground or not ai.off_stage:
            self.side_b = False
        if ai.action in [Action.EDGE_HANGING, Action.EDGE_CATCHING]:
            self.macro_mode = False
            self.macro_queue = []
            self.macro_idx = 0
        elif (not self.macro_mode) and ai.off_stage:
            if ai.character in [Character.MARIO, Character.DOC]:
                self.mario_recovery()
            elif ai.character == Character.LINK:
                self.link_recovery()
            elif ai.character == Character.PIKACHU:
                self.pikachu_recovery()
            elif ai.character == Character.YOSHI:
                self.yoshi_recovery()
            else:
                self.luigi_recovery()
        if self.macro_mode:
            self.action = self.macro_queue[self.macro_idx]
            self.macro_idx += 1
            if self.macro_idx >= len(self.macro_queue):
                self.macro_idx = 0
                self.macro_mode = False
                self.macro_queue = []
            return self.action, self._current_log_prob
        
        if self.action_cnt >= 3:   
            rnn = {"rnn": self._rnn_initial_states["policy"]} if self._rnn else {}
            actions, log_prob, outputs = self.policy.act({"states": self._state_preprocessor(states), **rnn}, role="policy")
            self._current_log_prob = log_prob
            self.action = actions
            self.action_cnt = 1
            
            if self._rnn:
                self._rnn_final_states["policy"] = outputs.get("rnn", [])
        else:
            self.action_cnt += 1
        return self.action, self._current_log_prob
        #return 0, self._current_log_prob
    
    def state_preprocess(self, gamestate):
        return state_preprocess(gamestate, self.agent_id, self.platform)
    
    def record_transition(self, states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps):
        super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)
        self.gamestate = infos["gamestate"]
    
    def mario_recovery(self):
        # TODO: check avaiable recoverable distance and use down-B
        self.macro_mode = True
        self.macro_idx = 0
        ai = self.gamestate.players[self.agent_id]
        edge_pos = melee.stages.EDGE_GROUND_POSITION[self.gamestate.stage]
        is_left = ai.position.x < 0
        if ai.position.y > 0: # just move
            self.macro_queue = [2, 2, 2] if is_left else [1, 1, 1]
        else:
            if ai.jumps_left > 0: # jump
                self.macro_queue = [21, 21, 21] if is_left else [20, 20, 20]
            else: 
                if ai.position.y > -10 and abs(ai.position.x) - edge_pos > 0: # just move
                    self.macro_queue = [2, 2, 2] if is_left else [1, 1, 1]
                elif abs(ai.position.x) - edge_pos > 40 and not self.side_b: # 40 => more larger ex.50?
                    self.macro_queue = [10, 10, 10] if is_left else [9, 9, 9] # apply side B
                    self.macro_queue += [2, 2] * 20 if is_left else [1, 1] * 20 # this is important
                    self.side_b = True
                else: # up B
                    self.macro_queue = [14, 14, 14] if is_left else [13, 13, 13]
    
    def link_recovery(self):
        # TODO: ues z (lope) for catching side edge or side / use bomb
        self.macro_mode = True
        self.macro_idx = 0
        ai = self.gamestate.players[self.agent_id]
        edge_pos = melee.stages.EDGE_GROUND_POSITION[self.gamestate.stage]
        is_left = ai.position.x < 0
        if ai.position.y > 0: # just move
            self.macro_queue = [2, 2, 2] if is_left else [1, 1, 1]
        else:
            if ai.jumps_left > 0: # jump
                self.macro_queue = [21, 21, 21] if is_left else [20, 20, 20]
            else: 
                if ai.position.y > -10 and abs(ai.position.x) - edge_pos > 0: # just move
                    self.macro_queue = [2, 2, 2] if is_left else [1, 1, 1]
                else: # up B
                    self.macro_queue = [14, 14, 14] if is_left else [13, 13, 13]
    
    def pikachu_recovery(self):
        # TODO: need to use side B for reducing distance
        self.macro_mode = True
        self.macro_idx = 0
        ai = self.gamestate.players[self.agent_id]
        edge_pos = melee.stages.EDGE_GROUND_POSITION[self.gamestate.stage]
        is_left = ai.position.x < 0
        if ai.position.y > 0: # just move
            self.macro_queue = [2, 2, 2] if is_left else [1, 1, 1]
        else:
            if ai.jumps_left > 0: # jump
                self.macro_queue = [21, 21, 21] if is_left else [20, 20, 20]
            else: 
                if ai.position.y > -50 and abs(ai.position.x) - edge_pos > 0: # just move
                    self.macro_queue = [2, 2, 2] if is_left else [1, 1, 1]
                else: # up B
                    self.macro_queue = [12, 12, 12, 12]
                    self.macro_queue += [0] * 15 #20
                    self.macro_queue += [24, 24, 24] * 2 if is_left else [23, 23, 23] * 2
    
    def yoshi_recovery(self):
        # TODO: need to use side B for reducing distance
        self.macro_mode = True
        self.macro_idx = 0
        ai = self.gamestate.players[self.agent_id]
        edge_pos = melee.stages.EDGE_GROUND_POSITION[self.gamestate.stage]
        is_left = ai.position.x < 0
        if ai.position.y > 20: # just move
            self.macro_queue = [2, 2, 2] if is_left else [1, 1, 1]
        else:
            if ai.jumps_left > 0: # jump
                self.macro_queue = [21, 21, 21] if is_left else [20, 20, 20]
            else: 
                if ai.position.y > -10 and abs(ai.position.x) - edge_pos > 0: # just move
                    self.macro_queue = [2, 2, 2] if is_left else [1, 1, 1]
                else: # L
                    self.macro_queue = [19, 19, 19] # need to change
    
    def luigi_recovery(self):
        # TODO: make more stable when left B fires, and move side using down B
        self.macro_mode = True
        self.macro_idx = 0
        ai = self.gamestate.players[self.agent_id]
        edge_pos = melee.stages.EDGE_GROUND_POSITION[self.gamestate.stage]
        is_left = ai.position.x < 0
        if ai.position.y > 30:
            self.macro_queue = [10, 10, 10, 0] if is_left else [9, 9, 9, 0]
        elif ai.position.y > 0: # just move
            self.macro_queue = [2, 2, 2] if is_left else [1, 1, 1]
        else:
            if ai.jumps_left > 0: # jump
                self.macro_queue = [21, 21, 21] if is_left else [20, 20, 20]
            else: 
                if ai.position.y > -10 and abs(ai.position.x) - edge_pos > 0: # just move
                    self.macro_queue = [2, 2, 2] if is_left else [1, 1, 1]
                elif abs(ai.position.x) - edge_pos > 20 and not self.cyclone: # cyclone if possible
                    self.macro_queue = [11, 0] * 25
                    self.cyclone = True
                elif abs(ai.position.x) - edge_pos < 15: # up B
                    self.macro_queue = [14, 14, 14] if is_left else [13, 13, 13]
                else:
                    self.macro_queue = [2, 2, 2] if is_left else [1, 1, 1]

def state_preprocess(gamestate, agent_id, platform=False):
    proj_mapping = {
            enums.ProjectileType.MARIO_FIREBALL: 0,
            enums.ProjectileType.DR_MARIO_CAPSULE: 1,
            enums.ProjectileType.LINK_BOMB: 2,
            enums.ProjectileType.LINK_HOOKSHOT: 3,
            enums.ProjectileType.LINK_ARROW: 4,
            enums.ProjectileType.PIKACHU_THUNDER: 5,
            enums.ProjectileType.MARIO_CAPE: 6,
            enums.ProjectileType.DR_MARIO_CAPE: 7,
            enums.ProjectileType.YOSHI_EGG_THROWN: 8,
            enums.ProjectileType.YOSHI_TONGUE: 9,
            enums.ProjectileType.YOSHI_STAR: 10,
            enums.ProjectileType.PIKACHU_THUNDERJOLT_1: 11,
            enums.ProjectileType.PIKACHU_THUNDERJOLT_2: 12,
            enums.ProjectileType.LUIGI_FIRE: 13
        }
    
    if agent_id == 1:
        p1 = gamestate.players[1]
        p2 = gamestate.players[2]
    else:
        p1 = gamestate.players[2]
        p2 = gamestate.players[1]
        
    edge_pos = melee.stages.EDGE_GROUND_POSITION[gamestate.stage]
    
    if not platform:
        state = np.zeros((869,), dtype=np.float32)
    else:
        state = np.zeros((885,), dtype=np.float32)
        
    state[0] = p1.position.x / edge_pos
    state[1] = p1.position.y / edge_pos
    state[2] = p2.position.x / edge_pos
    state[3] = p2.position.y / edge_pos
    state[4] = gamestate.distance / 20
    state[5] = (edge_pos - abs(p1.position.x)) / edge_pos
    state[6] = (edge_pos - abs(p2.position.x)) / edge_pos
    state[7] = 1.0 if p1.facing else -1.0
    state[8] = 1.0 if p2.facing else -1.0
    state[9] = 1.0 if (p1.position.x - p2.position.x) * state[7] < 0 else -1.0
    state[10] = p1.hitstun_frames_left / 10
    state[11] = p2.hitstun_frames_left / 10
    state[12] = p1.invulnerability_left / 20
    state[13] = p2.invulnerability_left / 20
    state[14] = p1.jumps_left - 1
    state[15] = p2.jumps_left - 1
    state[16] = p1.off_stage * 1.0
    state[17] = p2.off_stage * 1.0
    state[18] = p1.on_ground * 1.0
    state[19] = p2.on_ground * 1.0
    state[20] = (p1.percent - 50) / 50
    state[21] = (p2.percent - 50) / 50
    state[22] = (p1.shield_strength - 30) / 30
    state[23] = (p2.shield_strength - 30) / 30
    state[24] = p1.speed_air_x_self / 2
    state[25] = p2.speed_air_x_self / 2
    state[26] = p1.speed_ground_x_self / 2
    state[27] = p2.speed_ground_x_self / 2
    state[28] = p1.speed_x_attack
    state[29] = p2.speed_x_attack
    state[30] = p1.speed_y_attack
    state[31] = p2.speed_y_attack
    state[32] = p1.speed_y_self
    state[33] = p2.speed_y_self
    state[34] = (p1.action_frame - 15) / 15
    state[35] = (p2.action_frame - 15) / 15
    
    state[36] = (p1.ecb.top.y - 12) / 2.5
    state[37] = (p1.ecb.bottom.y - 2) / 2
    state[38] = (p1.ecb.left.x - 2.7)
    state[39] = (p1.ecb.left.y - 7) / 2
    state[40] = p1.ecb.right.x + 2.8
    state[41] = (p1.ecb.right.y + 2.8) / 10
    
    if p1.action.value < 386:
        state[41 + p1.action.value] = 1.0
    if p2.action.value < 386:
        state[41 + 386 + p2.action.value] = 1.0
    
    # if the type is same, then apply only once
    projs = [x for x in gamestate.projectiles if x.owner == 2 and x.type in proj_mapping.keys()]
    for i, proj in enumerate(projs):
        state[41 + 386 * 2 + 4 * proj_mapping[proj.type]] = proj.position.x / edge_pos
        state[41 + 386 * 2 + 4 * proj_mapping[proj.type] + 1] = proj.position.y / edge_pos
        state[41 + 386 * 2 + 4 * proj_mapping[proj.type] + 2] = proj.speed.x / 2
        state[41 + 386 * 2 + 4 * proj_mapping[proj.type] + 3] = proj.speed.y / 2
    
    if platform:
        p1_on_left = False
        p2_on_left = False
        p1_on_right = False
        p2_on_right = False
        p1_on_top = False
        p2_on_top = False
        left_height, left_left, left_right = melee.left_platform_position(gamestate)
        right_height, right_left, right_right = melee.right_platform_position(gamestate)
        top_height, top_left, top_right = melee.top_platform_position(gamestate)
        p1_on_left = left_left - 1 < p1.position.x < left_right + 1 \
            and (p1.position.y - left_height) <= 1
        p2_on_left = left_left - 1 < p2.position.x < left_right + 1 \
            and (p2.position.y - left_height) <= 1
        p1_on_right = right_left - 1 < p1.position.x < right_right + 1 \
            and (p1.position.y - right_height) <= 1
        p2_on_right = right_left - 1 < p2.position.x < right_right + 1 \
            and (p2.position.y - right_height) <= 1

        if top_height is not None:
            p1_on_top = top_left - 1 < p1.position.x < top_right + 1 \
                and (p1.position.y - top_height) <= 1
            p2_on_top = top_left - 1 < p2.position.x < top_right + 1 \
                and (p2.position.y - top_height) <= 1

        state[869] = 1.0 if p1_on_left else 0
        state[870] = 1.0 if p2_on_left else 0
        state[871] = 1.0 if p1_on_right else 0
        state[872] = 1.0 if p2_on_right else 0
        state[873] = 1.0 if p1_on_top else 0
        state[874] = 1.0 if p2_on_top else 0
        state[875] = (abs(p1.position.x) - right_left) / edge_pos
        state[876] = (abs(p2.position.x) - right_left) / edge_pos
        state[877] = (abs(p1.position.x) - right_right) / edge_pos
        state[878] = (abs(p2.position.x) - right_right) / edge_pos
        state[879] = (p1.position.y - right_height) / 70
        state[880] = (p2.position.y - right_height) / 70
        state[881] = (abs(p1.position.x) - top_right) / edge_pos if top_height is not None else 0
        state[882] = (abs(p2.position.x) - top_right) / edge_pos if top_height is not None else 0
        state[883] = (p1.position.y - top_height) / top_height if top_height is not None else 0
        state[884] = (p2.position.y - top_height) / top_height if top_height is not None else 0
        
    return state