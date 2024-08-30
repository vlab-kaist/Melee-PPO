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
from heuristics.sdi import SDI  
import random   


class PPOGRUAgent(PPO_RNN):
    def __init__(self, agent_id=1, players=None, csv_path=None,
                 is_selfplay=False, platform=False, delay=3, tou=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_id = agent_id
        self.action = torch.tensor(0).unsqueeze(0)
        self.prev = torch.tensor(0).unsqueeze(0)
        
        self.delay = delay
        self.tou = tou
        self.action_cnt = 3
        self.players = players
        self.gamestate = None
        self.csv_path = csv_path
        self.is_selfplay = is_selfplay
        self.platform = platform
        
        self.macro_mode = False
        self.macro_queue = []
        self.macro_idx = 0

        self.mash_mode = False
        self.mash_queue = [4,8,22,19]
        self.mash_idx = 0

        self.side_b = False # for mario
        self.cyclone = False # for mario and luigi
        self.mash_counter = 0 # for mash action, 
        self.cc_test = None # for cc effectiveness check
        self.sdi = None # for sdi implication
        self.prev_action = None # for action confirming
        self.prev_op_action = None
        self.shield_charging = False
        self.cnt = 0

        
    # Priority : 1. Recovery & SDI (never possible simultaenously due to conditions) 2. Grab Mash 3. L_cancel 4. no_shield  
    def act(self, states, timestep: int, timesteps: int):
        self.cnt += 1
        #if env is not myenv, env gives gamestate itself to an agent
        if not isinstance(states, torch.Tensor):
            self.gamestate = states
            states = self.state_preprocess(states)
            states = torch.tensor(states, device=self.device, dtype=torch.float32).view(1, -1)
        if isinstance(self.gamestate, melee.gamestate.GameState):
            ai = self.gamestate.players[self.agent_id]
            op = self.gamestate.players[3 - self.agent_id]

            #print(op.position.x, op.position.y)
            if ai.action != self.prev_action:
                #print("ai:", ai.action, self.cnt)
                self.prev_action = ai.action
            if op.action != self.prev_op_action:
                #print("op:", op.action, self.cnt)
                self.prev_op_action = op.action
            
            if ai.shield_strength >= 40:
                self.shield_charging = False
            # initial sdi setup
            if ai.hitlag_left <= 1:
                self.sdi = None

            actionable_landing = ai.action == Action.LANDING and ai.action_frame >= 4

            #5 Crouch cancel?
            '''if ai.action in [Action.STANDING] or actionable_landing:  
                if self.cc_test == "occ":
                    self.crouch_cancel(mode = 1)
                elif self.cc_test == "dcc": 
                    self.crouch_cancel(mode = 2)
                else:
                    self.crouch_cancel(mode = 3)
            '''
            #4 Do not over shield
            if ai.shield_strength < 10:
                self.no_shield()
                self.shield_charging = True



            #3 L_cancel check
            if ai.speed_y_self < 0 and abs(ai.speed_y_self) > 2:
                self.l_cancel_check()
            #2 Grab Mash    
            if ai.action in [Action.GRABBED_WAIT_HIGH, Action.GRAB_PULLING_HIGH, Action.PUMMELED_HIGH, Action.GRAB_WAIT, Action.GRAB_PULL, Action.GRABBED, Action.GRAB_PUMMELED, Action.YOSHI_EGG, Action.CAPTURE_YOSHI, Action.THROWN_FORWARD, Action.THROWN_BACK, Action.THROWN_UP, Action.THROWN_DOWN, Action.THROWN_DOWN_2]:
                self.mash_mode = True
            else:
                self.mash_mode = False
  
            if self.gamestate.projectiles:
                for projectiles in self.gamestate.projectiles:
            
                    if projectiles.type != enums.ProjectileType.ARROW and (abs(ai.position.x - projectiles.position.x) <= 25 and abs(ai.position.y - projectiles.position.y) <=10):
                        if ai.shield_strength > 20 and not self.shield_charging:                            
                            self.emergency_shield()
            
            # Related aspects to recovery
            if (ai.on_ground and ai.action == Action.SWORD_DANCE_2_HIGH) or ai.action.value <= 10: # cyclone is charged when down b occurs on ground
                self.cyclone = False
            
            if ai.on_ground or not ai.off_stage:
                self.side_b = False
            
            if ai.action in [Action.EDGE_HANGING, Action.EDGE_CATCHING] and ai.character != Character.LINK:
                self.macro_mode = False
                self.macro_queue = []
                self.macro_idx = 0

            #1 SDI
            elif ai.hitlag_left > 1:
                op = self.gamestate.players[3 - self.agent_id]
                if self.sdi is None:
                    self.sdi = SDI()
                self.action = self.sdi.get_action(self.gamestate, ai, op)
                #print(str(self.action)+ "\n")
                
                return self.action, self._current_log_prob
            #1 Recovery
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

        if self.mash_mode:
            #print("mash", self.cnt)
            self.action = self.mash_queue[self.mash_idx]
            self.prev = self.action
            self.mash_idx += 1

            if self.mash_idx == len(self.mash_queue):
                self.mash_idx = 0 

            return self.action, self._current_log_prob



        if self.macro_mode:
            #print("macro", self.cnt)
            
            self.action = self.macro_queue[self.macro_idx]
            self.prev = self.action
            self.macro_idx += 1
            
            if self.macro_idx >= len(self.macro_queue):
                self.macro_idx = 0
                self.macro_mode = False
                self.macro_queue = []
                self.action_cnt = 3 # initialize action queue

            #print(self.action, "in agent macro")
            

            
            return self.action, self._current_log_prob
            
        else:
            
            if self.action_cnt >= self.delay : #or 21 <= self.prev <= 24: # 3
                rnn = {"rnn": self._rnn_initial_states["policy"]} if self._rnn else {}
                actions, log_prob, outputs = self.policy.act({"states": self._state_preprocessor(states), **rnn}, role="policy")
                self._current_log_prob = log_prob
                if not self.training:
                    prob = F.softmax(outputs["net_output"] / self.tou, dim=-1)
                    # prevent suicide action
                    if isinstance(self.gamestate, melee.gamestate.GameState):
                        prob = self.mask_suicide_action(prob)
                    self.action = torch.multinomial(prob, num_samples=1)
                    # self.action = torch.argmax(outputs["net_output"]).unsqueeze(0)
                else:
                    self.action = actions
                self.action_cnt = 1
                
                if self._rnn:
                    self._rnn_final_states["policy"] = outputs.get("rnn", [])
            else:
                self.action_cnt += 1
            self.prev = self.action
            #self.action = 0
            #print(self.action, "in agent normal")
            return self.action, self._current_log_prob
                  
    def state_preprocess(self, gamestate):
        return state_preprocess(gamestate, self.agent_id, self.platform)
    
    
    def l_cancel_check(self):
        ai = self.gamestate.players[self.agent_id]
        edge_pos = melee.stages.EDGE_GROUND_POSITION[self.gamestate.stage]

        left_height, left_left, left_right = melee.left_platform_position(self.gamestate)
        right_height, right_left, right_right = melee.right_platform_position(self.gamestate)
        top_height, top_left, top_right = melee.top_platform_position(self.gamestate)

        is_valid_action = ai.action in [Action.FALLING,Action.FALLING_FORWARD,Action.FALLING_BACKWARD,Action.FALLING_AERIAL,
        Action.FALLING_AERIAL_FORWARD,Action.FALLING_AERIAL_BACKWARD,Action.DOWN_B_AIR,Action.SWORD_DANCE_1_AIR, 
        Action.SWORD_DANCE_2_HIGH_AIR, Action.SWORD_DANCE_2_MID_AIR, Action.SWORD_DANCE_3_HIGH_AIR, 
        Action.SWORD_DANCE_3_MID_AIR, Action.SWORD_DANCE_3_LOW_AIR, Action.SWORD_DANCE_4_HIGH_AIR, 
        Action.SWORD_DANCE_4_MID_AIR, Action.SWORD_DANCE_4_LOW_AIR, Action.SPECIAL_FALL_FORWARD, Action.SPECIAL_FALL_BACK, 
        Action.DOWNSMASH, Action.NAIR, Action.FAIR, Action.BAIR, Action.UAIR, Action.DAIR
]

        if self.prev_action == Action.YOSHI_EGG:
            return
        if (ai.position.y > -1e-4 and ai.position.y < 3) and (abs(ai.position.x) < edge_pos - 10 and is_valid_action):
            self.macro_mode = True
            b = random.random()
            self.macro_idx = 0
            print("l cancel",b)
            self.macro_queue = [19]
            return
        elif ((-1e-4 <ai.position.y - left_height < 3) or (-1e-4 <ai.position.y - top_height < 4) if top_height else top_height)and ((left_left <ai.position.x < left_right) or (right_left <ai.position.x < right_right)):
            self.macro_mode = True
            b = random.random()
            self.macro_idx = 0
            print("l cancel",b)
            self.macro_queue = [19]
        else:
            return
        
    def no_shield(self):
        ai = self.gamestate.players[self.agent_id]
        if ai.action in [Action.SHIELD, Action.SHIELD_REFLECT, Action.SHIELD_RELEASE]:
            self.macro_mode = True
            self.macro_queue = [22] #May change to random selectabel oos options
            self.macro_idx = 0
        else:
            self.macro_mode = False
            self.macro_idx = 0
            self.macro_queue = []
    def emergency_shield(self):

        ai = self.gamestate.players[self.agent_id]
        
        if ai.on_ground:
            
            self.macro_mode = True
            print("emergency shield")
            self.macro_queue = [19]*2
            self.macro_idx = 0
            
            return
        
        else:

            return

    def mash(self):

        #mash_inputs = [False]*7 + [0]*6
        #mash_keys = [0,1,2,3,5] #A B X Y L -> A B X Y L ...
        return

    def crouch_cancel(self, mode):
        ai = self.gamestate.players[self.agent_id]
        num = mode
        if num == 3:
            return
        
        if ai.percent > 60 and num == 2:
            
            self.macro_mode = False
            self.macro_idx = 0
            self.macro_queue = []
            
            return

        if not ai.on_ground:
            
            self.macro_mode = False
            self.macro_idx = 0
            self.macro_queue = []
            
            return
            
        if ai.action in [Action.CROUCHING, Action.CROUCH_START, Action.CROUCH_END]:
            self.macro_mode = False
            self.macro_idx = 0
            self.macro_queue = []
            
            return

        self.macro_mode = True
        self.macro_idx = 0
        self.macro_queue = [3, 3, 3]

        
        return
    
    # make possible to use gamestate during training
    def record_transition(self, states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps):
        super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)
        self.gamestate = infos["gamestate"]
    
    def mario_recovery(self):
        # maybe optimal?            
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
                if ai.position.y > -20 and abs(ai.position.x) - edge_pos > 0: # just move
                    self.macro_queue = [2, 2, 2] if is_left else [1, 1, 1]
                elif abs(ai.position.x) - edge_pos > 40 and not self.cyclone:
                    self.macro_queue = [33, 33, 31, 31] * 15 if is_left else [32, 32, 30, 30] * 15
                    self.cyclone = True
                elif abs(ai.position.x) - edge_pos > 40 and ai.position.y > -10 and not self.side_b: # 40 => more larger ex.50?
                    self.macro_queue = [10, 10, 10] if is_left else [9, 9, 9] # apply side B
                    self.macro_queue += [2, 2] * 20 if is_left else [1, 1] * 20 # this is important
                    self.side_b = True
                else: # up B
                    self.macro_queue = [14, 14, 14] if is_left else [13, 13, 13]
    
    def link_recovery(self):
        # maybe optimal?
        edge_pos = melee.stages.EDGE_GROUND_POSITION[self.gamestate.stage]
        self.macro_mode = True
        self.macro_idx = 0
        ai = self.gamestate.players[self.agent_id]
        is_left = ai.position.x < 0
        if abs(ai.position.x) < edge_pos:
            is_left = not is_left
        edge_diff= abs(ai.position.x) - edge_pos
        
        # impossible hook recovery
        if ((not ai.facing) and is_left) or (ai.facing and (not is_left)):
            if ai.position.y >= -5 or ai.speed_y_self > 0: # just move -> consider side B only once
                self.macro_queue = [2] if is_left else [1]
            elif ai.jumps_left > 0: #jump
                self.macro_queue = [21] if is_left else [20]
            elif(ai.position.y < -50): #b special
                self.macro_queue = [14] if is_left else [13]
            else:
                self.macro_queue = [2] if is_left else [1]
        elif ai.action in [Action.EDGE_HANGING, Action.EDGE_CATCHING, Action.EDGE_GETUP_SLOW, \
    Action.EDGE_GETUP_QUICK, Action.EDGE_ATTACK_SLOW, Action.EDGE_ATTACK_QUICK, Action.EDGE_ROLL_SLOW, Action.EDGE_ROLL_QUICK]:
            self.macro_queue = [19, 0] # use L key
        elif ai.action in [Action.DOWN_B_GROUND]: # if agent holds grab
            self.macro_queue = [15, 0]
        elif ai.position.y >= -5 or ai.speed_y_self > 0: # just move -> consider side B only once
            self.macro_queue = [2] if is_left else [1]
        else:
            if (ai.action in [Action.AIRDODGE]): #grab should be executed within 49 frame after airdodge
                if (edge_diff < 50 or ai.action_frame >= 30) and ai.position.y <= float(1e-04):
                    self.macro_queue = [17] if is_left else [16]
                else:
                    self.macro_queue = [19]    
            elif ai.jumps_left > 0: #jump
                self.macro_queue = [21] if is_left else [20]
            elif (ai.facing and not is_left) or ((not ai.facing) and is_left):
                self.macro_queue = [10] if is_left else [9]
            else: 
                if(ai.position.y < -50): #b special
                    self.macro_queue = [14] if is_left else [13]
                elif(edge_diff < 120): # airdodge
                    self.macro_queue = [35] if is_left else [34] #arrow + L. 
                else: #move
                    self.macro_queue = [2] if is_left else [1]
    
    def pikachu_recovery(self):
        # maybe optimal?
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
        edge_diff = abs(ai.position.x) - edge_pos
        is_left = ai.position.x < 0
        if ai.position.y >= -5 or ai.speed_y_self > 0: # just move -> consider side B only once
            self.macro_queue = [2] if is_left else [1]
        elif ai.jumps_left > 0: #jump
            self.macro_queue = [21] if is_left else [20]
        else: 
            if (edge_diff < 20) and not ai.action in [Action.JUMPING_FORWARD, Action.JUMPING_BACKWARD,
                                                      Action.JUMPING_ARIAL_FORWARD, Action.JUMPING_ARIAL_BACKWARD]: # airdodge
                self.macro_queue = [35] if is_left else [34]
            else: #move
                self.macro_queue = [2] if is_left else [1]
            
    
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
                elif abs(ai.position.x) - edge_pos > 20 and not self.cyclone:
                    self.macro_queue = [33, 33, 31, 31] * 15 if is_left else [32, 32, 30, 30] * 15
                    self.cyclone = True
                elif abs(ai.position.x) - edge_pos < 15: # up B
                    self.macro_queue = [14, 14, 14] if is_left else [13, 13, 13]
                else:
                    self.macro_queue = [2, 2, 2] if is_left else [1, 1, 1]
   
    def mask_suicide_action(self, prob):
        # prevent suicide
        ai = self.gamestate.players[self.agent_id]
        edge_pos = melee.stages.EDGE_GROUND_POSITION[self.gamestate.stage]
        edge_diff = edge_pos - abs(ai.position.x)
        is_left = True if ai.position.x < 0 else False

        if edge_diff < 35: # prevent to jump
            prob[0][23 if is_left else 24] = 0
            prob[0][20 if is_left else 21] = 0
                     
        if edge_diff < 32 and ai.position.y > float(1e-04):
            # prevent dodge or attack while jump
            prob[0][6 if is_left else 7] = 0
            prob[0][34 if is_left else 35] = 0
        if edge_diff < 25:
            prob[0][1 if is_left else 2] = 0
                        
        if ai.character in [Character.MARIO, Character.DOC]:
            prob = self.mario_mask(prob)
        elif ai.character == Character.LINK:
            prob = self.link_mask( prob)
        elif ai.character == Character.PIKACHU:
            prob = self.pikachu_mask(prob)
        elif ai.character == Character.YOSHI:
            prob = self.yoshi_mask(prob)
        elif ai.character == Character.LUIGI:
            prob = self.luigi_mask(prob)
        return prob
    
    def luigi_mask(self, prob):
        ai = self.gamestate.players[self.agent_id]
        edge_pos = melee.stages.EDGE_GROUND_POSITION[self.gamestate.stage]
        is_left = True if ai.position.x < 0 else False
        edge_diff = edge_pos - abs(ai.position.x)
        
        if edge_diff < 65: # prevent side b
            prob[0][9 if is_left else 10] = 0
        if edge_diff < 40: # prevent side cyclone
            prob[0][30 if is_left else 31] = 0
        return prob
    
    def mario_mask(self, prob):
        ai = self.gamestate.players[self.agent_id]
        edge_pos = melee.stages.EDGE_GROUND_POSITION[self.gamestate.stage]
        is_left = True if ai.position.x < 0 else False
        edge_diff = edge_pos - abs(ai.position.x)
        
        if edge_diff < 25: # prevent up b
            prob[0][13 if is_left else 14] = 0
        if edge_diff < 20: # prevent side cyclone
            prob[0][30 if is_left else 31] = 0
        return prob
    
    def yoshi_mask(self, prob):
        ai = self.gamestate.players[self.agent_id]
        edge_pos = melee.stages.EDGE_GROUND_POSITION[self.gamestate.stage]
        is_left = True if ai.position.x < 0 else False
        edge_diff = edge_pos - abs(ai.position.x)
        
        if edge_diff < 18:
            # prevent down B
            if (is_left and not ai.facing) or (not is_left and ai.facing):
                prob[0][11] = 0
                prob[0][30] = 0
                prob[0][31] = 0
        # if yoshi roll near edge, move opposite direction
        if edge_diff < 35 and ai.action in [Action.SWORD_DANCE_3_MID_AIR, Action.SWORD_DANCE_4_LOW, Action.SWORD_DANCE_1_AIR]:
            if is_left and (ai.speed_air_x_self < 0 or ai.speed_ground_x_self < 0):
                prob[0][2] = 1 # force move
            elif not is_left and (ai.speed_air_x_self > 0 or ai.speed_ground_x_self > 0):
                prob[0][1] = 1
        if edge_diff < 25 and not ai.action in [Action.SWORD_DANCE_3_MID_AIR, Action.SWORD_DANCE_4_LOW, Action.SWORD_DANCE_1_AIR]:
            # do not roll near the edge
            prob[0][9 if is_left else 10] = 0
            prob[0][13 if is_left else 14] = 0
        return prob
    
    def link_mask(self, prob):
        ai = self.gamestate.players[self.agent_id]
        edge_pos = melee.stages.EDGE_GROUND_POSITION[self.gamestate.stage]
        is_left = True if ai.position.x < 0 else False
        edge_diff = edge_pos - abs(ai.position.x)
        
        if edge_diff < 25:
            if ai.position.y > float(1e-04):
                if (is_left and ai.speed_air_x_self < 0) or ((not is_left) and ai.speed_air_x_self > 0):
                    prob[0][3] = 0
                    prob[0][15] = 0
                    prob[0][16] = 0
                    prob[0][17] = 0
        return prob
    
    def pikachu_mask(self, prob):
        ai = self.gamestate.players[self.agent_id]
        edge_pos = melee.stages.EDGE_GROUND_POSITION[self.gamestate.stage]
        is_left = True if ai.position.x < 0 else False
        edge_diff = edge_pos - abs(ai.position.x)
        if edge_diff < 20: # need to verify distance
            if ai.position.y > float(1e-04):
                if (is_left and ai.speed_air_x_self < 0) or ((not is_left) and ai.speed_air_x_self > 0):
                    prob[0][11] = 0
        if edge_diff < 65:
            prob[0][9 if is_left else 10] = 0
        if edge_diff < 58:
            prob[0][13 if is_left else 14] = 0
        return prob

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
        state = np.zeros((864,), dtype=np.float32)
    else:
        state = np.zeros((880,), dtype=np.float32)
        
    state[0] = p1.position.x / edge_pos
    state[1] = p1.position.y / edge_pos
    state[2] = p2.position.x / edge_pos
    state[3] = p2.position.y / edge_pos
    state[4] = gamestate.distance / 20
    state[5] = (edge_pos - abs(p1.position.x)) / edge_pos
    state[6] = (edge_pos - abs(p2.position.x)) / edge_pos
    state[7] = 1.0 if p1.facing else -1.0 # facing right 1 else 0
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
    
    # state[36] = (p1.ecb.top.y - 12) / 2.5
    # state[37] = (p1.ecb.bottom.y - 2) / 2
    # state[38] = (p1.ecb.left.x - 2.7)
    # state[39] = (p1.ecb.left.y - 7) / 2
    # state[40] = p1.ecb.right.x + 2.8
    # state[41] = (p1.ecb.right.y + 2.8) / 10
    
    if p1.action.value < 386:
        state[36 + p1.action.value] = 1.0
    if p2.action.value < 386:
        state[36 + 386 + p2.action.value] = 1.0
    
    # if the type is same, then apply only once
    projs = [x for x in gamestate.projectiles if x.owner == 3 - agent_id and x.type in proj_mapping.keys()]
    for i, proj in enumerate(projs):
        state[36 + 386 * 2 + 4 * proj_mapping[proj.type]] = proj.position.x / edge_pos
        state[36 + 386 * 2 + 4 * proj_mapping[proj.type] + 1] = proj.position.y / edge_pos
        state[36 + 386 * 2 + 4 * proj_mapping[proj.type] + 2] = proj.speed.x / 2
        state[36 + 386 * 2 + 4 * proj_mapping[proj.type] + 3] = proj.speed.y / 2
    
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

        state[864] = 1.0 if p1_on_left else 0
        state[865] = 1.0 if p2_on_left else 0
        state[866] = 1.0 if p1_on_right else 0
        state[867] = 1.0 if p2_on_right else 0
        state[868] = 1.0 if p1_on_top else 0
        state[869] = 1.0 if p2_on_top else 0
        state[870] = (abs(p1.position.x) - right_left) / edge_pos
        state[871] = (abs(p2.position.x) - right_left) / edge_pos
        state[872] = (abs(p1.position.x) - right_right) / edge_pos
        state[873] = (abs(p2.position.x) - right_right) / edge_pos
        state[874] = (p1.position.y - right_height) / 70
        state[875] = (p2.position.y - right_height) / 70
        state[876] = (abs(p1.position.x) - top_right) / edge_pos if top_height is not None else 0
        state[877] = (abs(p2.position.x) - top_right) / edge_pos if top_height is not None else 0
        state[878] = (p1.position.y - top_height) / top_height if top_height is not None else 0
        state[879] = (p2.position.y - top_height) / top_height if top_height is not None else 0
        
    return state
