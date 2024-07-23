import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from skrl.agents.torch.ppo import PPO, PPO_RNN, PPO_DEFAULT_CONFIG

import melee
import numpy as np
from enum import Enum
    
class PPOAgent(PPO):
    def __init__(self, agent_id=1, players=None, csv_path=None,
                 is_selfplay=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_id = agent_id
        self.action = torch.tensor([[0]], device=self.device)
        self.action_cnt = 3
        self.players = players
        self.gamestate = None
        self.csv_path = csv_path
        self.is_selfplay = is_selfplay
        
    def act(self, states, timestep: int, timesteps: int):
        #if env is not myenv, env gives gamestate itself to an agent
        if not isinstance(states, torch.Tensor):
            states = self.state_preprocess(states)
            states = torch.tensor(states, device=self.device, dtype=torch.float32).view(1, -1)
        #states = torch.tensor(states, device=self.device, dtype=torch.float32).view(1, -1)
        obs = states
        # apply same action for 3 frames (Daboy style)
        if self.action_cnt >= 3:   
            # sample stochastic actions
            actions, log_prob, outputs = self.policy.act({"states": self._state_preprocessor(obs)}, role="policy")
            self._current_log_prob = log_prob
            self.action = actions
            self.action_cnt = 1
        else:
            self.action_cnt += 1
        return self.action, self._current_log_prob

    def state_preprocess(self, gamestate):
        return state_preprocess(gamestate, self.agent_id)
    
    def record_transition(self, states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps):
        super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)
        #print(terminated[0].item())
        
        if self.is_selfplay:
            for i in range(states.shape[0]):
                if terminated[i] and self.gamestate is not None:
                    # load data from csv file
                    with open(self.csv_path, mode='r') as file:
                        reader = csv.DictReader(file)
                        for row in reader:
                            if int(row["ID"]) == self.players["p1"].id:  
                                self.players["p1"].elo = float(row["elo"])
                                self.players["p1"].wins = int(row["wins"])
                                self.players["p1"].loses = int(row["loses"])
                            if int(row["ID"]) == self.players["p2"].id:  
                                self.players["p2"].elo = float(row["elo"])
                                self.players["p2"].wins = int(row["wins"])
                                self.players["p2"].loses = int(row["loses"])
                    
                    # calculate elo rating
                    k1 = 1 + 18 / (1 + 2 ** ((self.players["p1"].elo - 1500) / 63))
                    k2 = 1 + 18 / (1 + 2 ** ((self.players["p2"].elo - 1500) / 63))
                    e1 = 1 / (1 + 10 ** ((self.players["p2"].elo - self.players["p1"].elo) / 4))
                    e2 = 1 / (1 + 10 ** ((self.players["p1"].elo - self.players["p2"].elo) / 4))
                    
                    if self.gamestate.player[1].stock > self.gamestate.player[2].stock:
                        self.players["p1"].wins += 1
                        self.players["p2"].loses += 1
                        self.players["p1"].elo += k1 * (1 - e1)
                        self.players["p2"].elo += k2 * (0 - e2)
                    elif self.gamestate.player[1].stock < self.gamestate.player[2].stock:
                        self.players["p1"].loses += 1
                        self.players["p2"].wins += 1
                        self.players["p1"].elo += k1 * (0 - e1)
                        self.players["p2"].elo += k2 * (1 - e2)
                    else:
                        self.players["p1"].elo += k1 * (0.5 - e1)
                        self.players["p2"].elo += k2 * (0.5 - e2)
                        
                    # write chagned data to csv file
                    p1_data = [self.players["p1"].id, self.players["p1"].wins, self.players["p1"].loses, self.players["p1"].elo]
                    p2_data = [self.players["p2"].id, self.players["p2"].wins, self.players["p2"].loses, self.players["p2"].elo]
                    with open(self.csv_path, 'r', newline='') as file:
                        reader = csv.reader(file)
                        rows = list(reader)
                    rows[self.players["p1"].id + 1] = p1_data
                    rows[self.players["p2"].id + 1] = p2_data
                    with open(self.csv_path, 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerows(rows)
        self.gamestate = infos["gamestate"]
        

class StackedPPOAgent(PPO): # 
    def __init__(self, agent_id=1, stack_size=4, players=None, csv_path=None,
                 is_selfplay=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_id = agent_id
        self.action = torch.tensor([[0]], device=self.device)
        self.action_cnt = 3
        self.stack_size = stack_size
        self.stacked_obs = None
        self.gamestate = None
        self.players = players
        self.csv_path = csv_path
        self.is_selfplay = is_selfplay
        
    def act(self, states, timestep: int, timesteps: int):
        # when agent is not training. it use gamestate itself
        if not isinstance(states, torch.Tensor):
            states = self.state_preprocess(states)
            states = torch.tensor(states, device=self.device, dtype=torch.float32).view(1, -1)
            if self.stacked_obs is None:
                self.stacked_obs = np.tile(states.cpu(), (self.stack_size, 1))
            else:
                self.stacked_obs = np.roll(self.stacked_obs, -1, axis=0)
                self.stacked_obs[-1] = states.cpu()
            stacked_obs = torch.tensor(self.stacked_obs, device=self.device, dtype=torch.float32).view(1, -1)
        else:
            stacked_obs = states # when use stacked env
        # apply same action for 3 frames (Daboy style)
        if self.action_cnt >= 3:   
            # sample stochastic actions
            actions, log_prob, outputs = self.policy.act({"states": self._state_preprocessor(stacked_obs)}, role="policy")
            self._current_log_prob = log_prob
            self.action = actions
            self.action_cnt = 1
        else:
            self.action_cnt += 1
        return self.action, self._current_log_prob
    
    def state_preprocess(self, gamestate):
        return state_preprocess(gamestate, self.agent_id)
    
    def record_transition(self, states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps):
        super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)
        
        # when terminated, update wins, loses, elos in csv file
        if self.is_selfplay:
            for i in range(states.shape[0]):
                if terminated[i] and self.gamestate is not None:
                    # load data from csv file
                    with open(self.csv_path, mode='r') as file:
                        reader = csv.DictReader(file)
                        for row in reader:
                            if int(row["ID"]) == self.players["p1"].id:  
                                self.players["p1"].elo = float(row["elo"])
                                self.players["p1"].wins = int(row["wins"])
                                self.players["p1"].loses = int(row["loses"])
                            if int(row["ID"]) == self.players["p2"].id:  
                                self.players["p2"].elo = float(row["elo"])
                                self.players["p2"].wins = int(row["wins"])
                                self.players["p2"].loses = int(row["loses"])
                    
                    # calculate elo rating
                    k1 = 1 + 18 / (1 + 2 ** ((self.players["p1"].elo - 1500) / 63))
                    k2 = 1 + 18 / (1 + 2 ** ((self.players["p2"].elo - 1500) / 63))
                    e1 = 1 / (1 + 10 ** ((self.players["p2"].elo - self.players["p1"].elo) / 4))
                    e2 = 1 / (1 + 10 ** ((self.players["p1"].elo - self.players["p2"].elo) / 4))
                    
                    if self.gamestate.player[1].stock > self.gamestate.player[2].stock:
                        self.players["p1"].wins += 1
                        self.players["p2"].loses += 1
                        self.players["p1"].elo += k1 * (1 - e1)
                        self.players["p2"].elo += k2 * (0 - e2)
                    elif self.gamestate.player[1].stock < self.gamestate.player[2].stock:
                        self.players["p1"].loses += 1
                        self.players["p2"].wins += 1
                        self.players["p1"].elo += k1 * (0 - e1)
                        self.players["p2"].elo += k2 * (1 - e2)
                    else:
                        self.players["p1"].elo += k1 * (0.5 - e1)
                        self.players["p2"].elo += k2 * (0.5 - e2)
                        
                    # write chagned data to csv file
                    p1_data = [self.players["p1"].id, self.players["p1"].wins, self.players["p1"].loses, self.players["p1"].elo]
                    p2_data = [self.players["p2"].id, self.players["p2"].wins, self.players["p2"].loses, self.players["p2"].elo]
                    with open(self.csv_path, 'r', newline='') as file:
                        reader = csv.reader(file)
                        rows = list(reader)
                    rows[self.players["p1"].id + 1] = p1_data
                    rows[self.players["p2"].id + 1] = p2_data
                    with open(self.csv_path, 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerows(rows)
                        
        self.gamestate = infos["gamestate"]        
        
class PPOGRUAgent(PPO_RNN):
    def __init__(self, agent_id=1, players=None, csv_path=None,
                 is_selfplay=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_id = agent_id
        self.action = torch.tensor([[0]], device=self.device)
        self.action_cnt = 3
        self.players = players
        self.gamestate
        self.csv_path = csv_path
        self.is_selfplay = is_selfplay
        
    def act(self, states, timestep: int, timesteps: int):
        #if env is not myenv, env gives gamestate itself to an agent
        if not isinstance(states, torch.Tensor):
            states = self.state_preprocess(states)
            states = torch.tensor(states, device=self.device, dtype=torch.float32).view(1, -1)
        # apply same action for 3 frames (Daboy style)
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
    
    def state_preprocess(self, gamestate):
        return state_preprocess(gamestate, self.agent_id)
    
    def record_transition(self, states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps):
        super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)
        #print(terminated[0].item())
        
        if self.is_selfplay:
            for i in range(states.shape[0]):
                if terminated[i] and self.gamestate is not None:
                    # load data from csv file
                    with open(self.csv_path, mode='r') as file:
                        reader = csv.DictReader(file)
                        for row in reader:
                            if int(row["ID"]) == self.players["p1"].id:  
                                self.players["p1"].elo = float(row["elo"])
                                self.players["p1"].wins = int(row["wins"])
                                self.players["p1"].loses = int(row["loses"])
                            if int(row["ID"]) == self.players["p2"].id:  
                                self.players["p2"].elo = float(row["elo"])
                                self.players["p2"].wins = int(row["wins"])
                                self.players["p2"].loses = int(row["loses"])
                    
                    # calculate elo rating
                    k1 = 1 + 18 / (1 + 2 ** ((self.players["p1"].elo - 1500) / 63))
                    k2 = 1 + 18 / (1 + 2 ** ((self.players["p2"].elo - 1500) / 63))
                    e1 = 1 / (1 + 10 ** ((self.players["p2"].elo - self.players["p1"].elo) / 4))
                    e2 = 1 / (1 + 10 ** ((self.players["p1"].elo - self.players["p2"].elo) / 4))
                    
                    if self.gamestate.player[1].stock > self.gamestate.player[2].stock:
                        self.players["p1"].wins += 1
                        self.players["p2"].loses += 1
                        self.players["p1"].elo += k1 * (1 - e1)
                        self.players["p2"].elo += k2 * (0 - e2)
                    elif self.gamestate.player[1].stock < self.gamestate.player[2].stock:
                        self.players["p1"].loses += 1
                        self.players["p2"].wins += 1
                        self.players["p1"].elo += k1 * (0 - e1)
                        self.players["p2"].elo += k2 * (1 - e2)
                    else:
                        self.players["p1"].elo += k1 * (0.5 - e1)
                        self.players["p2"].elo += k2 * (0.5 - e2)
                        
                    # write chagned data to csv file
                    p1_data = [self.players["p1"].id, self.players["p1"].wins, self.players["p1"].loses, self.players["p1"].elo]
                    p2_data = [self.players["p2"].id, self.players["p2"].wins, self.players["p2"].loses, self.players["p2"].elo]
                    with open(self.csv_path, 'r', newline='') as file:
                        reader = csv.reader(file)
                        rows = list(reader)
                    rows[self.players["p1"].id + 1] = p1_data
                    rows[self.players["p2"].id + 1] = p2_data
                    with open(self.csv_path, 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerows(rows)
        self.gamestate = infos["gamestate"]
    

def state_preprocess(gamestate, agent_id):
    p1 = gamestate.players[1]
    p2 = gamestate.players[2]
    edge_pos = melee.stages.EDGE_GROUND_POSITION[gamestate.stage]
            
    state1 = np.zeros((808,), dtype=np.float32)
    
    state1[0] = p1.position.x / edge_pos
    state1[1] = p1.position.y / edge_pos
    state1[2] = p2.position.x / edge_pos
    state1[3] = p2.position.y / edge_pos
    state1[4] = gamestate.distance / 20
    state1[5] = (edge_pos - abs(p1.position.x)) / edge_pos
    state1[6] = (edge_pos - abs(p2.position.x)) / edge_pos
    state1[7] = 1.0 if p1.facing else -1.0
    state1[8] = 1.0 if p2.facing else -1.0
    state1[9] = 1.0 if (p1.position.x - p2.position.x) * state1[7] < 0 else -1.0
    state1[10] = p1.hitstun_frames_left / 10
    state1[11] = p2.hitstun_frames_left / 10
    state1[12] = p1.invulnerability_left / 20
    state1[13] = p2.invulnerability_left / 20
    state1[14] = p1.jumps_left - 1
    state1[15] = p2.jumps_left - 1
    state1[16] = p1.off_stage * 1.0
    state1[17] = p2.off_stage * 1.0
    state1[18] = p1.on_ground * 1.0
    state1[19] = p2.on_ground * 1.0
    state1[20] = (p1.percent - 50) / 50
    state1[21] = (p2.percent - 50) / 50
    state1[22] = (p1.shield_strength - 30) / 30
    state1[23] = (p2.shield_strength - 30) / 30
    state1[24] = p1.speed_air_x_self / 2
    state1[25] = p2.speed_air_x_self / 2
    state1[26] = p1.speed_ground_x_self / 2
    state1[27] = p2.speed_ground_x_self / 2
    state1[28] = p1.speed_x_attack
    state1[29] = p2.speed_x_attack
    state1[30] = p1.speed_y_attack
    state1[31] = p2.speed_y_attack
    state1[32] = p1.speed_y_self
    state1[33] = p2.speed_y_self
    state1[34] = (p1.action_frame - 15) / 15
    state1[35] = (p2.action_frame - 15) / 15
    
    if p1.action.value < 386:
        state1[36 + p1.action.value] = 1.0
    if p2.action.value < 386:
        state1[36 + 386 + p2.action.value] = 1.0
    # need to consider projectile, ecb, 
    p1 = gamestate.players[2]
    p2 = gamestate.players[1]
    # state for player 2
    state2 = np.zeros((808,), dtype=np.float32) 

    state2[0] = p1.position.x / edge_pos
    state2[1] = p1.position.y / edge_pos
    state2[2] = p2.position.x / edge_pos
    state2[3] = p2.position.y / edge_pos
    state2[4] = gamestate.distance / 20
    state2[5] = (edge_pos - abs(p1.position.x)) / edge_pos
    state2[6] = (edge_pos - abs(p2.position.x)) / edge_pos
    state2[7] = 1.0 if p1.facing else -1.0
    state2[8] = 1.0 if p2.facing else -1.0
    state2[9] = 1.0 if (p1.position.x - p2.position.x) * state2[7] < 0 else -1.0
    state2[10] = p1.hitstun_frames_left / 10
    state2[11] = p2.hitstun_frames_left / 10
    state2[12] = p1.invulnerability_left / 20
    state2[13] = p2.invulnerability_left / 20
    state2[14] = p1.jumps_left - 1
    state2[15] = p2.jumps_left - 1
    state2[16] = p1.off_stage * 1.0
    state2[17] = p2.off_stage * 1.0
    state2[18] = p1.on_ground * 1.0
    state2[19] = p2.on_ground * 1.0
    state2[20] = (p1.percent - 50) / 50
    state2[21] = (p2.percent - 50) / 50
    state2[22] = (p1.shield_strength - 30) / 30
    state2[23] = (p2.shield_strength - 30) / 30
    state2[24] = p1.speed_air_x_self / 2
    state2[25] = p2.speed_air_x_self / 2
    state2[26] = p1.speed_ground_x_self / 2
    state2[27] = p2.speed_ground_x_self / 2
    state2[28] = p1.speed_x_attack
    state2[29] = p2.speed_x_attack
    state2[30] = p1.speed_y_attack
    state2[31] = p2.speed_y_attack
    state2[32] = p1.speed_y_self
    state2[33] = p2.speed_y_self
    state2[34] = (p1.action_frame - 15) / 15
    state2[35] = (p2.action_frame - 15) / 15
                    
    if p1.action.value < 386:
        state2[36 + p1.action.value] = 1.0
    if p2.action.value < 386:
        state2[36 + 386 + p2.action.value] = 1.0
    
    return state1 if agent_id == 1 else state2