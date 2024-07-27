import numpy as np
import melee
from math import log

class ObservationSpace:
    def __init__(self):
        self.previous_observation = np.empty(0)
        self.previous_gamestate = None
        self.current_gamestate = None
        # should we save previous gamestates? stack obs? vs lstm?
        self.curr_action = None
        self.player_count = None
        self.current_frame = 0
        self.intial_process_complete = False

    def __call__(self, gamestate):
        reward = (0, 0)
        info = None
        self.current_gamestate = gamestate
        self.player_count = len(list(gamestate.players.keys()))
        
        if self.previous_gamestate is not None:
            p1_dmg = (
                self.current_gamestate.players[1].percent
                - self.previous_gamestate.players[1].percent
            )
            p1_shield_dmg = (
                self.previous_gamestate.players[1].shield_strength
                - self.current_gamestate.players[1].shield_strength
            ) / (self.current_gamestate.players[1].shield_strength + 1)
            
            # animation enum <= 10 denote dead
            p1_stock_loss = 0.0
            if (self.current_gamestate.players[1].action.value <= 10 and 
                self.previous_gamestate.players[1].action.value > 10):
                p1_stock_loss = 1.0
            
            p2_dmg = (
                self.current_gamestate.players[2].percent
                - self.previous_gamestate.players[2].percent
            )
            p2_shield_dmg = (
                self.previous_gamestate.players[2].shield_strength
                - self.current_gamestate.players[2].shield_strength
            ) / (self.current_gamestate.players[2].shield_strength + 1)
            p2_stock_loss = 0.0
            if (self.current_gamestate.players[2].action.value <= 10 and 
                self.previous_gamestate.players[2].action.value > 10):
                p2_stock_loss = 1.0
            # low percent is more good => efficient kill
            p1_stock_loss *= abs(200 - self.current_gamestate.players[1].percent) / 200
            p2_stock_loss *= abs(200 - self.current_gamestate.players[2].percent) / 200

            p1_dmg = max(p1_dmg, 0)
            p2_dmg = max(p2_dmg, 0)
            if p1_stock_loss > 1:
                p1_stock_loss = 0
            if p2_stock_loss > 1:
                p2_stock_loss = 0
            p1_stock_loss = max(p1_stock_loss, 0)
            p2_stock_loss = max(p2_stock_loss, 0)
            p1_shield_dmg = max(p1_shield_dmg, 0)
            p2_shield_dmg = max(p2_shield_dmg, 0)

            w_dmg, w_shield, w_stock = 0.2, 0.3, 6
            p1_loss = (
                w_dmg * p1_dmg + w_shield * p1_shield_dmg + w_stock * p1_stock_loss
            )
            p2_loss = (
                w_dmg * p2_dmg + w_shield * p2_shield_dmg + w_stock * p2_stock_loss
            )
            
            reward = (p2_loss - p1_loss, p1_loss - p2_loss)
        else:
            reward = (0, 0)

        self.previous_gamestate = self.current_gamestate

        stocks = np.array(
            [gamestate.players[i].stock for i in list(gamestate.players.keys())]
        )
        done = not np.sum(stocks[np.argsort(stocks)][::-1][1:])
        
        # state for player 1
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
        
        return (state1, state2), reward, done, gamestate

    def _reset(self):
        self.__init__()
        print("observation space got reset!")

class ActionSpace:
    def __init__(self):
        mid = np.sqrt(2) / 2
        self.action_space = np.array(
            [
                [0, 0, 0],
                [-1, 0, 0],  # 0
                [1, 0, 0],  # 1
                [0, -1, 0],  # 2
                [0, 0, 1],  # 3
                [0, -1, 1],  # 4
                [-1, 0, 1],  # 5
                [1, 0, 1],  # 6
                [0, 0, 2], # 7
                [-1, 0, 2], # 8
                [1, 0, 2], # 9
                [0, -1, 2],  # 10
                [0, 1, 2],  # 11
                [-mid, mid, 2],  # 12
                [mid, mid, 2],  # 13
                [0, 0, 3],  # 14
                [-1, 0, 3],  # 15
                [1, 0, 3],  # 16
                [0, -1, 3],  # 17
                [0, 0, 4],  # 18
                [-1, 0, 5],  # 19
                [1, 0, 5],  # 20
                [0, 0, 5],  # 21
                [-mid, mid, 5],  # 22
                [mid, mid, 5],  # 23
                [1, 0, 6],  # 24
                [-1, 0, 6],  # 25
                [0, 1, 6],  # 26
                [0, -1, 6],  # 27
            ],
            dtype=np.float32,
        )
        self.size = self.action_space.shape[0]
        
    def sample(self):
        return np.random.choice(self.size)

    def __call__(self, action):
        if action > self.size - 1:
            exit("Error: invalid action!")

        return ControlState(self.action_space[action])

class ControlState: # need to change
    def __init__(self, state):
        self.state = state
        self.buttons = [
            False, #0
            melee.enums.Button.BUTTON_A, #1
            melee.enums.Button.BUTTON_B, #2
            melee.enums.Button.BUTTON_Z, #3
            melee.enums.Button.BUTTON_R, #4
            melee.enums.Button.BUTTON_X, #5
            melee.enums.Button.BUTTON_C #6
            ]
        
    def __call__(self, controller):
        controller.release_all()      
        if self.state[2]:
            if self.state[2] == 4.0:
                controller.press_shoulder(melee.enums.Button.BUTTON_R, 1)
            elif self.state[2] == 6.0:
                controller.tilt_analog_unit(melee.enums.Button.BUTTON_C, 
                                    self.state[0], self.state[1])
            else: 
                controller.press_button(self.buttons[int(self.state[2])])
        if self.state[2] != 6.0:
            controller.tilt_analog_unit(melee.enums.Button.BUTTON_MAIN, 
                                        self.state[0], self.state[1])
        
def from_observation_space(act):
    def get_observation(self, *args):
        gamestate = args[0]
        observation = self.observation_space(gamestate)
        return act(self, observation)
    return get_observation

def from_action_space(act):
    def get_action_encoding(self, *args):
        gamestate = args[0]
        action = act(self, gamestate)
        control = self.action_space(action)
        control(self.controller)
        return 
    return get_action_encoding