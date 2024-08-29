import numpy as np
import melee
from melee import enums
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
        self.current_frame += 1
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

            w_dmg, w_shield, w_stock = 0.01, 0, 1 #0.2, 0.3, 6
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
        done = not np.sum(stocks[np.argsort(stocks)][::-1][1:]) or self.current_frame > 28500
        
        state1 = state_preprocess(gamestate, 1, platform=False)
        state2 = state_preprocess(gamestate, 2, platform=False)
        
        return (state1, state2), reward, done, gamestate

    def _reset(self):
        self.__init__()
        print("observation space got reset!")

class PlatformObservationSpace:
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
        self.current_frame += 1
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

            w_dmg, w_shield, w_stock = 0.01, 0, 1 #0.2, 0.3, 6
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
        done = not np.sum(stocks[np.argsort(stocks)][::-1][1:]) or self.current_frame > 28500
        
        # state for player 1
        state1 = state_preprocess(gamestate, 1, platform=True)
        state2 = state_preprocess(gamestate, 2, platform=True)
        
        return (state1, state2), reward, done, gamestate

    def _reset(self):
        self.__init__()
        print("observation space got reset!")

class ActionSpace:
    def __init__(self):
        mid = np.sqrt(2) / 2
        # melee.enums.Button.BUTTON_A, #1
        # melee.enums.Button.BUTTON_B, #2
        # melee.enums.Button.BUTTON_Z, #3
        # melee.enums.Button.BUTTON_L, #4
        # melee.enums.Button.BUTTON_Y, #5
        # melee.enums.Button.BUTTON_C #6
        self.action_space = np.array(
            [
                [0, 0, 0], #0
                [-1, 0, 0],  # 1
                [1, 0, 0],  # 2
                [0, -1, 0],  # 3
                [0, 0, 1],  # 4
                [0, -1, 1],  # 5
                [-1, 0, 1],  # 6
                [1, 0, 1],  # 7
                [0, 0, 2], # 8
                [-1, 0, 2], # 9
                [1, 0, 2], # 10
                [0, -1, 2],  # 11
                [0, 1, 2],  # 12
                [-mid, mid, 2],  # 13
                [mid, mid, 2],  # 14
                [0, 0, 3],  # 15
                [-1, 0, 3],  # 16
                [1, 0, 3],  # 17
                [0, -1, 3],  # 18
                [0, 0, 4],  # 19
                [-1, 0, 5],  # 20
                [1, 0, 5],  # 21
                [0, 0, 5],  # 22
                [-mid, mid, 5],  # 23
                [mid, mid, 5],  # 24
                [1, 0, 6],  # 25
                [-1, 0, 6],  # 26
                [0, 1, 6],  # 27
                [0, -1, 6],  # 28
                [0, 1, 3], # 29
                [-mid, -mid, 2], # 30
                [mid, -mid, 2], # 31
                [-mid, -mid, 0], # 32
                [mid, -mid, 0], # 33
                [-1, 0, 4], # 34
                [1, 0, 4], # 35
            ],
            dtype=np.float32,
        )
        self.size = self.action_space.shape[0]
        
    def sample(self):
        return np.random.choice(self.size)
    
    def to_controller(self, action):
        x, y, button = self.action_space[action]
        # state: [A, B, X, Y, Z, digital L, digital R, main x, main y, c stick x, c stick y, L, R]
        state = [False, False, False, False, False, False, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if button == 1.0: state[0] = True # A
        if button == 2.0: state[1] = True # B
        if button == 5.0: state[3] = True # Y
        if button == 3.0: state[4] = True # Z
        if button == 4.0: state[5] = True # Digital L
        
        if button == 6.0:
            state[9], state[10] = x, y # c stick
        else:
            state[7], state[8] = x, y # main stick
        return state

    def __call__(self, action):
        if isinstance(action, list):
            return ControlState(action)
        # # when action is index of action space
        return ControlState(self.to_controller(action))
    
class ControlState:
    def __init__(self, state):
        # state: [A, B, X, Y, Z, digital L, digital R, main x, main y, c stick x, c stick y, L, R]
        # range of state => idx 0~6: bool, 7~10: -1~1 float, 11~12: 0~1 flaot
        self.state = state
        self.buttons = [
            melee.enums.Button.BUTTON_A,
            melee.enums.Button.BUTTON_B,
            melee.enums.Button.BUTTON_X,
            melee.enums.Button.BUTTON_Y,
            melee.enums.Button.BUTTON_Z
        ]

    def __call__(self, controller):
        controller.release_all()
        for i in range(5):
            if self.state[i]:
                controller.press_button(self.buttons[i])
        controller.tilt_analog_unit(melee.enums.Button.BUTTON_MAIN,
                                    self.state[7], self.state[8])
        controller.tilt_analog_unit(melee.enums.Button.BUTTON_C,
                                self.state[9], self.state[10])
        if self.state[5]:
            controller.press_button(melee.Button.BUTTON_L)
        else:
            controller.press_shoulder(melee.enums.Button.BUTTON_L, self.state[11])
        
        if self.state[6]:
            controller.press_button(melee.Button.BUTTON_R)
        else:
            controller.press_shoulder(melee.enums.Button.BUTTON_R, self.state[12])
                   
def state_preprocess(gamestate, agent_id, platform=False):
    proj_mapping = {
            enums.ProjectileType.MARIO_FIREBALL: 0,
            enums.ProjectileType.DR_MARIO_CAPSULE: 1,
            enums.ProjectileType.LINK_BOMB: 2,
            enums.ProjectileType.LINK_HOOKSHOT: 3,
            enums.ProjectileType.ARROW: 4, #LINK ARROW
            enums.ProjectileType.PIKACHU_THUNDER: 5,
            enums.ProjectileType.MARIO_CAPE: 6,
            enums.ProjectileType.DR_MARIO_CAPE: 7,
            enums.ProjectileType.YOSHI_EGG_THROWN: 8,
            enums.ProjectileType.YOSHI_TONGUE: 9,
            enums.ProjectileType.LINK_BOOMERANG: 10,
            #enums.ProjectileType.YOSHI_STAR: 10,
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
