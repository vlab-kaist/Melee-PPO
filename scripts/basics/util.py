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
        state1 = state_preprocess(gamestate, 1, platform=True)
        state2 = state_preprocess(gamestate, 2, platform=True)
        
        return (state1, state2), reward, done, gamestate

    def _reset(self):
        self.__init__()
        print("observation space got reset!")

class ActionSpace:
    def __init__(self):
        mid = np.sqrt(2) / 2
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
            melee.enums.Button.BUTTON_L, #4
            melee.enums.Button.BUTTON_Y, #5
            melee.enums.Button.BUTTON_C #6
            ]
        
    def __call__(self, controller):
        controller.release_all()      
        if self.state[2]:
            if self.state[2] == 4.0:
                controller.press_shoulder(melee.enums.Button.BUTTON_L, 1.0)
            elif self.state[2] == 6.0:
                controller.tilt_analog_unit(melee.enums.Button.BUTTON_C, 
                                    self.state[0], self.state[1])
            else: 
                controller.press_button(self.buttons[int(self.state[2])])
        if self.state[2] != 6.0:
            controller.tilt_analog_unit(melee.enums.Button.BUTTON_MAIN, 
                                        self.state[0], self.state[1])
        
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
