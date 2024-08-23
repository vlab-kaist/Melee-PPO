import math
from melee.enums import Action, Button, Character

class SDI():
    def __init__(self):
        self.cardinal_direction = None

    @staticmethod
    def angle_to_cardinal(angle):
        """Converts an angle to the nearest cardinal direction (8 directions)."""
        angle %= 360
        if angle <= 22.5 or angle > 337.5:
            return 1, 0
        if 22.5 < angle <= 67.5:
            return 1, 1
        if 67.5 < angle <= 112.5:
            return 0, 1
        if 112.5 < angle <= 157.5:
            return -1, 1
        if 157.5 < angle <= 202.5:
            return -1, 0
        if 202.5 < angle <= 247.5:
            return -1, -1
        if 247.5 < angle <= 292.5:
            return 0, -1
        if 292.5 < angle <= 337.5:
            return 1, -1

        return 1, 1  # Default case (shouldn't happen)

    @staticmethod
    def cardinal_left(direction):
        """Returns the cardinal direction to the left of the given direction."""
        directions = {
            (1, 0): (1, 1),
            (1, 1): (0, 1),
            (0, 1): (-1, 1),
            (-1, 1): (-1, 0),
            (-1, 0): (-1, -1),
            (-1, -1): (0, -1),
            (0, -1): (1, -1),
            (1, -1): (1, 0)
        }
        return directions.get(direction, (1, 1))

    @staticmethod
    def cardinal_right(direction):
        """Returns the cardinal direction to the right of the given direction."""
        directions = {
            (1, 0.5): (1, 0),
            (1, 0): (0.5, 0),
            (0.5, 0): (0, 0),
            (0, 0): (0, 0.5),
            (0, 0.5): (0, 1),
            (0, 1): (0.5, 1),
            (0.5, 1): (1, 1),
            (1, 1): (1, 0.5)
        }
        return directions.get(direction, (1, 1))
    
    @staticmethod
    def analog_input_to_action(x, y):
        return [False, False, False, False, False, x, y, 0.0, 0.0, 0.0]

    @staticmethod
    def touching_ground(state):
        """Checks if the character is on or very close to the ground."""
        if state.on_ground:
            return True
        return abs(state.position.y) < 0.25

    def get_action(self, gamestate, smashbot_state, opponent_state):
        x, y = self.get_analog_input(gamestate, smashbot_state, opponent_state)
        return SDI.analog_input_to_action(x, y)
    
    def get_analog_input(self, gamestate, smashbot_state, opponent_state):
        print("Opponent state", opponent_state.character, opponent_state.action)
        if self.cardinal_direction is not None:
            return self.handle_committed_sdi(gamestate, smashbot_state)
            
        if (opponent_state.character, opponent_state.action) in [(Character.PIKACHU, Action.DOWNSMASH), (Character.MARIO, Action.SWORD_DANCE_1)]:
            return self.handle_situational_sdi(gamestate, smashbot_state, opponent_state)

        if smashbot_state.off_stage:
            return self.handle_off_stage_sdi(gamestate, smashbot_state)
            
        absolute_speed = math.sqrt(smashbot_state.speed_x_attack ** 2 + smashbot_state.speed_y_attack ** 2)
        if smashbot_state.percent > 60 and absolute_speed > 3:
            return self.handle_survival_sdi(gamestate, smashbot_state, opponent_state)

        return self.handle_combo_sdi(gamestate, smashbot_state, opponent_state)

    def handle_committed_sdi(self, gamestate, smashbot_state):
        print("Notes", "Committed SDI cardinal: " + str(self.cardinal_direction))
        if SDI.touching_ground(smashbot_state):
            if self.cardinal_direction[1] == 0:
                if gamestate.frame % 2:
                    return (0, 0)
                else:
                    return (self.cardinal_direction[0], 0)

        if gamestate.frame % 2:
            x, y = SDI.cardinal_right(self.cardinal_direction)
        else:
            x, y = SDI.cardinal_left(self.cardinal_direction)
        return (x, y)

    def handle_situational_sdi(self, gamestate, smashbot_state, opponent_state):
        angle = math.degrees(math.atan2(smashbot_state.speed_y_attack, smashbot_state.speed_x_attack))
        self.cardinal_direction = SDI.angle_to_cardinal(angle)
        self.cardinal_direction = (self.cardinal_direction[0], 1)
        print("Notes", "Downsmash SDI angle: " + str(angle))

        if gamestate.frame % 2:
            x, y = SDI.cardinal_right(self.cardinal_direction)
        else:
            x, y = SDI.cardinal_left(self.cardinal_direction)
        return (x, y)
    
    def handle_off_stage_sdi(self, gamestate, smashbot_state):
        cardinal = (int(smashbot_state.position.x < 0), int(smashbot_state.position.y < 0))
        print("Notes", "Off-stage SDI cardinal: " + str(cardinal))

        if gamestate.frame % 2:
            x, y = SDI.cardinal_right(cardinal)
        else:
            x, y = SDI.cardinal_left(cardinal)

        return (x, y)

    def handle_survival_sdi(self, gamestate, smashbot_state, opponent_state):
        if smashbot_state.position.y < 6:
            self.cardinal_direction = (int(smashbot_state.position.x < 0), 0)
        else:
            angle = math.degrees(math.atan2(smashbot_state.speed_y_attack, smashbot_state.speed_x_attack))
            angle = (angle + 180) % 360
            self.cardinal_direction = SDI.angle_to_cardinal(angle)
            print("Notes", "Survival SDI angle: " + str(angle) + " " + str(smashbot_state.speed_x_attack) + " " + str(smashbot_state.speed_y_attack))

            if smashbot_state.on_ground:
                if angle < 90 or angle > 270:
                    self.cardinal_direction = (1, 0)
                else:
                    self.cardinal_direction = (-1, 0)

            if SDI.touching_ground(smashbot_state):
                if self.cardinal_direction[1] == -1:
                    self.cardinal_direction = (self.cardinal_direction[0], 0)
                    if self.cardinal_direction[0] == 0:
                        self.cardinal_direction = (1, 0)

        if gamestate.frame % 2:
            x, y = SDI.cardinal_right(self.cardinal_direction)
        else:
            x, y = SDI.cardinal_left(self.cardinal_direction)

        return (x, y)

    def handle_combo_sdi(self, gamestate, smashbot_state, opponent_state):
        angle = math.degrees(math.atan2(smashbot_state.position.y - opponent_state.position.y, smashbot_state.position.x - opponent_state.position.x))
        angle = (angle + 360) % 360
        self.cardinal_direction = SDI.angle_to_cardinal(angle)
        print("Notes", "Combo SDI angle: " + str(angle))

        if smashbot_state.on_ground:
            if angle < 90 or angle > 270:
                self.cardinal_direction = (1, 0)
            else:
                self.cardinal_direction = (-1, 0)

        if SDI.touching_ground(smashbot_state):
            if self.cardinal_direction[1] == -1:
                self.cardinal_direction = (self.cardinal_direction[0], 0)
                if self.cardinal_direction[0] == 0:
                    self.cardinal_direction = (1, 0)

        if self.cardinal_direction[1] == 0:
            if gamestate.frame % 2:
                return (0, 0)
            else:
                return (self.cardinal_direction[0], 0)

        if gamestate.frame % 2:
            x, y = SDI.cardinal_right(self.cardinal_direction)
        else:
            x, y = SDI.cardinal_left(self.cardinal_direction)

        return (x, y)
