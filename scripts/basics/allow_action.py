from enum import Enum

from melee.enums import Action
class Input(Enum):
    No = 0x00
    ARROW = 0x01
    A = 0x02
    B = 0x04
    L = 0x08
    Z = 0x10
    ALL = 0x1f

def allow_action(ps):
    useless_action = [Action.ENTRY, Action.ENTRY_START, Action.ENTRY_END]
    if ps.action in useless_action:
        return Input.No
    IASA = {
        Action.DAIR : 48,
        Action.DOWNTILT : 19,
        Action.DOWNSMASH : 51,
        Action.UPSMASH : 41
    }
    if ps.action in IASA.keys():
        return Input.No if ps.action_frame < IASA else Input.ALL
    move_only = [Action.NAIR, Action.FAIR, Action.BAIR, Action.UAIR]
    if ps.action in move_only:
        return Input.ARROW
    no_action = [Action.SWORD_DANCE_2_HIGH_AIR, Action.DOWN_B_GROUND, Action.DOWN_B_GROUND_START, Action.SWORD_DANCE_3_MID_AIR, Action.SHINE_TURN, Action.SWORD_DANCE_3_LOW_AIR] #Down B
    no_action += [Action.LASER_GUN_PULL, Action.NEUTRAL_B_CHARGING] #B
    