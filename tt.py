import melee
from melee import enums

import time

def soft_reset(controller, console):
    console = console
    gamestate = console.step()
    while gamestate.menu_state != enums.Menu.CHARACTER_SELECT:
        cnt = 0
        if gamestate.menu_state in [enums.Menu.IN_GAME, enums.Menu.SUDDEN_DEATH]:
            controller.press_button(enums.Button.BUTTON_START)
            while gamestate.menu_state != enums.Menu.CHARACTER_SELECT:
                if cnt == 5:
                    break
                cnt += 1
                if gamestate.frame % 2 == 0:
                    controller.release_all()
                    gamestate = console.step()
                    continue
                time.sleep(1)
                print("starting process")
                for t in range(5):
                    controller.press_button(enums.Button.BUTTON_START)
                    try:    
                        gamestate = console.step()
                    except:
                        pass
                print("preparing")
                time.sleep(2)
                print("letsgo")
                controller.release_all()
                controller.press_button(enums.Button.BUTTON_L)
                controller.press_button(enums.Button.BUTTON_R)
                controller.press_button(enums.Button.BUTTON_A)
                controller.press_button(enums.Button.BUTTON_START)
                controller.flush()
                
        elif gamestate.menu_state == enums.Menu.MAIN_MENU:
            while gamestate.menu_state == enums.Menu.MAIN_MENU:
                if cnt == 5:
                    break
                cnt += 1
                if gamestate.frame % 2 == 0:
                    controller.release_all()
                    gamestate = console.step()
                    continue
                if gamestate.submenu == enums.SubMenu.ONLINE_PLAY_SUBMENU:
                    if gamestate.menu_selection == 2:
                        controller.press_button(enums.Button.BUTTON_A)
                    elif gamestate.menu_selection == 3:
                        controller.press_button(enums.Button.BUTTON_A)
                    else:
                        controller.tilt_analog(enums.Button.BUTTON_MAIN, 0.5, 0)
                elif gamestate.submenu == enums.SubMenu.MAIN_MENU_SUBMENU:
                    controller.press_button(enums.Button.BUTTON_A)
                elif gamestate.submenu == enums.SubMenu.ONEP_MODE_SUBMENU:
                    if gamestate.menu_selection == 2:
                        controller.press_button(enums.Button.BUTTON_A)
                    else:
                        controller.tilt_analog(enums.Button.BUTTON_MAIN, 0.5, 0)

                elif gamestate.submenu == enums.SubMenu.NAME_ENTRY_SUBMENU:
                    pass
                else:
                    controller.press_button(enums.Button.BUTTON_B)
                gamestate = console.step()
        else:
            gamestate = console.step()
            print("else-ed")

        
console = melee.console.Console(
    slippi_port = 51441,
    path = "/home/yoshi/.local/share/melee-env/Slippi/squashfs-root/usr/bin",
    fullscreen=False,
    disable_audio=True,
    blocking_input=True,
    setup_gecko_codes = True,
    gfx_backend = "null"
)

ctrl = melee.controller.Controller(console=console, port=4)
console.run(
    iso_path="/home/yoshi/ssbm.iso"
)
time.sleep(3)

if console.connect():
    print("히히 연결 발사~")
else:
    print("히히 던지면 그만이야 미카가 헤세드 못때리면 그만이야")

ctrl.connect()
ctrl.release_all()

time.sleep(1)

while True:
    gamestate = console.step()
    soft_reset(ctrl,console)
    while gamestate.menu_state == melee.enums.Menu.CHARACTER_SELECT:
        gamestate = console.step()