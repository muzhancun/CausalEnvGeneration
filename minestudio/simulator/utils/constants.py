'''
Date: 2024-11-15 14:57:39
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2024-11-15 14:58:06
FilePath: /minestudio/simulator/constants.py
'''
import pyglet
import pyglet.window.key as key
import pyglet.window.mouse as mouse

# Mapping from simulator's action space names to pyglet keys
MINERL_ACTION_TO_KEYBOARD = {
    #"ESC":       key.ESCAPE, # Used in BASALT to end the episode
    "attack":    mouse.LEFT,
    "back":      key.S,
    #"drop":      key.Q,
    "forward":   key.W,
    "hotbar.1":  key._1,
    "hotbar.2":  key._2,
    "hotbar.3":  key._3,
    "hotbar.4":  key._4,
    "hotbar.5":  key._5,
    "hotbar.6":  key._6,
    "hotbar.7":  key._7,
    "hotbar.8":  key._8,
    "hotbar.9":  key._9,
    "inventory": key.E,
    "jump":      key.SPACE,
    "left":      key.A,
    # "pickItem":  pyglet.window.mouse.MIDDLE,
    "right":     key.D,
    "sneak":     key.LSHIFT,
    "sprint":    key.LCTRL,
    #"swapHands": key.F,
    "use":       mouse.RIGHT, 
    # "switch":    key.TAB,
    # "reset":     key.F1,
}

KEYBOARD_TO_MINERL_ACTION = {v: k for k, v in MINERL_ACTION_TO_KEYBOARD.items()}

IGNORED_ACTIONS = {"chat"}

# Camera actions are in degrees, while mouse movement is in pixels
# Multiply mouse speed by some arbitrary multiplier
MOUSE_MULTIPLIER = 0.1

MOUSE_MULTIPLIER = 0.1

MINERL_FPS = 25
MINERL_FRAME_TIME = 1 / MINERL_FPS

SCALE = 2
WINDOW_WIDTH = 640*SCALE
WINDOW_HEIGHT = 360*(SCALE+1)

screen = pyglet.canvas.get_display().get_default_screen()
ratio = 0.8

if screen.width < WINDOW_WIDTH * ratio or screen.height < WINDOW_HEIGHT * ratio:
    scale = min(screen.width * ratio / WINDOW_WIDTH, screen.height * ratio / WINDOW_HEIGHT)
    WINDOW_WIDTH = int(WINDOW_WIDTH * scale)
    WINDOW_HEIGHT = int(WINDOW_HEIGHT * scale)
    

FRAME_HEIGHT = WINDOW_HEIGHT // 4 * 3

INFO_WIDTH = WINDOW_WIDTH
INFO_HEIGHT = WINDOW_HEIGHT // 4

NUM_ROWS = 4
NUM_COLS = 6
GRID_WIDTH = WINDOW_WIDTH // NUM_COLS
GRID_HEIGHT = INFO_HEIGHT // NUM_ROWS
GRID = {}
GRID_ID = 0
for R in range(NUM_ROWS):
    for C in range(NUM_COLS):
        X = C * GRID_WIDTH + GRID_WIDTH // 5
        Y = R * GRID_HEIGHT + GRID_HEIGHT // 2
        GRID[GRID_ID] = (X, Y)
        GRID_ID += 1