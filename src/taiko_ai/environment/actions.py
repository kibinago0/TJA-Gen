from enum import IntEnum


class Action(IntEnum):
    REST = 0
    DON = 1
    KA = 2
    BIG_DON = 3
    BIG_KA = 4
    ROLL_START = 5
    BALLOON_START = 6


ACTION_COUNT = len(Action)
