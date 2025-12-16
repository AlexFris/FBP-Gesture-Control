from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Optional


class ActionType(Enum):
    # Global / mode states
    SET_MODE_IDLE = auto()
    SET_MODE_LIGHT = auto()
    SET_MODE_SOUND = auto()

    # Light actions
    LIGHT_ON = auto()
    LIGHT_OFF = auto()
    SET_LIGHT_BRIGHTNESS = auto()
    SET_LIGHT_COLOR = auto()

    # Sound actions
    SOUND_PLAY = auto()
    SOUND_PAUSE = auto()
    SET_SOUND_VOLUME = auto()

    # Submodes (for OSC / visuals)
    ENTER_LIGHT_COLOR = auto()
    EXIT_LIGHT_COLOR = auto()
    ENTER_LIGHT_BRIGHTNESS = auto()
    EXIT_LIGHT_BRIGHTNESS = auto()
    ENTER_SOUND_VOLUME = auto()
    EXIT_SOUND_VOLUME = auto()


@dataclass
class Action:
    type: ActionType
    value: Optional[Any] = None
