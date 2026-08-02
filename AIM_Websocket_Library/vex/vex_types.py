# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
""" 
AIM WebSocket API - Types

This module defines various types and enums used in the AIM WebSocket API.
"""
from enum import Enum
from typing import Union
import re

class SoundType(str, Enum):
    '''The defined sounds for AIM.'''
    DOORBELL = "DOORBELL"
    TADA = "TADA"
    FAIL = "FAIL"
    SPARKLE = "SPARKLE"
    FLOURISH = "FLOURISH"
    FORWARD = "FORWARD"
    REVERSE = "REVERSE"
    RIGHT = "RIGHT"
    LEFT = "LEFT"
    BLINKER = "BLINKER"
    CRASH = "CRASH"
    BRAKES = "BRAKES"
    HUAH = "HUAH"
    PICKUP = "PICKUP"
    CHEER = "CHEER"
    SENSING = "SENSING"
    DETECTED = "DETECTED"
    OBSTACLE = "OBSTACLE"
    LOOPING = "LOOPING"
    COMPLETE = "COMPLETE"
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    SEND = "SEND"
    RECEIVE = "RECEIVE"

    ACT_HAPPY = "ACT_HAPPY"
    ACT_SAD = "ACT_SAD"
    ACT_EXCITED = "ACT_EXCITED"
    ACT_ANGRY = "ACT_ANGRY"
    ACT_SILLY = "ACT_SILLY"

class FontType(str, Enum):
    '''A unit representing font type and size'''
    MONO20 = "MONO20"
    MONO24 = "MONO24"
    MONO30 = "MONO30"
    MONO36 = "MONO36"
    MONO40 = "MONO40"
    MONO60 = "MONO60"
    PROP20 = "PROP20"
    PROP24 = "PROP24"
    PROP30 = "PROP30"
    PROP36 = "PROP36"
    PROP40 = "PROP40"
    PROP60 = "PROP60"
    MONO15 = "MONO15"
    MONO12 = "MONO12"

class KickType(str, Enum):
    """The defined units for the kicker kick strength."""
    SOFT = "kick_soft"
    MEDIUM = "kick_medium"
    HARD = "kick_hard"

class AxisType(Enum):
    """The defined units for inertial sensor axis."""
    X_AXIS = 0
    Y_AXIS = 1
    Z_AXIS = 2

class TurnType(Enum):
    """The defined units for robot turn direction"""
    LEFT = 0
    RIGHT = 1

class OrientationType(Enum):
    '''The defined units for inertial sensor orientation.'''
    ROLL = 0
    PITCH = 1
    YAW = 2

class AccelerationType(Enum):
    '''The defined units for inertial sensor acceleration.'''
    FORWARD = 0
    RIGHTWARD = 1
    DOWNWARD = 2

class PercentUnits(Enum):
    '''The measurement units for percentage values.'''
    PERCENT = 0
    '''A percentage unit that represents a value from 0% to 100%'''

class RotationUnits(Enum):
    '''The measurement units for rotation values.'''
    DEG = 0
    '''A rotation unit that is measured in degrees.'''
    REV = 1
    '''A rotation unit that is measured in revolutions.'''
    RAW = 99
    '''A rotation unit that is measured in raw data form.'''

class DriveVelocityUnits(Enum):
    '''The measurement units for drive velocity values.'''
    PERCENT = 0
    '''A velocity unit that is measured in percentage.'''
    MMPS = 1
    '''A velocity unit that is measured in mm per second.'''

class TurnVelocityUnits(Enum):
    '''The measurement units for turn velocity values.'''
    PERCENT = 0
    '''A velocity unit that is measured in percentage.'''
    DPS = 1
    '''A velocity unit that is measured in degrees per second.'''

class TimeUnits(Enum):
    '''The measurement units for time values.'''
    SECONDS = 0
    '''A time unit that is measured in seconds.'''
    MSEC = 1
    '''A time unit that is measured in milliseconds.'''

class DistanceUnits:
    '''The measurement units for distance values.'''
    MM = 0
    '''A distance unit that is measured in millimeters.'''
    IN = 1
    '''A distance unit that is measured in inches.'''
    CM = 2
    '''A distance unit that is measured in centimeters.'''

# ----------------------------------------------------------
# globals
# ----------------------------------------------------------
PERCENT = PercentUnits.PERCENT
'''A percentage unit that represents a value from 0% to 100%'''
LEFT = TurnType.LEFT
'''A turn unit that is defined as left turning.'''
RIGHT = TurnType.RIGHT
'''A turn unit that is defined as right turning.'''
DEGREES = RotationUnits.DEG
'''A rotation unit that is measured in degrees.'''
TURNS = RotationUnits.REV
'''A rotation unit that is measured in revolutions.'''
SECONDS = TimeUnits.SECONDS
'''A time unit that is measured in seconds.'''
MSEC = TimeUnits.MSEC
'''A time unit that is measured in milliseconds.'''
INCHES = DistanceUnits.IN
'''A distance unit that is measured in inches.'''
MM = DistanceUnits.MM
'''A distance unit that is measured in millimeters.'''
MMPS = DriveVelocityUnits.MMPS
'''units of mm per second'''
DPS = TurnVelocityUnits.DPS
'''units of degrees per second'''
OFF = False
'''used to turn off an LED'''

# drivetrain move functions take either DriveVelocity or percentage units
DriveVelocityPercentUnits = Union[DriveVelocityUnits, PercentUnits]
# drivetrain turn functions take either TurnVelocity or percentage units
TurnVelocityPercentUnits = Union[TurnVelocityUnits, PercentUnits]

class LightType(str, Enum):
    '''The defined indexes for the AIM LEDs.'''
    LED1 = "light1"
    LED2 = "light2"
    LED3 = "light3"
    LED4 = "light4"
    LED5 = "light5"
    LED6 = "light6"
    ALL_LEDS = "all"

class Color:
    '''### Color class - create a new color

    This class is used to create instances of color objects

    #### Arguments:
        value: The color value, can be specified in various ways, see examples.

    #### Returns:
        An instance of the Color class

    #### Examples:
    ```
        # create blue using hex value
        c = Color(0x0000ff)
        # create blue using r, g, b values
        c = Color(0, 0, 255)
        # create blue using web string
        c = Color("#00F")
        # create blue using web string (alternate)
        c = Color("#0000FF")
        # create red using an existing object
        c = Color(Color.RED)
    ```
    '''
    class DefinedColor:
        '''### DefinedColor class - create a new DefinedColor'''
        def __init__(self, value, transparent=False):
            self.value = value
            self.transparent = transparent

        def __str__(self):
            return Color._format_color_as_str(self.value)

        def __repr__(self):
            return Color._format_color_as_str(self.value)

    BLACK       = DefinedColor(0x000000)
    '''predefined Color black'''
    WHITE       = DefinedColor(0xFFFFFF)
    '''predefined Color white'''
    RED         = DefinedColor(0xFF0000)
    '''predefined Color red'''
    GREEN       = DefinedColor(0x00FF00)
    '''predefined Color green'''
    BLUE        = DefinedColor(0x001871)
    '''predefined Color blue'''
    YELLOW      = DefinedColor(0xFFFF00)
    '''predefined Color yellow'''
    ORANGE      = DefinedColor(0xFF8500)
    '''predefined Color orange'''
    PURPLE      = DefinedColor(0xFF00FF)
    '''predefined Color purple'''
    CYAN        = DefinedColor(0x00FFFF)
    '''predefined Color cyan'''
    TRANSPARENT = DefinedColor(0x000000, True)
    '''predefined Color transparent'''

    def __init__(self, *args):
        self.transparent = False
        if len(args) == 1 and isinstance(args[0], int):
            self.value: int = args[0]
        elif len(args) == 1 and isinstance(args[0], str):
            self.value = self._color_from_str(args[0])
        elif len(args) == 3 and all(isinstance(arg, int) for arg in args):
            self.value = ((args[0] & 0xFF) << 16) + ((args[1] & 0xFF) << 8) + (args[2] & 0xFF)
        else:
            raise TypeError("bad parameters")

    def __str__(self):
        return Color._format_color_as_str(self.value)

    def __repr__(self):
        return Color._format_color_as_str(self.value)

    @staticmethod
    def _format_color_as_str(value):
        return f'{value:#08x}'

    def _color_from_str(self, color:str) -> int:
        '''### convert string to color value

        #### Arguments:
            color: A valid webcolor in the form '#FFF' or '#FFFFFF'

        #### Returns:
            integer value representing the color
        '''
        value = 0
        if re.match("#[a-fA-F0-9]{3}$", color):
            r,g,b = int(color[1], 16), int(color[2], 16), int(color[3], 16)
            value = (r << 20) + (r << 16) + (g  << 12) + (g  << 8) + (b << 4) + b
        elif re.match("#[a-fA-F0-9]{6}$", color):
            value = int(color[1:], 16)             
        return value

    def set_rgb(self, *args):
        '''### change existing Color instance to new rgb value

        #### Arguments:
            value: The color value, can be specified in various ways, see examples.

        #### Returns:
            integer value representing the color

        #### Examples:
        ```
            # create a color that is red
            c = Color(0xFF0000)
            # change color to blue using single value
            c.rgb(0x0000FF)
            # change color to green using three values
            c.rgb(0, 255, 0)
        ```
        '''
        if len(args) == 1 and isinstance(args[0], int):
            self.value = args[0]
        if len(args) == 3 and all(isinstance(arg, int) for arg in args):
            self.value = ((args[0] & 0xFF) << 16) + ((args[1] & 0xFF) << 8) + (args[2] & 0xFF)

class EmojiType(Enum):
    '''The defined emoji for AIM.'''
    EXCITED = 0
    CONFIDENT = 1
    SILLY = 2
    AMAZED = 3
    STRONG = 4
    THRILLED = 5
    HAPPY = 6
    PROUD = 7
    LAUGHING = 8
    OPTIMISTIC = 9
    DETERMINED = 10
    AFFECTIONATE = 11
    CALM = 12
    QUIET = 13
    SHY = 14
    CHEERFUL = 15
    LOVED = 16
    SURPRISED = 17
    THINKING = 18
    TIRED = 19
    CONFUSED = 20
    BORED = 21
    EMBARRASSED = 22
    WORRIED = 23
    SAD = 24
    SICK = 25
    DISAPPOINTED = 26
    NERVOUS = 27
    ANNOYED = 28
    STRESSED = 29
    ANGRY = 30
    FRUSTRATED = 31
    JEALOUS = 32
    SHOCKED = 33
    FEAR = 34
    DISGUST = 35

Emoji = EmojiType

class EmojiLookType(Enum):
    '''The defined directions the emoji can look.'''
    LOOK_FORWARD = 0
    LOOK_LEFT = 1
    LOOK_RIGHT = 2

EmojiLook = EmojiLookType

class StackingType(Enum):
    '''The types for robot movement stacking (unused).'''
    STACKING_OFF = 0
    STACKING_MOVE_RELATIVE = 1
    STACKING_MOVE_GLOBAL = 2

class SensitivityType(Enum):
    '''The defined units for the crash sensitivity.'''
    LOW = 0
    MEDIUM = 1
    HIGH = 2
