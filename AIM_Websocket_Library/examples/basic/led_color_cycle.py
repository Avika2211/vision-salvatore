# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
"""
------------------------------------------------------------------------------------

Project:      LED Color Cycle
Description:  This project cycles through each LED color,
              pausing for one second between each

------------------------------------------------------------------------------------
"""
# Library imports
from vex import *
from vex.vex_globals import *

# Robot initialization for AIM platform
robot = Robot()

# Begin project code
# Create an array of all possible color options
color_list = [
    RED, GREEN, BLUE, WHITE, YELLOW, ORANGE, PURPLE, CYAN
]

# Iterate over each color directly
for color in color_list:
    robot.led.on(ALL_LEDS, color)
    wait(1, SECONDS)

robot.led.off(ALL_LEDS)
