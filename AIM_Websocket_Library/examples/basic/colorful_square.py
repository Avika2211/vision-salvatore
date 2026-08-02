# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
"""
------------------------------------------------------------------------------------

Project:      Colorful Square
Description:  This project makes the robot move in a square path, 
              changing the LED color at each color

------------------------------------------------------------------------------------
"""
# Library imports
from vex import *
from vex.vex_globals import *

# Robot initialization for AIM platform
robot=Robot()

# Begin project code
robot.led.on(ALL_LEDS, BLUE)
robot.move_for(100, 0)
robot.led.on(ALL_LEDS, RED)
robot.move_for(100, 270)
robot.led.on(ALL_LEDS, GREEN)
robot.move_for(100, 180)
robot.led.on(ALL_LEDS, ORANGE)
robot.move_for(100, 90)
