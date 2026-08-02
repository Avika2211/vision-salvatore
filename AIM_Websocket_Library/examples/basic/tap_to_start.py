# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
"""
------------------------------------------------------------------------------------

Project:      Tap to Start
Description:  This project waits for a touchscreen tap before moving forward 200 mm 
              and displaying an emoji at the end

------------------------------------------------------------------------------------
"""
# Library imports
from vex import *
from vex.vex_globals import *

# Robot initialization for AIM platform
robot = Robot()

# Restore text / background defaults
robot.screen.set_pen_color(WHITE)
robot.screen.set_fill_color(BLUE)
robot.screen.set_font(MONO24)
robot.screen.clear_screen(BLUE)

# Begin project code
# Display message and pause until the screen is pressed
robot.screen.print_at("Tap to Start", x=55, y=120)
while not robot.screen.pressing():
    wait(5, MSEC)
# Move and give visual feedback when the screen is pressed
robot.move_for(200, 0)
robot.screen.show_emoji(PROUD, LOOK_FORWARD)
wait(2, SECONDS)
