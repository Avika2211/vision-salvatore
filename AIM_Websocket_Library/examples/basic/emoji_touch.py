# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
"""
------------------------------------------------------------------------------------

Project:      Emoji Touch
Description:  This project changes the screen's emoji based on how many times the
              screen has been pressed, showing a different expression for each press

------------------------------------------------------------------------------------
"""
# Library imports
from vex import *
from vex.vex_globals import *

# Robot initialization for AIM platform
robot = Robot()

# Begin project code
touch_count = 0
# Callback to run when the screen pressed event has been fired
def robot_pressed():
    global touch_count
    # When the screen is pressed, increase touch_count by 1
    if touch_count > 5:
        touch_count = 0
    touch_count = touch_count + 1

# System event handler for screen pressed event
robot.screen.pressed(robot_pressed)
# Add 15ms delay to make sure events are registered correctly.
wait(15, MSEC)

# Reset touch counter
touch_count = 0
# Repeat forever to update the emoji on the screen
while True:
    # Show a different emoji depending on how many times the screen was touched
    if touch_count == 1:
        robot.screen.show_emoji(AMAZED, LOOK_FORWARD)
    elif touch_count == 2:
        robot.screen.show_emoji(THRILLED, LOOK_FORWARD)
    elif touch_count == 3:
        robot.screen.show_emoji(SURPRISED, LOOK_FORWARD)
    elif touch_count == 4:
        robot.screen.show_emoji(LAUGHING, LOOK_FORWARD)
    elif touch_count == 5:
        robot.screen.show_emoji(CONFIDENT, LOOK_FORWARD)
    else:
        robot.screen.show_emoji(HAPPY, LOOK_FORWARD)
    wait(5, MSEC)
