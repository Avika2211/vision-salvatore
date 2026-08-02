# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
"""
------------------------------------------------------------------------------------

Project:      Get Blue Barrel and Celebrate
Description:  This project will look for a blue barrel, move to it, then celebrate
              once it has the blue barrel.

------------------------------------------------------------------------------------
"""
# Library imports
from vex import *
from vex.vex_globals import *

# Robot should be defined by default
robot = Robot()

# Begin project code
robot.set_turn_velocity(30, PERCENT)

# Robot initialization for AIM platform
robot.turn(LEFT)
while True:
    vision_data = robot.vision.get_data(BLUE_BARREL)

    if vision_data is not None and len(vision_data) > 0:
        # Turn to the object by adding your current heading and the vision bearing offset
        robot.turn_to(robot.inertial.get_heading() + vision_data[0].bearing)
        break
    wait(20, MSEC)

# This is an updated get blue barrel barrel macro
# Get blue barrel and celebrate
while True:
    vision_data = robot.vision.get_data(BLUE_BARREL)

    if vision_data is not None and len(vision_data) > 0:
        if robot.has_blue_barrel():
            robot.stop_all_movement()
            # Add celebration
            robot.screen.show_emoji(EXCITED, LOOK_FORWARD)
            robot.sound.play(TADA)
            break
        else:
            robot.move_at(vision_data[0].bearing)
    else:
        robot.move_at(0)
    wait(20, MSEC)