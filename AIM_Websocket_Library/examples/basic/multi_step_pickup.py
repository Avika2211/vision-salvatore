# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
"""
------------------------------------------------------------------------------------

Project:      Multi-Step Pickup
Description:  This project turns right until an orange barrel is found, moves
              towards it until it has the barrel, then moves backwards and places it

------------------------------------------------------------------------------------
"""
# Library imports
from vex import *
from vex.vex_globals import *

# Robot initialization for AIM platform
robot = Robot()


# Begin project code
robot.set_turn_velocity(40)

# Turn right until orange barrel is detected
robot.turn(RIGHT)
while True:
    vision_data = robot.vision.get_data(ORANGE_BARREL)

    if vision_data is not None and len(vision_data) > 0:
        # Turn to the object by adding your current heading and the vision bearing offset
        robot.turn_to(robot.inertial.get_heading() + vision_data[0].bearing)
        break
    wait(20, MSEC)

# Get orange barrel
while True:
    vision_data = robot.vision.get_data(ORANGE_BARREL)

    if vision_data is not None and len(vision_data) > 0:
        if robot.has_orange_barrel():
            robot.stop_all_movement()
            break
        else:
            robot.move_at(vision_data[0].bearing)
    else:
        robot.move_at(0)
    wait(20, MSEC)

# Reverse 200 mm and place the orange barrel
robot.move_for(200, 180)
robot.kicker.place()
