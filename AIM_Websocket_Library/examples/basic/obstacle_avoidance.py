# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
"""
------------------------------------------------------------------------------------

Project:      Obstacle Avoidance
Description:  This project will move the AIM robot forward until any cargo is found.
              When detected, the robot will turn right for 45 degrees.

------------------------------------------------------------------------------------
"""
# Library imports
from vex import *
from vex.vex_globals import *

# Robot initialization for AIM platform
robot = Robot()

# Begin project code
while True:
    robot.move_at(0)
    vision_data = robot.vision.get_data(ALL_CARGO)
    if vision_data is not None and len(vision_data) > 0:
        robot.turn_for(RIGHT, 45)
    wait(5, MSEC)
