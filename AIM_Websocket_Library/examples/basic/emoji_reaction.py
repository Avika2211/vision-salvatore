# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
"""
------------------------------------------------------------------------------------

Project:      Emoji Reaction
Description:  This project displays a happy emoji when the robot’s angle is stable 
              and a surprised emoji when tilted.

------------------------------------------------------------------------------------
"""
# Library imports
from vex import *
from vex.vex_globals import *

# Robot initialization for AIM platform
robot = Robot()

# Begin project code
while True:
    # Detect if the robot's inertial sensor roll angle is outside the threshold of stable positioning
    while math.fabs(robot.inertial.get_roll()) > 5:
        robot.screen.show_emoji(SURPRISED, LOOK_FORWARD)
        wait(5, MSEC)
    robot.screen.show_emoji(HAPPY, LOOK_FORWARD)
    wait(5, MSEC)
    