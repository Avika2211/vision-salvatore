# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
"""
------------------------------------------------------------------------------------

Project:      Conditional Kicker
Description:  This project will turn the AIM robot to rotate 180 degrees to the right. 
              If the robot is holding an orange barrel, it will kick it;
              otherwise, the robot will do nothing.

------------------------------------------------------------------------------------
"""
# Library imports
from vex import *
from vex.vex_globals import *

# Robot initialization for AIM platform
robot = Robot()

# Begin project code
robot.turn_for(RIGHT, 180)
if robot.has_orange_barrel():
    robot.kicker.kick(MEDIUM)
