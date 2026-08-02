# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
"""
------------------------------------------------------------------------------------

Project:      Angled Moving
Description:  This project moves the robot at 45-degree angles at 100%
              velocity to trace a diamond shape

------------------------------------------------------------------------------------
"""
# Library imports
from vex import *
from vex.vex_globals import *

# Robot initialization for AIM platform
robot=Robot()

# Begin project code
robot.set_move_velocity(100)
robot.move_for(100, 45)
robot.move_for(100, 315)
robot.move_for(100, 225)
robot.move_for(100, 135)
