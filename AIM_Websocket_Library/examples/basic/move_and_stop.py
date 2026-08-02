# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
"""
------------------------------------------------------------------------------------

Project:      Move and Stop
Description:  This project moves the robot forward 300 mm and plays 
              the "complete" sound once it has completed movement

------------------------------------------------------------------------------------
"""
# Library imports
from vex import *
from vex.vex_globals import *

# Robot initialization for AIM platform
robot = Robot()

# Begin project code
robot.move_for(300, 0)
robot.sound.play(COMPLETE)
