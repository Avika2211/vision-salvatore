# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
"""
------------------------------------------------------------------------------------

Project:      Object Detection Alert
Description:  This project will move the AIM robot forward until any cargo is found.
              When detected, the robot will play a beep sound, and stop all robot movement.

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
        robot.sound.play(BLINKER)
        while robot.sound.is_active():
            wait(50, MSEC)
        robot.stop_all_movement()
        break
    wait(5, MSEC)
