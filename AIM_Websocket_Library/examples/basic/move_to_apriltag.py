# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
"""
------------------------------------------------------------------------------------

Project:      Move to AprilTag
Description:  This project turns the robot until it detects
              AprilTag 1, then moves toward it

------------------------------------------------------------------------------------
"""

# Library imports
from vex import *
from vex.vex_globals import *

# Robot initialization for AIM platform
robot = Robot()
robot.vision.tag_detection(True)

# Begin project code
robot.set_turn_velocity(20)

# Turn right until AprilTag ID 1 is detected
robot.turn(RIGHT)
while True:
    vision_data = robot.vision.get_data(TAG1)

    if vision_data is not None and len(vision_data) > 0:
        # Turn to the object by adding your current heading and the vision bearing offset
        robot.turn_to(robot.inertial.get_heading() + vision_data[0].bearing)
        break
    wait(20, MSEC)

# Move to AprilTag ID 1
while True:
    vision_data = robot.vision.get_data(TAG1)

    if vision_data is not None and len(vision_data) > 0:
        if vision_data[0].width >= 60:
            robot.stop_all_movement()
            break
        else:
            robot.move_at(vision_data[0].bearing)
    else:
        robot.move_at(0)
    wait(20, MSEC)

robot.sound.play(TADA)
