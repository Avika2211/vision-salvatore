# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
"""
------------------------------------------------------------------------------------

Project:      Drive two robots from the same project
Description:  This project makes two robots to move in a square path, 
              changing the LED color at each color
Note:         pass the IP Address of the robot to the Robot constructor
------------------------------------------------------------------------------------
"""
# Library imports
from vex import *
from vex.vex_globals import *

#update robot's IP address
robot1=Robot("192.168.2.51")
robot2 = Robot("192.168.2.53")

# Begin project code
def move_in_square(robot):
    """
    Function to move the robot in a square path
    """
    robot.led.on(ALL_LEDS, BLUE)
    robot.move_for(100, 0)
    robot.led.on(ALL_LEDS, RED)
    robot.move_for(100, 270)
    robot.led.on(ALL_LEDS, GREEN)
    robot.move_for(100, 180)
    robot.led.on(ALL_LEDS, ORANGE)
    robot.move_for(100, 90)

# Create threads to run the move_in_square function for each robot
move_robot_1_thread = Thread(move_in_square, args=(robot1,))
move_robot_2_thread = Thread(move_in_square, args=(robot2,))
