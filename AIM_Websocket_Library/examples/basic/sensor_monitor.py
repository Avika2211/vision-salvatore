# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
"""
------------------------------------------------------------------------------------

Project:      Sensor Monitor
Description:  This project will print the timer, heading, and yaw orientation values
              to the Robot screen

------------------------------------------------------------------------------------
"""
# Library imports
from vex import *
from vex.vex_globals import *

# Robot initialization for AIM platform
robot = Robot()

# Restore text / background defaults
robot.screen.set_pen_color(WHITE)
robot.screen.set_fill_color(BLUE)
robot.screen.set_font(MONO24)
robot.screen.clear_screen(BLUE)

# Reset the Inertial Sensor heading and rotation
robot.inertial.reset_heading()
robot.inertial.reset_rotation()

# Begin project code
# Change the font size to fit on the Robot screen
robot.screen.set_font(MONO20)
while True:
    # set cursor back to the top-left corner at the start of each loop iteration.
    robot.screen.set_cursor(1, 1)
    # Print the values of the sensor data
    robot.screen.print("Timer: ")
    robot.screen.print(robot.timer.time(SECONDS))
    robot.screen.next_row()
    robot.screen.print("Heading: ")
    robot.screen.print(f"{robot.inertial.get_heading():6.2f}")
    robot.screen.next_row()
    robot.screen.print("Yaw Orientation: ")
    robot.screen.next_row()
    robot.screen.print(f"{robot.inertial.get_yaw():6.2f}")
    robot.screen.next_row()
