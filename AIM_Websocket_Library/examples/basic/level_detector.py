# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
"""
------------------------------------------------------------------------------------

Project:      Level Detector
Description:  In this project, the screen will show a green background with the word
              "FLAT" when the robot is not tilted. If the robot tilts, the screen
              will turn red and say "TILTED".

------------------------------------------------------------------------------------
"""
# Library imports
from vex import *
from vex.vex_globals import *

# Robot initialization for AIM platform
robot = Robot()
update_screen = False
robot_tilted = True
# Begin project code
robot.screen.set_pen_color(BLACK)
while True:
    # Only setting update_screen to True if robot_tilted has changed
    # Conditional based on robot's acceleration
    if math.fabs(robot.inertial.get_acceleration(FORWARD)) > 0.3 or math.fabs(robot.inertial.get_acceleration(RIGHTWARD)) > 0.3:
        if not robot_tilted:
            robot_tilted = True
            update_screen = True
    else:
        if robot_tilted:
            robot_tilted = False
            update_screen = True
    # Only update the screen if it needs an update
    if update_screen:
        update_screen = False
        if robot_tilted:
            # Prints "TILTED" to screen
            robot.screen.clear_screen(RED)
            robot.screen.set_fill_color(RED)
            robot.screen.set_font(MONO60)
            robot.screen.print_at("TILTED", x=40, y=140)
        else:
            # Prints "FLAT" to screen
            robot.screen.clear_screen(GREEN)
            robot.screen.set_fill_color(GREEN)
            robot.screen.set_font(MONO60)
            robot.screen.print_at("FLAT", x=60, y=140)
    wait(20, MSEC)
