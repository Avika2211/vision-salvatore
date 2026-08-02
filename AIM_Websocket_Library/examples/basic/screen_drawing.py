# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
"""
------------------------------------------------------------------------------------

Project:      Screen Drawing
Description:  This project draws a line, square, and circle on its screen, 
              showing the name of each shape while it's being drawn

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

# Begin project code
# Draw a red rectangle
robot.screen.print_at("Rectangle", x=70, y=40)
robot.screen.set_fill_color(RED)
robot.screen.draw_rectangle(80, 80, 80, 100)
robot.screen.set_fill_color(TRANSPARENT)
wait(3, SECONDS)

# Draw a yellow line
robot.screen.clear_screen()
robot.screen.print_at("Line", x=90, y=40)
robot.screen.set_pen_color(YELLOW)
robot.screen.draw_line(80, 80, 160, 160)
robot.screen.set_pen_color(WHITE)
wait(3, SECONDS)

# Draw a green circle
robot.screen.clear_screen()
robot.screen.print_at("Circle", x=90, y=40)
robot.screen.set_fill_color(GREEN)
robot.screen.draw_circle(120, 120, 40)
robot.screen.set_fill_color(TRANSPARENT)
wait(3, SECONDS)
robot.screen.clear_screen()
