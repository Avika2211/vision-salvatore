# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
"""
------------------------------------------------------------------------------------

Project:      Timing Events
Description:  This project starts driving forward but stops automatically once five 
              seconds have elapsed, then displays “Time’s up!” on the screen.

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
# Callback to stop all movement
def event_timer():
    robot.screen.clear_screen()
    robot.screen.set_font(MONO30)
    robot.screen.print_at("Time's up!", x=50, y=120)
    robot.stop_all_movement()

# System event handler fired after 5 seconds
robot.timer.event(event_timer, 5000 )
# Add 15ms delay to make sure events are registered correctly.
wait(15, MSEC)

def timer_countdown():
    countdown = 5
    # Display the timer countdown on the robot's screen
    for repeat_count in range(countdown):
        robot.screen.set_font(MONO60)
        robot.screen.print_at(countdown, x=90, y=130)
        countdown -= 1
        wait(1, SECONDS)

robot.set_move_velocity(30, PERCENT)
robot.move_at(0)
timer_countdown()
