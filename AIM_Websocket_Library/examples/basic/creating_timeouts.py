# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
"""
--------------------------------------------------------------------

Project:        Creating Timeouts
Description:    This project will turn the AIM robot in place while searching for
                an orange barrel. If none is found within 10 seconds, it plays a
                'fail' sound and stops the search.

--------------------------------------------------------------------
"""
# Library imports
from vex import *
from vex.vex_globals import *

# Robot initialization for AIM platform
robot = Robot()

# Begin project code
stop_searching = False
object_found = False

# Function to end the search
def stop_searching_after_timeout():
    global stop_searching
    stop_searching = True

# Register a timer event that calls our timeout function after 10 seconds (10,000 milliseconds)
robot.timer.event(stop_searching_after_timeout, 10000)

# Pause briefly to ensure the timer event is registered
wait(15, MSEC) 

robot.screen.show_aivision()
print("Searching for an orange barrel...")

robot.set_turn_velocity(20, PERCENT)
robot.turn(RIGHT)

while True:
    vision_data = robot.vision.get_data(ORANGE_BARREL)

    if vision_data is not None and len(vision_data) > 0:
        print("Found an orange barrel!")
        object_found = True
        robot.stop_all_movement()
        break

    if stop_searching:
        print("Could not find an orange barrel!")
        robot.stop_all_movement()
        robot.sound.play(FAIL)
        while robot.sound.is_active():
            wait(50, MSEC)
        break

    wait(5, MSEC)
