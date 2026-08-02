# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
"""
------------------------------------------------------------------------------------

Project:      AI Vision For Loops
Description:  This project uses a proper Python For Loop to iterate through AI Vision
              data to calculate a mid-point between an Orange Barrel and Blue Barrel

------------------------------------------------------------------------------------
"""
# Library imports
from vex import *
from vex.vex_globals import *

# Robot initialization for AIM platform
robot=Robot()

# Begin project code
while True:
    # Create variables to store the centerX of each barrel, reset after each loop
    orange_center_x = None
    blue_center_x = None
    vision_data = robot.vision.get_data(ALL_CARGO)
    # Use a for loop to iterate through the vision data
    for obj in vision_data:
        if obj.id == ORANGE_BARREL:
            # If an orange barrel was detected, assign the orange barrel centerX coordinate
            orange_center_x = obj.centerX
        if obj.id == BLUE_BARREL:
            # If a blue barrel was detected, assign the blue barrel centerX coordinate
            blue_center_x = obj.centerX
        wait(5, MSEC)
    # Check if we have centerX variables set for each barrel
    if orange_center_x is not None and blue_center_x is not None:
        # Calculate the midpoint coordinates
        mid_center_x = (orange_center_x + blue_center_x) / 2
        robot.screen.print_at("Midpoint centerX:", x=20, y=120)
        robot.screen.print_at(mid_center_x, x=80, y=150)
    else:
        robot.screen.print_at("No barrels found", x=20, y=120)
        robot.screen.print_at("                 ", x=20, y=150)

    wait(0.5, SECONDS)
