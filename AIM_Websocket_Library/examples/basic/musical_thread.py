# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
"""
------------------------------------------------------------------------------------
 
Project:      Musical Threads
Description:  This project will play a song (using notes) in one thread while
              driving in another thread at the same time
 
------------------------------------------------------------------------------------
"""
# Library imports
from vex import *
from vex.vex_globals import *

# Robot initialization for AIM platform
robot = Robot()

# Begin project code
# Function to drive in a loop
def drive_square_loop():
    while True:
        robot.move_for(100, 0)
        robot.turn_for(RIGHT, 90)
        wait(5, MSEC)

# Function to play melody in a loop
def play_melody_loop():
    while True:
        robot.sound.play_note("G#5", 500)
        while robot.sound.is_active():
            wait(50, MSEC)
        robot.sound.play_note("C5", 500)
        while robot.sound.is_active():
            wait(50, MSEC)
        robot.sound.play_note("E5", 500)
        while robot.sound.is_active():
            wait(50, MSEC)
        robot.sound.play_note("A5", 250)
        while robot.sound.is_active():
            wait(50, MSEC)
        wait(5, MSEC)

# Register functions to threads
melody_thread = Thread(play_melody_loop)
drive_thread = Thread(drive_square_loop)
