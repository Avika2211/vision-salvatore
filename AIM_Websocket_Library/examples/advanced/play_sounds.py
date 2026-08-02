# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
"""
------------------------------------------------------------------------------------

Project:     Play sound file and notes
Description: The robot will play a local sound file, then play some notes from
             Beethoven's Symphony No. 5.
------------------------------------------------------------------------------------
"""
from aim_example_base import AimExampleBase
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #use of abspath is necessary if Python < 3.9
from vex import *

def test_play_sound_file(robot: Robot, file):
    """
    Play the specified file and wait until it is finished
    """
    filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), file))

    robot.sound.play_local_file(filepath)
    while robot.sound.is_active():
        print("playing sound file")
        sleep(100)
    print("finished")
    sleep(500)

NOTE_TIME    = 1000
def test_beethoven(robot: Robot):
    """
    Play the first few notes from Beethoven's Symphony No. 5.
    """
    play_note_and_wait(robot, "G5",  NOTE_TIME/4, volume=60)
    play_note_and_wait(robot, "G5",  NOTE_TIME/4, volume=60)
    play_note_and_wait(robot, "G5",  NOTE_TIME/4, volume=60)
    play_note_and_wait(robot, "Eb5", NOTE_TIME,   volume=60)

    sleep(NOTE_TIME/4)
    play_note_and_wait(robot, "F5",  NOTE_TIME/4, volume=60)
    play_note_and_wait(robot, "F5",  NOTE_TIME/4, volume=60)
    play_note_and_wait(robot, "F5",  NOTE_TIME/4, volume=60)
    play_note_and_wait(robot, "D5",  NOTE_TIME*2)

    sleep(NOTE_TIME/4)
    play_note_and_wait(robot, "G5",  NOTE_TIME/4, volume=20)
    play_note_and_wait(robot, "G5",  NOTE_TIME/4, volume=20)
    play_note_and_wait(robot, "G5",  NOTE_TIME/4, volume=20)

    play_note_and_wait(robot, "Eb5", NOTE_TIME/4, volume=20)
    play_note_and_wait(robot, "Ab5", NOTE_TIME/4, volume=20)
    play_note_and_wait(robot, "Ab5", NOTE_TIME/4, volume=20)
    play_note_and_wait(robot, "Ab5", NOTE_TIME/4, volume=20)

    play_note_and_wait(robot, "G5",  NOTE_TIME/4, volume=20)
    play_note_and_wait(robot, "Eb6", NOTE_TIME/4, volume=20)
    play_note_and_wait(robot, "Eb6", NOTE_TIME/4, volume=20)
    play_note_and_wait(robot, "Eb6", NOTE_TIME/4, volume=20)

    play_note_and_wait(robot, "C6",  NOTE_TIME*0.75, volume=20)


def play_note_and_wait(robot: Robot, note: str, duration: int, volume: int = 50):
    """
    Play the specified note on AIM and for the specified duration.
    """
    start_time = time.time()
    robot.sound.play_note(note, duration, volume)
    while robot.sound.is_active():
        time_elapsed = time.time() - start_time
        if time_elapsed*1000 > duration+3: # limit gap between notes for more seamless playback
            break
        sleep(10)

class TestSoundFile(AimExampleBase):
    def __init__(self):
        super().__init__()
        self.parser.add_argument('--file', type=str, default="./audio/i-am-vex-aim.wav", help="name of the file to play (WAV or MP3)")

    def run(self):
        self.init_robot()

        test_play_sound_file(self.robot, self.args.file)
        test_beethoven(self.robot)

if __name__ == "__main__":
    example = TestSoundFile()
    example.run()