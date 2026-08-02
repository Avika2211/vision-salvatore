# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
"""
------------------------------------------------------------------------------------

Project:     Save jpg image
Description: Save image in JPEG format to filesystem given by filename, defaulting to aim_img.jpg
------------------------------------------------------------------------------------
"""
import sys
import os
import time
import argparse
# add aim_example_base folder to sys.path, use of abspath is necessary if Python < 3.9:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../examples/advanced')))
from aim_example_base import AimExampleBase
from vex import aim

    
    
def save_jpg_to_file(robot: aim.Robot, filename):
    """Example function demonstrating use of robot.get_camera_image() to save jpg to filesystem"""
    start_time = time.time()
    try:
        image = robot.vision.get_camera_image()
    except aim.NoImageException:
        print("NoImageException: probably timed out, time_elapsed: %f" %(time.time() - start_time))
        return
    print("robot.get_camera_image() took", time.time() - start_time)
    print("image variable type is", type(image))
    if image == 0:
        print("error, no image received from robot")
        return
    with open(filename, "wb") as file:
        file.write(image)
        file.close()
    print("saved \"%s\" to current working directory" %(filename))

class SaveJpg(AimExampleBase):
    def __init__(self):
        super().__init__()
        self.parser.add_argument('--file', type=str, default="aim_img.jpg", help="what name to save the jpg as.")
    
    def run(self):
        self.init_robot()
        save_jpg_to_file(self.robot, self.args.file)

if __name__ == "__main__":
    example = SaveJpg()
    example.run()