# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
"""
------------------------------------------------------------------------------------

Project:     Image video stream
Description: JPEG image stream, using opencv to display to screen.
------------------------------------------------------------------------------------
"""
import time
import cv2
import numpy as np
from vex import *
# Create an instance of the Robot class
robot = Robot()

def stream_video():
    """Call this function to open a new window and start video stream from AIM"""
    print("opening new window with AIM video stream")
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)

    ok_flag = True
    image = bytes(1)

    #for testing FPS
    image_last = bytes(1)
    time_elapsed = 0
    frames = 0
    fps = 0
    time_last = time.time()
    while ok_flag:
        time.sleep(0.010)
       
        try:
            image = robot.vision.get_camera_image()
        except (NoImageException, DisconnectedException):
            image = bytes(1)
        # calculate FPS
        if image != b'\x00' and image != image_last:
            time_elapsed += time.time() - time_last
            time_last = time.time()
            image_last = image
            frames += 1
            if time_elapsed > 1:
                fps = frames/time_elapsed
                frames = 0
                time_elapsed = 0

        x = np.frombuffer(image, dtype='uint8')

        # decode the array into an image
        img = cv2.imdecode(x, cv2.IMREAD_UNCHANGED)
        # display the FPS
        cv2.putText(img, "FPS: %.2f" %(fps), (0, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # display esc key to exit
        cv2.putText(img, "Press Esc to exit", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # show the image frame
        cv2.imshow("img", img)
        if cv2.waitKey(1) == 27: # exit if esc key is pressed
            ok_flag = False

    cv2.destroyAllWindows()

    print("done")

# start the video stream
stream_video()
