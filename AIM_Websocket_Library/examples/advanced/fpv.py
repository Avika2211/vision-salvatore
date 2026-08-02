# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
"""
------------------------------------------------------------------------------------

Project:      AIM FPV (First Person View) control
Description:  This project allows the user to control the robot while streaming images
              from the camera.  LEDs are green when robot is moving, red when robot is
              stopped.
              Audio recording won't work in WSL.
Image window keybindings:
    Esc: Quit the program
    Spacebar: Actuate kicker
    Left mouse button: click to drag a line that will command robot to move
    R: press to start recording audio, then press again to stop and send
------------------------------------------------------------------------------------
"""
from threading import Thread
import math
import time
import enum
import wave
import cv2
import pyaudio
import numpy as np
from vex import aim
from vex import vex
from aim_example_base import AimExampleBase

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000
WAVE_OUTPUT_FILENAME = "voice.wav"

class RecordingState(enum.Enum):
    IDLE = 0,
    START_RECORDING = 1,
    RECORDING = 2,
    STOP_RECORDING = 3,
    SAVING = 4

class AudioThread(Thread):
    """
    Thread to manage recording and sending of audio to AIM for playback.
    """
    def __init__(self, robot: aim.Robot):
        Thread.__init__(self)
        self.current_state = RecordingState.IDLE
        self.robot_instance = robot

    def run(self):
        total_size = 0
        while True:
            if self.current_state == RecordingState.IDLE:
                time.sleep(0.1)
            if self.current_state == RecordingState.START_RECORDING:
                self.p = pyaudio.PyAudio()
                stream = self.p.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RATE,
                                input=True,
                                frames_per_buffer=CHUNK)

                print("* recording")
                frames = []
                total_size = 0
                self.current_state = RecordingState.RECORDING

            if self.current_state == RecordingState.RECORDING:
                data = stream.read(CHUNK)
                if (total_size+len(data)) < 255 * 1024:
                    total_size += len(data)
                    frames.append(data)
                else: # if the file is at max size, stop recording and send
                    self.current_state = RecordingState.STOP_RECORDING

            if self.current_state == RecordingState.STOP_RECORDING:
                print("* done recording, frames length: %d, audio size: %d" %(len(frames), total_size))

                stream.stop_stream()
                stream.close()
                self.p.terminate()
                self.current_state = RecordingState.SAVING

            if self.current_state == RecordingState.SAVING:
                wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                self.robot_instance.sound.play_local_file(WAVE_OUTPUT_FILENAME, 100)
                self.current_state = RecordingState.IDLE

mouseY = 0
mouseX = 0


btnDown = False
def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY,btnDown

    if event == cv2.EVENT_MOUSEMOVE:
        mouseX = x
        mouseY = y
    if event == cv2.EVENT_LBUTTONDOWN:
        btnDown = True
    if event == cv2.EVENT_LBUTTONUP:
        btnDown = False


max_speed = 200.0

def fpv_control_loop(robot: aim.Robot, crosshairs):
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('img',draw_circle)
    ok_flag = True
    image = bytes(1)
    driving = False
    last_W1 = 0.0
    last_W2 = 0.0
    last_W3 = 0.0
    # For displaying FPS
    image_last = bytes(1)
    time_current_img = time.time() # just an initial value
    time_diff = 0
    time_elapsed = 0
    frames = 0
    fps = 0
    time_last = time.time()
    audio_thread = AudioThread(robot)
    audio_thread.daemon = True
    audio_thread.start()
    while ok_flag:
        # Use numpy to construct an array from the bytes
        try:
            image = robot.vision.get_camera_image()
        except aim.NoImageException or aim.DisconnectedException:
            image = bytes(1)

        # For displaying FPS
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

        #decode the array into an image
        img = cv2.imdecode(x, cv2.IMREAD_UNCHANGED)

        # Draw various items (text, lines) to the image before displaying
        cv2.putText(img, "ESC to exit", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, -100), 1, cv2.LINE_AA)
        cv2.putText(img, "%.2f" %(fps), (0, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, -100), 1, cv2.LINE_AA) #for testing FPS
        if audio_thread.current_state == RecordingState.RECORDING:
            cv2.putText(img, "recording sound", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, -100), 1, cv2.LINE_AA)
        if(crosshairs): # if user passed "-x" as an argument
            img = cv2.line(img, (320,0), (320,480), (0, 255, 0), 1)
            img = cv2.line(img, (0,240), (640,240), (0, 255, 0), 1)
        if(btnDown): # if left moust ubtton is currently pressed
            img = cv2.line(img, (320,240), (mouseX,mouseY), (0, 255, 0), 9)
            jsX = 0.0
            jsY = (float(mouseY - 240) / 240.0) * -1.0
            R = (float(mouseX - 320.0) / 320.0) * -1.0
            W1 = (-0.5 * jsX - (math.sqrt(3)/2) * jsY + R) * -1.0
            W2 = (-0.5 * jsX + (math.sqrt(3)/2) * jsY + R) * -1.0
            W3 = (jsX + R) * -1.0
            print("W1: %f, W2: %f, W3: %f" %(W1, W2, W3))
            if W1 != last_W1 or W2 != last_W2 or W3 != last_W3:
                last_W1 = W1
                last_W2 = W2
                last_W3 = W3
                robot.spin_wheels(W1*max_speed, W2*max_speed, W3*max_speed)
            if not driving:
                print("start")
                robot.led.on(vex.LightType.ALL_LEDS, vex.Color.GREEN)
            driving = True
        else:
            if not crosshairs:
                img = cv2.line(img, (320,240), (320,240), (0, 255, 0), 5)
            if driving:
                print("stop")
                robot.spin_wheels(0, 0, 0)
                robot.led.on(vex.LightType.ALL_LEDS, vex.Color.RED)
            driving = False
        # Display the image:
        try:
            cv2.imshow("img", img)
        except:
            print("error showing img (not available yet?)")
            time.sleep(0.050)

        key = cv2.waitKey(1)
        if key == 27: # exit if esc key is pressed
            print("detected escape key")
            ok_flag = False

        elif key == ord(' '): # space
            robot.kicker.kick(vex.KickType.MEDIUM)

        elif key == ord('r'): #press r to start recording, then r again to stop recording and send
            print("detected r key")
            if audio_thread.current_state == RecordingState.IDLE:
                audio_thread.current_state = RecordingState.START_RECORDING

            if audio_thread.current_state == RecordingState.RECORDING:
                audio_thread.current_state = RecordingState.STOP_RECORDING

    cv2.destroyAllWindows()

    print("done")

class AIMFPV(AimExampleBase):
    def __init__(self):
        super().__init__()
        self.parser.add_argument('-x', action="store_true", help="draw crosshairs")

    def run(self):
        self.init_robot()
        fpv_control_loop(self.robot, self.args.x)

if __name__ == "__main__":
    example = AIMFPV()
    example.run()