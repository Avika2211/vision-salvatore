import os
import cv2
import numpy as np
from aim_fsm import *
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
FALLEN_MODEL_PATH = os.path.join(PROJECT_DIR, "fallen.pt")

class TestFallenModel(StateMachineProgram):
    def __init__(self):
        super().__init__()
        print(f"Loading fallen.pt from: {FALLEN_MODEL_PATH}")
        self.model = YOLO(FALLEN_MODEL_PATH)
        self.model.eval()

    class RunInference(StateNode):
        def start(self, event=None):
            super().start(event)
            print("\n[TEST] Taking camera frame...")
            image = self.robot.camera_image

            if image is None:
                print("[TEST] ERROR: Camera frame empty/None!")
                self.post_failure()
                return

            print(f"[TEST] Frame captured ({image.shape[1]}x{image.shape[0]}). Running model...")
            
            # FIX: self.model -> self.parent.model
            try:
                results = self.parent.model.predict(source=image, conf=0.15, iou=0.45, verbose=False)
            except Exception as ex:
                print(f"[TEST] Exception during predict: {ex}")
                self.post_failure()
                return

            if len(results) > 0 and results[0].masks is not None and len(results[0].masks.data) > 0:
                result = results[0]
                masks = result.masks.data
                print(f"[TEST] SUCCESS! Found {len(masks)} fallen domino mask(s).")

                # OpenCV window me display overlay
                annotated = result.plot()
                imshow("Fallen Domino Test Feed", annotated)
                self.post_success()
            else:
                print("[TEST] NO MASKS DETECTED! (Model ran but found 0 objects).")
                imshow("Fallen Domino Test Feed", image)
                self.post_failure()

    def setup(self):
        begin = StateNode().set_name("begin").set_parent(self)
        run_test = self.RunInference().set_name("run_test").set_parent(self)
        done = Print("Test completed! Check the OpenCV window.").set_name("done").set_parent(self)
        failed = Print("Test failed! Check logs above.").set_name("failed").set_parent(self)

        TimerTrans(0.2).add_sources(begin).add_destinations(run_test)
        SuccessTrans().add_sources(run_test).add_destinations(done)
        FailureTrans().add_sources(run_test).add_destinations(failed)

        return self