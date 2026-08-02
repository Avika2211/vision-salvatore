from aim_fsm import *
from ultralytics import YOLO
import torch
import cv2
import numpy as np

from DominoCNN import DominoCNN


# ----------------------------
# Label mapping
# ----------------------------
def make_domino_class_maps():
    pair_to_class = {}
    class_to_pair = {}
    k = 0
    for hi in range(7):
        for lo in range(hi + 1):
            pair_to_class[(hi, lo)] = k
            class_to_pair[k] = (hi, lo)
            k += 1
    return pair_to_class, class_to_pair


pair_to_class, class_to_pair = make_domino_class_maps()


def id_to_str(i):
    hi, lo = class_to_pair[i]
    return f"{hi}-{lo}"


# ----------------------------
# Perspective crop helper
# ----------------------------
def warp_obb(image, pts):
    pts = pts.astype(np.float32)

    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype="float32")

    w = int(rect[1][0])
    h = int(rect[1][1])

    w = max(w, 1)
    h = max(h, 1)

    dst = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(image, M, (w, h))

    return warped


# ----------------------------
# FSM Program
# ----------------------------
class DominoClassifier(StateMachineProgram):

    def __init__(self):
        super().__init__()

        print("Loading YOLO...")
        self.yolo = YOLO("yolo_best_weights.pt")

        print("Loading CNN...")
        self.device = "cpu"

        self.cnn = DominoCNN(num_classes=28)
        ckpt = torch.load("best_domino_cnn.pt", map_location=self.device)
        self.cnn.load_state_dict(ckpt["model_state_dict"])
        self.cnn.eval()

        self.last_result = None

    # ----------------------------
    # Detect
    # ----------------------------
    class Detect(StateNode):
        def start(self, event=None):
            super().start(event)

            img = self.parent.robot.camera_image
            results = self.parent.yolo(img, conf=0.5)

            self.parent.last_result = results[0]
            self.post_data(self.parent.last_result)

    # ----------------------------
    # Classify
    # ----------------------------
    class Classify(StateNode):
        def start(self, event=None):
            super().start(event)

            r = self.parent.last_result
            img = r.orig_img

            if r.obb is None or len(r.obb) == 0:
                print("No domino detected")
                return

            # pick highest confidence detection
            confs = r.obb.conf.cpu().numpy()
            idx = int(np.argmax(confs))

            pts = r.obb.xyxyxyxy[idx].cpu().numpy()

            # warp to upright domino
            crop = warp_obb(img, pts)

            # resize to training format
            crop = cv2.resize(crop, (100, 75))

            # normalize for CNN
            crop = torch.tensor(crop).permute(2, 0, 1).float() / 255.0
            crop = crop.unsqueeze(0)

            with torch.no_grad():
                logits = self.parent.cnn(crop)
                probs = torch.softmax(logits, dim=1)
                pred = probs.argmax(1).item()
                conf = probs.max().item()

            label = id_to_str(pred)

            print(f"I see a {label} domino! (conf: {conf:.2f})")

    # ----------------------------
    # FSM graph
    # ----------------------------
    def setup(self):

        begin = StateNode().set_name("begin").set_parent(self)
        loop = Print("Press tm to classify domino").set_name("loop").set_parent(self)
        detect = self.Detect().set_name("detect").set_parent(self)
        classify = self.Classify().set_name("classify").set_parent(self)

        TimerTrans(2).add_sources(begin).add_destinations(loop)

        TextMsgTrans().add_sources(loop).add_destinations(detect)

        DataTrans().add_sources(detect).add_destinations(classify)

        NullTrans().add_sources(classify).add_destinations(loop)

        return self