import os
import cv2
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image

from aim_fsm import *
from ultralytics import YOLO

# ==========================================
#  PROJECT DIR & MODEL PATHS
# ==========================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Standing Domino Models (Kick Phase)
SEG_MODEL_PATH = os.path.join(PROJECT_DIR, "bestieee.pt")
CLS_MODEL_PATH = os.path.join(PROJECT_DIR, "different.pt")

# Fallen Domino Model (Pickup Phase)
FALLEN_SEG_MODEL_PATH = os.path.join(PROJECT_DIR, "fallen.pt")

# ==========================================
#  CALIBRATION & CONSTANTS FOR KICK
# ==========================================
KNOWN_LENGTH = 4.8      # Real domino length in cm
FOCAL_LENGTH = 396.3    # Exact ground-truth setup focal value
CAMERA_HFOV_DEG = 62.0  # Horizontal FOV

NUM_CLASSES = 7
MIN_HALF_SIZE = 4
WHITE_THRESHOLD = 250
BORDER_MARGIN = 4

ALIGN_TOL_PX_NEAR = 4.0
ALIGN_TOL_PX_FAR = 10.0
NEAR_ALIGN_DISTANCE_CM = 18.0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PYTORCH_TENSOR_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ==========================================
#  CONSTANTS FOR FALLEN PICKUP STRATEGY
# ==========================================
SETTLE_DELAY_SEC = 0.30
MODEL_SIZE = 160
PREDICT_CONF = 0.15      # Relaxed for initial detection
PREDICT_IOU = 0.45
DETECT_TRIES = 5         # Increased retry attempts
DETECT_RETRY_DELAY_SEC = 0.20

TABLE_MIN_Y_RATIO = 0.05 # Lowered table line to accept closer/lower objects
PRIOR_CENTER_PX = None
PRIOR_ROI_HALF_W = 320
PRIOR_ROI_HALF_H = 240

REDETECT_TABLE_MIN_Y_RATIO = 0.05
REDETECT_PREDICT_CONF = 0.10
REDETECT_PRIOR_ROI_HALF_W = 340
REDETECT_PRIOR_ROI_HALF_H = 240
REDETECT_TRIES = 5
REDETECT_RETRY_DELAY_SEC = 0.12
REDETECT_SETTLE_DELAY_SEC = 0.30

MIN_MASK_AREA_PX = 100.0   # Relaxed area limits
MAX_MASK_AREA_PX = 200000.0
MIN_SIDE_PX = 6.0
ASPECT_MIN = 1.00          # Relaxed aspect ratios
ASPECT_MAX = 10.00

REDETECT_MAX_MASK_AREA_PX = 250000.0
REDETECT_ASPECT_MAX = 12.00
REDETECT_ASPECT_MIN = 1.00
REDETECT_MIN_SIDE_PX = 5.0

W_PRIOR = 0.35
W_DETECT = 0.45
W_SHAPE = 0.20

PICKUP_FRONT_OFFSET_MM = 30.0
FINAL_STICK_MM = 10.0
CONTACT_TOL_MM = 2.0

MIN_TURN_DEG = 3.0
MAX_TURN_DEG = 18.0
TURN_COOLDOWN_SEC = 1.0

MIN_SIDE_MM = 8.0
MAX_SIDE_MM = 35.0

MIN_FORWARD_MM = 8.0
MAX_FORWARD_MM = 45.0

MAX_TURN_STEPS = 6
MAX_SIDE_STEPS = 6
MAX_FORWARD_STEPS = 8

REDETECT_BACKUP_MM = 20.0
FINAL_TOUCH_MAX_MM = 90.0

K1 = 1.55
K2 = -58.4
PICKUP_X_BIAS_MM = 30.0


# ==========================================
#  HELPER FUNCTIONS
# ==========================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def sign(v):
    if v > 0:
        return 1.0
    if v < 0:
        return -1.0
    return 0.0

def wrap_deg(deg):
    while deg > 180.0:
        deg -= 360.0
    while deg < -180.0:
        deg += 360.0
    return deg

def split_motion(distance_mm, max_step_mm, min_step_mm):
    if abs(distance_mm) < 1e-6:
        return []
    remaining = float(distance_mm)
    pieces = []
    while abs(remaining) > max_step_mm:
        step = max_step_mm * sign(remaining)
        pieces.append(float(step))
        remaining -= step
    if abs(remaining) >= 1e-6:
        if len(pieces) > 0 and abs(remaining) < min_step_mm:
            pieces[-1] += float(remaining)
        else:
            pieces.append(float(remaining))
    return pieces

def roboflow_fit_resize(img, size=MODEL_SIZE):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y, new_w, new_h

def map_mask_to_original(mask_reduced, pad_x, pad_y, new_w, new_h, orig_w, orig_h):
    cropped = mask_reduced[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
    restored = cv2.resize(cropped.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return (restored > 0).astype(np.uint8) * 255

def long_axis_from_rect(rect):
    (cx, cy), (w, h), angle = rect
    long_side = max(float(w), float(h))
    if w >= h:
        theta = math.radians(float(angle))
    else:
        theta = math.radians(float(angle) + 90.0)
    ux, uy = math.cos(theta), math.sin(theta)
    p1 = np.array([cx - 0.5 * long_side * ux, cy - 0.5 * long_side * uy], dtype=np.float32)
    p2 = np.array([cx + 0.5 * long_side * ux, cy + 0.5 * long_side * uy], dtype=np.float32)
    return p1, p2, long_side

def project_pixel_to_base(robot, px, py):
    hit = robot.kine.project_to_ground(float(px), float(py)).copy()
    hit[0] = K1 * hit[0] + K2
    return float(hit[0][0]) + PICKUP_X_BIAS_MM, float(hit[1][0])

def estimate_angle_and_distance_cm(center_x: float, image_w: float, major_length_px: float):
    focal_px_for_angle = (image_w * 0.5) / math.tan(math.radians(CAMERA_HFOV_DEG * 0.5))
    angle_deg = math.degrees(math.atan2(center_x - (image_w * 0.5), focal_px_for_angle))
    distance_cm = (KNOWN_LENGTH * FOCAL_LENGTH) / max(major_length_px, 1.0)
    return angle_deg, distance_cm

def estimate_divider_x(image_bgr: np.ndarray, x1: float, y1: float, x2: float, y2: float):
    h, w = image_bgr.shape[:2]
    ix1 = max(0, min(int(round(x1)), w - 1))
    iy1 = max(0, min(int(round(y1)), h - 1))
    ix2 = max(ix1 + 1, min(int(round(x2)), w))
    iy2 = max(iy1 + 1, min(int(round(y2)), h))

    if ix2 <= ix1 or iy2 <= iy1:
        return None

    roi = image_bgr[iy1:iy2, ix1:ix2]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    rw = gray.shape[1]
    if rw < 12:
        return None

    c0, c1 = int(0.30 * rw), int(0.70 * rw)
    if c1 <= c0 + 1:
        return None

    profile = gray.mean(axis=0)
    band = profile[c0:c1]
    local = int(np.argmin(band))
    divider_x_local = c0 + local

    darkest, median = float(band[local]), float(np.median(band))
    if (median - darkest) < 8.0:
        return None

    return float(ix1 + divider_x_local)


# ==========================================
#  MAIN INTEGRATED STATE MACHINE PROGRAM
# ==========================================
class KickAndPickupFallen(StateMachineProgram):

    def __init__(self):
        super().__init__()
        print("Initializing Combined Kick & Fallen Domino Pipeline...")

        # Load Kick Models
        self.segmenter = YOLO(SEG_MODEL_PATH)
        self.classifier = self._load_classifier(CLS_MODEL_PATH)
        self.clahe = None

        # Load Fallen Model (fallen.pt)
        self.fallen_model = None
        if os.path.exists(FALLEN_SEG_MODEL_PATH):
            print(f"Loading Fallen Domino Weights: {FALLEN_SEG_MODEL_PATH}")
            self.fallen_model = YOLO(FALLEN_SEG_MODEL_PATH)
            self.fallen_model.eval()
        else:
            print(f"Warning: Fallen weights '{FALLEN_SEG_MODEL_PATH}' not found!")

        # Kick Memory Variables
        self.target_data = None
        self.turn_steps = 0
        self.last_charge_distance_mm = 0.0
        self.calculated_near_sideways_mm = 0.0
        self.calculated_near_turn_rad = 0.0

        # Fallen Pickup Memory Variables
        self.fallen_target = None
        self.debug_view = None
        self.model_view = None
        self.prior_center_px = PRIOR_CENTER_PX
        self.last_cmd = ""
        self.fallen_turn_steps = 0
        self.fallen_side_steps = 0
        self.fallen_forward_steps = 0
        self.motion_plan = []
        self.motion_index = 0
        self.final_touch_step = None
        self.redetect_mode = False

    def _load_classifier(self, path):
        try:
            model = efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
            state_dict = torch.load(path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()
            print(f"Classifier loaded successfully on {DEVICE}")
            return model
        except Exception as exc:
            print(f"ERROR loading classifier: {exc}")
            return None

    # --- KICK UTILITY METHODS ---
    def rotate_crop(self, image, obb):
        points = obb.reshape(4, 2).astype(np.float32)
        rect = cv2.minAreaRect(points)
        center, size, angle = rect
        width, height = size
        if width <= 0 or height <= 0:
            return None, None, 0.0
        major_length = max(width, height)
        if width < height:
            angle += 90
            width, height = height, width
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        x, y = int(center[0]), int(center[1])
        pad = 20
        x1, y1 = max(0, x - int(width / 2) - pad), max(0, y - int(height / 2) - pad)
        x2, y2 = min(image.shape[1], x + int(width / 2) + pad), min(image.shape[0], y + int(height / 2) + pad)
        crop = rotated[y1:y2, x1:x2]
        return crop, (x1, y1, x2, y2), major_length

    def split_halves(self, image):
        if image is None or image.size == 0: return None, None
        h, w = image.shape[:2]
        mid = w // 2
        if mid < MIN_HALF_SIZE or (w - mid) < MIN_HALF_SIZE: return None, None
        return image[:, :mid].copy(), image[:, mid:].copy()

    def remove_white_border(self, image, threshold=WHITE_THRESHOLD, margin=BORDER_MARGIN):
        if image is None or image.size == 0: return image
        keep_mask = np.any(image < threshold, axis=2)
        if not np.any(keep_mask): return image
        ys, xs = np.where(keep_mask)
        y0, y1 = max(0, int(ys.min()) - margin), min(image.shape[0], int(ys.max()) + margin + 1)
        x0, x1 = max(0, int(xs.min()) - margin), min(image.shape[1], int(xs.max()) + margin + 1)
        trimmed = image[y0:y1, x0:x1]
        return trimmed if (trimmed is not None and trimmed.size > 0) else image

    def preprocess_half(self, half_bgr):
        if half_bgr is None or half_bgr.size == 0: return None
        gray = cv2.cvtColor(half_bgr, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 5, 75, 75)
        if self.clahe is None: self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = self.clahe.apply(filtered)
        three_channel = np.repeat(equalized[:, :, np.newaxis], 3, axis=2)
        pil_image = Image.fromarray(three_channel)
        return PYTORCH_TENSOR_TRANSFORMS(pil_image).unsqueeze(0)

    def predict_half(self, tensor):
        if self.classifier is None or tensor is None: return None, 0.0
        with torch.no_grad():
            tensor = tensor.to(DEVICE)
            probs = torch.softmax(self.classifier(tensor), dim=1)
            confidence, predicted = torch.max(probs, dim=1)
            return int(predicted.item()), float(confidence.item())

    def measure_lateral_offset_mm(self, frame, conf=0.15):
        if frame is None: return 0.0, False
        image_h, image_w = frame.shape[:2]
        results = self.segmenter(frame, conf=conf, verbose=False)
        if not results or len(results[0]) == 0: return 0.0, False
        result = results[0]
        if not hasattr(result, "boxes") or len(result.boxes) == 0: return 0.0, False
        box = result.boxes.xyxy.cpu().numpy()[0]
        x1_box, y1_box, x2_box, y2_box = box
        divider_x = estimate_divider_x(frame, x1_box, y1_box, x2_box, y2_box)
        aim_x = float(divider_x) if divider_x is not None else float((x1_box + x2_box) * 0.5)
        lateral_angle_rad = math.atan2(float(aim_x - (image_w * 0.5)), (image_w * 0.5) / math.tan(math.radians(CAMERA_HFOV_DEG * 0.5)))
        return (50.0 * math.tan(lateral_angle_rad)) - 10.0, True

    # ==========================================
    #  STATE NODES - KICK PHASE
    # ==========================================
    class ResetRun(StateNode):
        def start(self, event=None):
            super().start(event)
            print("[KICK] Resetting run parameters...")
            self.parent.turn_steps = 0
            self.parent.target_data = None
            self.parent.last_charge_distance_mm = 0.0
            self.parent.calculated_near_sideways_mm = 0.0
            self.parent.calculated_near_turn_rad = 0.0
            self.post_completion()

    class TrackAndIdentify(StateNode):
        def start(self, event=None):
            super().start(event)
            frame = self.robot.camera_image
            if frame is None:
                self.parent.target_data = None
                self.post_failure()
                return

            image_h, image_w = frame.shape[:2]
            results = self.parent.segmenter(frame, conf=0.25, iou=0.5, agnostic_nms=True, verbose=False)
            if not results:
                self.parent.target_data = None
                self.post_failure()
                return

            result = results[0]
            candidates = []
            obb = getattr(result, "obb", None)
            boxes = getattr(result, "boxes", None)
            quads_info, detected_indexes = [], set()

            if obb is not None and len(obb) > 0:
                for idx, q in enumerate(obb.xyxyxyxy.cpu().numpy()):
                    quads_info.append((q, None))
                    detected_indexes.add(idx)

            if boxes is not None and len(boxes) > 0:
                for idx, box in enumerate(boxes.xyxy.cpu().numpy()):
                    if idx not in detected_indexes:
                        x1b, y1b, x2b, y2b = box
                        quad = np.array([[x1b, y1b], [x2b, y1b], [x2b, y2b], [x1b, y2b]], dtype=np.float32)
                        quads_info.append((quad, max(x2b - x1b, y2b - y1b)))

            for raw_obb, fallback_length in quads_info:
                crop, bbox, pixel_length = self.parent.rotate_crop(frame, raw_obb)
                if pixel_length <= 0 and fallback_length is not None: pixel_length = fallback_length
                if crop is None or crop.size == 0 or crop.shape[0] < 50 or crop.shape[1] < 50: continue

                left_raw, right_raw = self.parent.split_halves(crop)
                if left_raw is None or right_raw is None: continue

                left_clean = self.parent.remove_white_border(left_raw)
                right_clean = self.parent.remove_white_border(right_raw)
                if left_clean is None or right_clean is None: continue

                left_tensor = self.parent.preprocess_half(left_clean)
                right_tensor = self.parent.preprocess_half(right_clean)
                if left_tensor is None or right_tensor is None: continue

                left_pred, left_conf = self.parent.predict_half(left_tensor)
                right_pred, right_conf = self.parent.predict_half(right_tensor)
                if left_pred is None or right_pred is None: continue

                x1_box, y1_box, x2_box, y2_box = bbox
                divider_x = estimate_divider_x(frame, x1_box, y1_box, x2_box, y2_box)
                aim_x = float(divider_x) if divider_x is not None else float((x1_box + x2_box) * 0.5)

                angle_deg, distance_cm = estimate_angle_and_distance_cm(aim_x, image_w, pixel_length)
                candidates.append({
                    "bbox": bbox, "label": f"{left_pred}-{right_pred}",
                    "confidence": min(left_conf, right_conf), "aim_x": aim_x,
                    "cy": float((y1_box + y2_box) * 0.5), "angle_deg": angle_deg,
                    "distance_cm": distance_cm, "aim_error_px": float(aim_x - (image_w * 0.5))
                })

            valid_targets = [c for c in candidates if c["label"] in ["3-4", "4-3"]]
            if not valid_targets:
                self.parent.target_data = None
                self.post_failure()
                return

            best_target = max(valid_targets, key=lambda c: c["cy"])
            self.parent.target_data = best_target
            print(f"[KICK] Target detected: {best_target['label']} at {best_target['distance_cm']:.1f}cm")
            self.post_data(best_target)

    class CalculateTrajectory(StateNode):
        def start(self, event=None):
            super().start(event)
            target = self.parent.target_data
            if target is None:
                self.post_failure()
                return

            angle_deg, aim_error_px, distance_cm = float(target["angle_deg"]), float(target["aim_error_px"]), float(target["distance_cm"])
            align_tol_px = ALIGN_TOL_PX_NEAR if distance_cm <= NEAR_ALIGN_DISTANCE_CM else ALIGN_TOL_PX_FAR

            if abs(aim_error_px) > align_tol_px and self.parent.turn_steps < MAX_TURN_STEPS:
                turn_deg = clamp(-0.8 * angle_deg, -MAX_TURN_DEG, MAX_TURN_DEG)
                if abs(turn_deg) < MIN_TURN_DEG: turn_deg = MIN_TURN_DEG * sign(-angle_deg)
                self.parent.turn_steps += 1
                self.post_data({"cmd": "turn", "angle_deg": float(turn_deg)})
                return

            self.parent.turn_steps = 0
            self.post_success()

    class ActuateMotion(ActionNode):
        def start(self, event=None):
            super().start(event)
            if isinstance(event, DataEvent) and event.data.get("cmd") == "turn":
                deg = float(event.data.get("angle_deg", 0.0))
                self.robot.actuators["drive"].turn(self, math.radians(deg), None)
            else:
                self.post_failure()

    class ApproachCheckpoint(ActionNode):
        def start(self, event=None):
            super().start(event)
            target = self.parent.target_data
            if target is None:
                self.post_failure()
                return
            total_distance_mm = float(target["distance_cm"]) * 10.0
            self.parent.last_charge_distance_mm = total_distance_mm
            app_dist = max(0.0, total_distance_mm - 70.0)
            print(f"[KICK] Approaching checkpoint: {app_dist:.1f}mm")
            self.robot.actuators["drive"].forward(self, app_dist)

    class ScanNearZoneDrift(StateNode):
        def start(self, event=None):
            super().start(event)
            time.sleep(0.15)
            frame = self.robot.camera_image
            if frame is None:
                self.post_success()
                return
            image_h, image_w = frame.shape[:2]
            base_line_slope_rad = 0.0
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
                mask = np.zeros_like(edges)
                mask[int(image_h * 0.45):int(image_h * 0.95), :] = 255
                pts = np.argwhere(cv2.bitwise_and(edges, mask) > 0)
                if len(pts) > 20:
                    [vx, vy, x0, y0] = cv2.fitLine(np.fliplr(pts), cv2.DIST_L2, 0, 0.01, 0.01)
                    base_line_slope_rad = math.atan2(vy[0], vx[0])
            except Exception:
                base_line_slope_rad = 0.0

            error_sideways_mm, _ = self.parent.measure_lateral_offset_mm(frame, conf=0.15)
            self.parent.calculated_near_turn_rad = base_line_slope_rad
            self.parent.calculated_near_sideways_mm = error_sideways_mm
            self.post_data({"cmd": "execute_readjustment"})

    class TurnParallelNear(ActionNode):
        def start(self, event=None):
            super().start(event)
            base_slope_rad = self.parent.calculated_near_turn_rad
            if abs(math.degrees(base_slope_rad)) > 0.4:
                self.robot.actuators["drive"].turn(self, -base_slope_rad, None)
            else:
                self.post_completion()

    class RescanLateralOffset(StateNode):
        def start(self, event=None):
            super().start(event)
            time.sleep(0.1)
            frame = self.robot.camera_image
            if frame is not None:
                error_sideways_mm, detected = self.parent.measure_lateral_offset_mm(frame, conf=0.15)
                if detected: self.parent.calculated_near_sideways_mm = error_sideways_mm
            self.post_completion()

    class ShiftSidewaysNear(ActionNode):
        def start(self, event=None):
            super().start(event)
            sideways_mm = self.parent.calculated_near_sideways_mm
            if abs(sideways_mm) > 1.0:
                self.robot.actuators["drive"].sideways(self, -sideways_mm - 10.0, None)
            else:
                self.post_completion()

    class PreKickAdjustment(ActionNode):
        def start(self, event=None):
            super().start(event)
            print("[KICK] Kicking domino! Driving forward 70mm...")
            self.robot.actuators["drive"].forward(self, 70.0)

    class RetreatAdjustment(ActionNode):
        def start(self, event=None):
            super().start(event)
            dist = self.parent.last_charge_distance_mm
            print(f"[RETREAT] Initiating retreat. Logged distance: {dist:.1f}mm")
            
            if dist <= 0.0:
                print("[RETREAT] Warning: Distance <= 0. Proceeding directly to fallen sequence...")
                self.post_completion()
                return

            print(f"[RETREAT] Moving backward by {-dist:.1f}mm...")
            self.robot.actuators["drive"].forward(self, -dist, None)

    # ==========================================
    #  STATE NODES - FALLEN PICKUP PHASE
    # ==========================================
    class ResetFallenRun(StateNode):
        def start(self, event=None):
            super().start(event)
            print("\n==============================================")
            print(">>> [FALLEN-PIPELINE] SWITCHING TO FALLEN.PT SCAN <<<")
            print("==============================================")
            self.parent.fallen_target = None
            self.parent.prior_center_px = PRIOR_CENTER_PX
            self.parent.last_cmd = ""
            self.parent.fallen_turn_steps = 0
            self.parent.fallen_side_steps = 0
            self.parent.fallen_forward_steps = 0
            self.parent.motion_plan = []
            self.parent.motion_index = 0
            self.parent.final_touch_step = None
            self.parent.redetect_mode = False
            
            print("[FALLEN-PIPELINE] Settling camera for 1.5s...")
            time.sleep(1.5)
            self.post_completion()

    class DetectFallenTarget(StateNode):
        def start(self, event=None):
            super().start(event)
            print("[FALLEN-DETECT] Running segmentation model on camera frame...")
            if self.parent.fallen_model is None:
                print("[FALLEN-DETECT] ERROR: fallen.pt model not loaded!")
                self.post_failure()
                return

            settle = REDETECT_SETTLE_DELAY_SEC if self.parent.redetect_mode else SETTLE_DELAY_SEC
            if settle > 0: time.sleep(settle)

            image = None
            results = []
            tries = REDETECT_TRIES if self.parent.redetect_mode else DETECT_TRIES
            retry_delay = REDETECT_RETRY_DELAY_SEC if self.parent.redetect_mode else DETECT_RETRY_DELAY_SEC

            for attempt in range(tries):
                image = self.robot.camera_image
                if image is None:
                    if attempt < (tries - 1): time.sleep(retry_delay)
                    continue

                canvas, scale, pad_x, pad_y, new_w, new_h = roboflow_fit_resize(image, MODEL_SIZE)
                conf_th = REDETECT_PREDICT_CONF if self.parent.redetect_mode else PREDICT_CONF
                try:
                    results = self.parent.fallen_model.predict(source=canvas, conf=conf_th, iou=PREDICT_IOU, verbose=False)
                except Exception as ex:
                    print(f"[FALLEN-DETECT] Exception during predict: {ex}")
                    results = []

                if len(results) > 0 and results[0].masks is not None and len(results[0].masks.data) > 0:
                    print(f"[FALLEN-DETECT] Valid mask found on attempt {attempt+1}/{tries}")
                    break
                
                if attempt < (tries - 1): time.sleep(retry_delay)

            if image is None or len(results) == 0:
                print("[FALLEN-DETECT] FAILED: Camera frame empty or no model output.")
                self.parent.fallen_target = None
                self.post_failure()
                return

            h, w = image.shape[:2]
            table_min_y_ratio = REDETECT_TABLE_MIN_Y_RATIO if self.parent.redetect_mode else TABLE_MIN_Y_RATIO

            if self.parent.prior_center_px is None:
                rx1, ry1, rx2, ry2 = 0, int(table_min_y_ratio * h), w - 1, h - 1
            else:
                pcx, pcy = self.parent.prior_center_px
                roi_hw = REDETECT_PRIOR_ROI_HALF_W if self.parent.redetect_mode else PRIOR_ROI_HALF_W
                roi_hh = REDETECT_PRIOR_ROI_HALF_H if self.parent.redetect_mode else PRIOR_ROI_HALF_H
                rx1, ry1 = max(0, int(round(pcx - roi_hw))), max(int(table_min_y_ratio * h), int(round(pcy - roi_hh)))
                rx2, ry2 = min(w - 1, int(round(pcx + roi_hw))), min(h - 1, int(round(pcy + roi_hh)))

            result = results[0]
            if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
                print("[FALLEN-DETECT] FAILED: Predict returned 0 masks.")
                self.parent.fallen_target = None
                self.post_failure()
                return

            mask_data = result.masks.data
            confs = result.boxes.conf.detach().cpu().numpy() if hasattr(result, "boxes") and result.boxes is not None else []
            roi_diag = math.hypot(max(rx2 - rx1, 1), max(ry2 - ry1, 1))
            prior = self.parent.prior_center_px
            candidates = []

            def overlaps_roi(x1, y1, x2, y2):
                return (min(float(x2), float(rx2)) > max(float(x1), float(rx1))) and (min(float(y2), float(ry2)) > max(float(y1), float(ry1)))

            for i in range(len(mask_data)):
                mask_small = mask_data[i].detach().cpu().numpy()
                mask_orig = map_mask_to_original(mask_small, pad_x, pad_y, new_w, new_h, w, h)
                mask_area = float(cv2.countNonZero(mask_orig))
                max_area = REDETECT_MAX_MASK_AREA_PX if self.parent.redetect_mode else MAX_MASK_AREA_PX
                if mask_area < MIN_MASK_AREA_PX or mask_area > max_area: continue

                contours, _ = cv2.findContours(mask_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours: continue

                cnt = max(contours, key=cv2.contourArea)
                if float(cv2.contourArea(cnt)) < MIN_MASK_AREA_PX: continue

                rect = cv2.minAreaRect(cnt)
                (cx, cy), (rw, rh), _ = rect
                min_side = REDETECT_MIN_SIDE_PX if self.parent.redetect_mode else MIN_SIDE_PX
                if min(rw, rh) < min_side: continue

                aspect = max(rw, rh) / max(min(rw, rh), 1.0)
                aspect_max = REDETECT_ASPECT_MAX if self.parent.redetect_mode else ASPECT_MAX
                aspect_min = REDETECT_ASPECT_MIN if self.parent.redetect_mode else ASPECT_MIN
                if aspect < aspect_min or aspect > aspect_max: continue

                x, y, bw, bh = cv2.boundingRect(cnt)
                if cy < table_min_y_ratio * h or not overlaps_roi(x, y, x + bw, y + bh): continue

                det_conf = float(confs[i]) if i < len(confs) else 0.50
                prior_score = 0.50 if prior is None else clamp(1.0 - (math.hypot(cx - prior[0], cy - prior[1]) / max(roi_diag, 1.0)), 0.0, 1.0)
                conf_floor = REDETECT_PREDICT_CONF if self.parent.redetect_mode else PREDICT_CONF
                det_score = clamp((det_conf - conf_floor) / max(1.0 - conf_floor, 1e-6), 0.0, 1.0)
                shape_score = clamp(1.0 - (abs(aspect - 2.0) / 2.5), 0.0, 1.0)

                score = (W_PRIOR * prior_score) + (W_DETECT * det_score) + (W_SHAPE * shape_score)
                candidates.append({"score": score, "cx": cx, "cy": cy, "rect": rect, "det_conf": det_conf, "mask_area": mask_area, "aspect": aspect})

            # FALLBACK IF REGULAR FILTERS FAIL
            if len(candidates) == 0:
                print("[FALLEN-DETECT] Filter rejected masks. Using fallback to largest blob...")
                best_fallback = None
                for i in range(len(mask_data)):
                    mask_small = mask_data[i].detach().cpu().numpy()
                    mask_orig = map_mask_to_original(mask_small, pad_x, pad_y, new_w, new_h, w, h)
                    mask_area = float(cv2.countNonZero(mask_orig))
                    if mask_area < MIN_MASK_AREA_PX: continue
                    contours, _ = cv2.findContours(mask_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours: continue
                    cnt = max(contours, key=cv2.contourArea)
                    rect = cv2.minAreaRect(cnt)
                    (cx, cy), _, _ = rect
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    if cy < table_min_y_ratio * h or not overlaps_roi(x, y, x + bw, y + bh): continue
                    if best_fallback is None or mask_area > best_fallback["mask_area"]:
                        best_fallback = {"score": float(mask_area), "cx": cx, "cy": cy, "rect": rect, "det_conf": 0.50, "mask_area": mask_area, "aspect": 0.0}
                if best_fallback is not None:
                    candidates.append(best_fallback)

            if len(candidates) == 0:
                print("[FALLEN-DETECT] FAILED: No valid candidate found.")
                self.parent.fallen_target = None
                self.post_failure()
                return

            best = max(candidates, key=lambda c: c["score"])
            face_a_img, face_b_img, long_side = long_axis_from_rect(best["rect"])

            try:
                center_x_mm, center_y_mm = project_pixel_to_base(self.robot, best["cx"], best["cy"])
                face_a_x_mm, face_a_y_mm = project_pixel_to_base(self.robot, face_a_img[0], face_a_img[1])
                face_b_x_mm, face_b_y_mm = project_pixel_to_base(self.robot, face_b_img[0], face_b_img[1])
            except Exception as proj_err:
                print(f"[FALLEN-DETECT] Projection error: {proj_err}")
                self.parent.fallen_target = None
                self.post_failure()
                return

            if math.hypot(face_a_x_mm, face_a_y_mm) <= math.hypot(face_b_x_mm, face_b_y_mm):
                pickup_x_mm, pickup_y_mm = face_a_x_mm, face_a_y_mm
            else:
                pickup_x_mm, pickup_y_mm = face_b_x_mm, face_b_y_mm

            inward_dx, inward_dy = float(center_x_mm - pickup_x_mm), float(center_y_mm - pickup_y_mm)
            inward_norm = max(math.hypot(inward_dx, inward_dy), 1e-6)
            inward_ux, inward_uy = inward_dx / inward_norm, inward_dy / inward_norm

            desired_heading_deg = wrap_deg(math.degrees(math.atan2(inward_uy, inward_ux)))
            pre_goal_x_mm = float(pickup_x_mm - ((PICKUP_FRONT_OFFSET_MM + FINAL_STICK_MM) * inward_ux))
            pre_goal_y_mm = float(pickup_y_mm - ((PICKUP_FRONT_OFFSET_MM + FINAL_STICK_MM) * inward_uy))

            target = {
                "cx_px": float(best["cx"]), "cy_px": float(best["cy"]),
                "pickup_x_mm": float(pickup_x_mm), "pickup_y_mm": float(pickup_y_mm),
                "goal_heading_deg": float(desired_heading_deg),
                "pre_goal_x_mm": float(pre_goal_x_mm), "pre_goal_y_mm": float(pre_goal_y_mm)
            }

            print(f"[FALLEN-DETECT] TARGET CONFIRMED! Target Pick Loc: ({pickup_x_mm:.1f}, {pickup_y_mm:.1f})mm | Heading: {desired_heading_deg:.1f}°")
            self.parent.fallen_target = target
            self.parent.prior_center_px = (target["cx_px"], target["cy_px"])
            self.post_data(target)

    class BuildFallenMotionPlan(StateNode):
        def start(self, event=None):
            super().start(event)
            target = event.data if isinstance(event, DataEvent) else self.parent.fallen_target
            if target is None:
                self.post_failure()
                return

            plan = []
            side_moves = split_motion(float(target["pre_goal_y_mm"]), MAX_SIDE_MM, MIN_SIDE_MM)
            forward_moves = split_motion(float(target["pre_goal_x_mm"]), MAX_FORWARD_MM, MIN_FORWARD_MM)
            turn_moves = split_motion(float(target["goal_heading_deg"]), MAX_TURN_DEG, MIN_TURN_DEG)

            if len(side_moves) > MAX_SIDE_STEPS or len(forward_moves) > MAX_FORWARD_STEPS or len(turn_moves) > MAX_TURN_STEPS:
                print("[FALLEN-PLAN] Motion steps exceed configured limits!")
                self.post_failure()
                return

            for mm in side_moves: plan.append({"cmd": "sideways", "distance_mm": float(mm)})
            for mm in forward_moves: plan.append({"cmd": "forward", "distance_mm": float(mm)})
            for deg in turn_moves: plan.append({"cmd": "turn", "angle_deg": float(deg)})

            self.parent.motion_plan = plan
            self.parent.motion_index = 0
            print(f"[FALLEN-PLAN] Planned {len(plan)} motion steps successfully.")
            self.post_completion()

    class DispatchFallenMotion(StateNode):
        def start(self, event=None):
            super().start(event)
            if self.parent.motion_index >= len(self.parent.motion_plan):
                print("[FALLEN-DISPATCH] All coarse motion steps executed!")
                self.post_success()
                return
            step = self.parent.motion_plan[self.parent.motion_index]
            print(f"[FALLEN-DISPATCH] Executing step {self.parent.motion_index+1}/{len(self.parent.motion_plan)}: {step}")
            self.post_data(step)

    class ExecuteFallenMotion(ActionNode):
        def start(self, event=None):
            super().start(event)
            if not isinstance(event, DataEvent) or not isinstance(event.data, dict):
                self.post_failure()
                return
            cmd = event.data.get("cmd", "")
            if cmd == "turn":
                self.parent.last_cmd = "turn"
                deg = float(event.data.get("angle_deg", 0.0))
                self.robot.actuators["drive"].turn(self, math.radians(deg), None)
            elif cmd == "sideways":
                self.parent.last_cmd = "sideways"
                mm = float(event.data.get("distance_mm", 0.0))
                self.robot.actuators["drive"].sideways(self, mm, None)
            elif cmd == "forward":
                self.parent.last_cmd = "forward"
                mm = float(event.data.get("distance_mm", 0.0))
                self.robot.actuators["drive"].forward(self, mm, None)
            else:
                self.post_failure()

    class AdvanceFallenPlan(StateNode):
        def start(self, event=None):
            super().start(event)
            self.parent.motion_index += 1
            self.post_completion()

    class BackupForRedetect(ActionNode):
        def start(self, event=None):
            super().start(event)
            if REDETECT_BACKUP_MM <= 0:
                self.post_completion()
                return
            print(f"[FALLEN-BACKUP] Backing up {REDETECT_BACKUP_MM}mm for fine re-detection...")
            self.robot.actuators["drive"].forward(self, -float(REDETECT_BACKUP_MM), None)

    class PrepareRedetect(StateNode):
        def start(self, event=None):
            super().start(event)
            self.parent.redetect_mode = True
            self.post_completion()

    class ComputeFinalTouch(StateNode):
        def start(self, event=None):
            super().start(event)
            self.parent.redetect_mode = False
            target = event.data if isinstance(event, DataEvent) else self.parent.fallen_target
            if target is None:
                step = {"cmd": "forward", "distance_mm": float(FINAL_STICK_MM)}
                self.parent.final_touch_step = step
                self.post_failure()
                return

            forward_to_contact = clamp(float(target["pickup_x_mm"]) - float(PICKUP_FRONT_OFFSET_MM) - float(CONTACT_TOL_MM), 0.0, FINAL_TOUCH_MAX_MM)
            cmd_mm = clamp(forward_to_contact + float(FINAL_STICK_MM), 0.0, FINAL_TOUCH_MAX_MM)
            step = {"cmd": "forward", "distance_mm": float(cmd_mm)}
            self.parent.final_touch_step = step
            print(f"[FALLEN-TOUCH] Final touch charge: Driving forward {cmd_mm:.1f}mm!")
            self.post_data(step)

    # ==========================================
    #  FSM PIPELINE SETUP
    # ==========================================
    def setup(self):
        begin = StateNode().set_name("begin").set_parent(self)
        reset = self.ResetRun().set_name("reset").set_parent(self)
        track = self.TrackAndIdentify().set_name("track").set_parent(self)
        analyze = self.CalculateTrajectory().set_name("analyze").set_parent(self)
        move = self.ActuateMotion().set_name("move").set_parent(self)

        approach_chk = self.ApproachCheckpoint().set_name("approach_chk").set_parent(self)
        scan_near = self.ScanNearZoneDrift().set_name("scan_near").set_parent(self)

        turn_parallel = self.TurnParallelNear().set_name("turn_parallel").set_parent(self)
        rescan_lateral = self.RescanLateralOffset().set_name("rescan_lateral").set_parent(self)
        shift_sideways = self.ShiftSidewaysNear().set_name("shift_sideways").set_parent(self)

        nudge = self.PreKickAdjustment().set_name("nudge").set_parent(self)
        wait_step = StateNode().set_name("wait_step").set_parent(self)
        retreat = self.RetreatAdjustment().set_name("retreat").set_parent(self)
        wait_scan = StateNode().set_name("wait_scan").set_parent(self)

        # --- FALLEN PICKUP NODES ---
        reset_fallen = self.ResetFallenRun().set_name("reset_fallen").set_parent(self)
        detect_fallen = self.DetectFallenTarget().set_name("detect_fallen").set_parent(self)
        build_fallen = self.BuildFallenMotionPlan().set_name("build_fallen").set_parent(self)
        dispatch_fallen = self.DispatchFallenMotion().set_name("dispatch_fallen").set_parent(self)
        exec_fallen = self.ExecuteFallenMotion().set_name("exec_fallen").set_parent(self)
        advance_fallen = self.AdvanceFallenPlan().set_name("advance_fallen").set_parent(self)

        backup_fallen = self.BackupForRedetect().set_name("backup_fallen").set_parent(self)
        prep_redetect = self.PrepareRedetect().set_name("prep_redetect").set_parent(self)
        redetect_fallen = self.DetectFallenTarget().set_name("redetect_fallen").set_parent(self)
        compute_touch = self.ComputeFinalTouch().set_name("compute_touch").set_parent(self)
        touch_exec = self.ExecuteFallenMotion().set_name("touch_exec").set_parent(self)

        wait_fallen_retry = StateNode().set_name("wait_fallen_retry").set_parent(self)
        done = Print(">>> FULL PIPELINE COMPLETED SUCCESSFULLY! <<<").set_name("done").set_parent(self)

        # --- TRANSITIONS (KICK PHASE) ---
        TimerTrans(0.5).add_sources(begin).add_destinations(reset)
        CompletionTrans().add_sources(reset).add_destinations(track)

        DataTrans().add_sources(track).add_destinations(analyze)
        FailureTrans().add_sources(track).add_destinations(wait_scan)
        TimerTrans(1.0).add_sources(wait_scan).add_destinations(track)

        DataTrans().add_sources(analyze).add_destinations(move)
        SuccessTrans().add_sources(analyze).add_destinations(approach_chk)
        FailureTrans().add_sources(analyze).add_destinations(wait_scan)

        CompletionTrans().add_sources(move).add_destinations(track)
        FailureTrans().add_sources(move).add_destinations(wait_scan)

        CompletionTrans().add_sources(approach_chk).add_destinations(scan_near)
        FailureTrans().add_sources(approach_chk).add_destinations(wait_scan)

        DataTrans().add_sources(scan_near).add_destinations(turn_parallel)
        SuccessTrans().add_sources(scan_near).add_destinations(nudge)
        FailureTrans().add_sources(scan_near).add_destinations(wait_scan)

        CompletionTrans().add_sources(turn_parallel).add_destinations(rescan_lateral)
        FailureTrans().add_sources(turn_parallel).add_destinations(wait_scan)

        CompletionTrans().add_sources(rescan_lateral).add_destinations(shift_sideways)
        FailureTrans().add_sources(rescan_lateral).add_destinations(wait_scan)

        CompletionTrans().add_sources(shift_sideways).add_destinations(nudge)
        FailureTrans().add_sources(shift_sideways).add_destinations(wait_scan)

        CompletionTrans().add_sources(nudge).add_destinations(wait_step)
        FailureTrans().add_sources(nudge).add_destinations(wait_scan)

        TimerTrans(0.1).add_sources(wait_step).add_destinations(retreat)

        # --- RETREAT -> FALLEN PIPELINE SHIFT (FIXED TRANSITIONS) ---
        # Added CompletionTrans, SuccessTrans, and FailureTrans to guarantee entry into reset_fallen
        CompletionTrans().add_sources(retreat).add_destinations(reset_fallen)
        SuccessTrans().add_sources(retreat).add_destinations(reset_fallen)
        FailureTrans().add_sources(retreat).add_destinations(reset_fallen)

        # --- TRANSITIONS (FALLEN PICKUP PHASE) ---
        CompletionTrans().add_sources(reset_fallen).add_destinations(detect_fallen)

        DataTrans().add_sources(detect_fallen).add_destinations(build_fallen)
        # Failure loops back with 1s delay to retry detection
        FailureTrans().add_sources(detect_fallen).add_destinations(wait_fallen_retry)
        TimerTrans(1.0).add_sources(wait_fallen_retry).add_destinations(detect_fallen)

        CompletionTrans().add_sources(build_fallen).add_destinations(dispatch_fallen)
        SuccessTrans().add_sources(build_fallen).add_destinations(backup_fallen)
        FailureTrans().add_sources(build_fallen).add_destinations(done)

        DataTrans().add_sources(dispatch_fallen).add_destinations(exec_fallen)
        SuccessTrans().add_sources(dispatch_fallen).add_destinations(backup_fallen)
        FailureTrans().add_sources(dispatch_fallen).add_destinations(done)

        CompletionTrans().add_sources(exec_fallen).add_destinations(advance_fallen)
        FailureTrans().add_sources(exec_fallen).add_destinations(done)

        CompletionTrans().add_sources(advance_fallen).add_destinations(dispatch_fallen)

        CompletionTrans().add_sources(backup_fallen).add_destinations(prep_redetect)
        CompletionTrans().add_sources(prep_redetect).add_destinations(redetect_fallen)

        DataTrans().add_sources(redetect_fallen).add_destinations(compute_touch)
        FailureTrans().add_sources(redetect_fallen).add_destinations(compute_touch)

        DataTrans().add_sources(compute_touch).add_destinations(touch_exec)
        FailureTrans().add_sources(compute_touch).add_destinations(touch_exec)

        CompletionTrans().add_sources(touch_exec).add_destinations(done)

        return self