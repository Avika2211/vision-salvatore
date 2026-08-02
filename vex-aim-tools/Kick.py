import os
import cv2
import math
import time
import numpy as np
from aim_fsm import *
from ultralytics import YOLO

# Import standing kick pipeline setup
from Kick import (
    Kick,
    SoftKick,
    SEG_MODEL_PATH,
    estimate_divider_x,
    estimate_angle_and_distance_cm,
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Fallen Model Weights
SEG_WEIGHTS_PRIMARY = os.path.join(PROJECT_DIR, "fallen.pt")
SEG_WEIGHTS_FALLBACK = os.path.join(
    PROJECT_DIR, "runs_play_dominos", "play_dominos_seg_v1_img160", "weights", "bestieee.pt"
)

# Fallen Detection Parameters (Relaxed to avoid dropping valid masks)
SETTLE_DELAY_SEC = 0.20
MODEL_SIZE = 160
PREDICT_CONF = 0.10
PREDICT_IOU = 0.45
DETECT_TRIES = 5
DETECT_RETRY_DELAY_SEC = 0.15

TABLE_MIN_Y_RATIO = 0.05
PRIOR_CENTER_PX = None
PRIOR_ROI_HALF_W = 320
PRIOR_ROI_HALF_H = 240

REDETECT_TABLE_MIN_Y_RATIO = 0.02
REDETECT_PREDICT_CONF = 0.08
REDETECT_PRIOR_ROI_HALF_W = 340
REDETECT_PRIOR_ROI_HALF_H = 240
REDETECT_TRIES = 5
REDETECT_RETRY_DELAY_SEC = 0.12
REDETECT_SETTLE_DELAY_SEC = 0.30

MIN_MASK_AREA_PX = 30.0
MAX_MASK_AREA_PX = 250000.0
MIN_SIDE_PX = 4.0
ASPECT_MIN = 0.80
ASPECT_MAX = 15.00

REDETECT_MAX_MASK_AREA_PX = 300000.0
REDETECT_ASPECT_MAX = 15.00
REDETECT_ASPECT_MIN = 0.80
REDETECT_MIN_SIDE_PX = 4.0

W_PRIOR = 0.35
W_DETECT = 0.45
W_SHAPE = 0.20

PICKUP_FRONT_OFFSET_MM = 30.0
FINAL_STICK_MM = 10.0
CONTACT_TOL_MM = 2.0

MAX_TURN_STEPS = 6
MAX_SIDE_STEPS = 6
MAX_FORWARD_STEPS = 8

MAX_TURN_DEG = 18.0
MIN_TURN_DEG = 3.0
MAX_SIDE_MM = 35.0
MIN_SIDE_MM = 8.0
MAX_FORWARD_MM = 45.0
MIN_FORWARD_MM = 8.0

TURN_COOLDOWN_SEC = 1.0
REDETECT_BACKUP_MM = 20.0
FINAL_TOUCH_MAX_MM = 90.0

# Ground Projection Calibration
K1 = 1.55
K2 = -58.4
PICKUP_X_BIAS_MM = 30.0


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
    restored = cv2.resize(
        cropped.astype(np.uint8),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    )
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


class kickpickrefine(StateMachineProgram):
    def __init__(self):
        super().__init__()

        # --- Standing Kick Setup ---
        print("Initializing Combined Pipeline (Kick -> Kinematic Fallen Pickup)...")
        self.segmenter = YOLO(SEG_MODEL_PATH)
        self.classifier = self._load_classifier("different.pt")
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.target_data = None
        self.last_charge_distance_mm = 0.0

        # --- Fallen Model Setup ---
        self.model_path = None
        if os.path.exists(SEG_WEIGHTS_PRIMARY):
            self.model_path = SEG_WEIGHTS_PRIMARY
        elif os.path.exists(SEG_WEIGHTS_FALLBACK):
            self.model_path = SEG_WEIGHTS_FALLBACK

        self.model = None
        if self.model_path is not None:
            print(f"Loading fallen segmentation weights: {self.model_path}")
            self.model = YOLO(self.model_path)
            self.model.eval()
        else:
            print("Warning: no fallen segmentation weights found.")

        self.target = None
        self.debug_view = None
        self.model_view = None
        self.prior_center_px = PRIOR_CENTER_PX
        self.last_cmd = ""
        self.turn_steps = 0
        self.side_steps = 0
        self.forward_steps = 0
        self.motion_plan = []
        self.motion_index = 0
        self.final_touch_step = None
        self.redetect_mode = False

    def _load_classifier(self, path):
        return Kick._load_classifier(self, path)

    def rotate_crop(self, image, obb):
        return Kick.rotate_crop(self, image, obb)

    def split_halves(self, image):
        return Kick.split_halves(self, image)

    def remove_white_border(self, image, threshold=250, margin=4):
        return Kick.remove_white_border(self, image, threshold, margin)

    def preprocess_half(self, half_bgr):
        return Kick.preprocess_half(self, half_bgr)

    def predict_half(self, tensor):
        return Kick.predict_half(self, tensor)

    def measure_lateral_offset_mm(self, frame, conf=0.15):
        return Kick.measure_lateral_offset_mm(self, frame, conf=conf)

    # Kick States Inherited
    ResetRun = Kick.ResetRun
    TrackAndIdentify = Kick.TrackAndIdentify
    CalculateTrajectory = Kick.CalculateTrajectory
    ActuateMotion = Kick.ActuateMotion
    ApproachCheckpoint = Kick.ApproachCheckpoint
    ScanNearZoneDrift = Kick.ScanNearZoneDrift
    TurnParallelNear = Kick.TurnParallelNear
    RescanLateralOffset = Kick.RescanLateralOffset
    ShiftSidewaysNear = Kick.ShiftSidewaysNear
    PreKickAdjustment = Kick.PreKickAdjustment

    class RetreatAdjustment(ActionNode):
        def start(self, event=None):
            super().start(event)
            distance_to_retreat = self.parent.last_charge_distance_mm
            print(f"\n[KICK-RETREAT] Retreating back {-distance_to_retreat:.1f}mm...")
            if distance_to_retreat <= 0.0:
                print("[KICK-RETREAT] Distance <= 0, skipping motion.")
                self.post_completion()
                return

            self.robot.actuators["drive"].forward(self, -distance_to_retreat, None)

    class PreparePickupRun(StateNode):
        def start(self, event=None):
            super().start(event)
            print("\n==============================================")
            print(">>> [SWITCHING TO FALLEN.PT PIPELINE] <<<")
            print("==============================================")
            self.parent.target = None
            self.parent.debug_view = None
            self.parent.model_view = None
            self.parent.prior_center_px = PRIOR_CENTER_PX
            self.parent.last_cmd = ""
            self.parent.turn_steps = 0
            self.parent.side_steps = 0
            self.parent.forward_steps = 0
            self.parent.motion_plan = []
            self.parent.motion_index = 0
            self.parent.final_touch_step = None
            self.parent.redetect_mode = False
            self.post_completion()

    class DetectTarget(StateNode):
        def start(self, event=None):
            super().start(event)
            print("[FALLEN-DETECT] Scanning frame using fallen.pt...")
            if self.parent.model is None:
                print("[FALLEN-DETECT] ERROR: Fallen model is missing/None!")
                self.parent.target = None
                self.parent.debug_view = None
                self.parent.model_view = None
                self.post_failure()
                return

            settle = REDETECT_SETTLE_DELAY_SEC if self.parent.redetect_mode else SETTLE_DELAY_SEC
            if settle > 0:
                time.sleep(settle)

            image = None
            results = []
            tries = REDETECT_TRIES if self.parent.redetect_mode else DETECT_TRIES
            retry_delay = REDETECT_RETRY_DELAY_SEC if self.parent.redetect_mode else DETECT_RETRY_DELAY_SEC

            for attempt in range(tries):
                image = self.robot.camera_image
                if image is None:
                    if attempt < (tries - 1):
                        time.sleep(retry_delay)
                    continue

                canvas, scale, pad_x, pad_y, new_w, new_h = roboflow_fit_resize(image, MODEL_SIZE)
                conf_th = REDETECT_PREDICT_CONF if self.parent.redetect_mode else PREDICT_CONF
                try:
                    results = self.parent.model.predict(
                        source=canvas,
                        conf=conf_th,
                        iou=PREDICT_IOU,
                        verbose=False,
                    )
                except Exception as ex:
                    print(f"[FALLEN-DETECT] Model predict failed: {ex}")
                    results = []

                if len(results) > 0 and results[0].masks is not None and len(results[0].masks.data) > 0:
                    print(f"[FALLEN-DETECT] Success! Found mask on attempt {attempt+1}/{tries}")
                    break

                if attempt < (tries - 1):
                    time.sleep(retry_delay)

            if image is None or len(results) == 0:
                print("[FALLEN-DETECT] FAILED: No detection/masks from fallen model.")
                self.parent.target = None
                self.parent.debug_view = None
                self.parent.model_view = None
                self.post_failure()
                return

            h, w = image.shape[:2]
            view = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
            table_min_y_ratio = REDETECT_TABLE_MIN_Y_RATIO if self.parent.redetect_mode else TABLE_MIN_Y_RATIO

            if self.parent.prior_center_px is None:
                rx1, ry1 = 0, int(table_min_y_ratio * h)
                rx2, ry2 = w - 1, h - 1
            else:
                pcx, pcy = self.parent.prior_center_px
                roi_hw = REDETECT_PRIOR_ROI_HALF_W if self.parent.redetect_mode else PRIOR_ROI_HALF_W
                roi_hh = REDETECT_PRIOR_ROI_HALF_H if self.parent.redetect_mode else PRIOR_ROI_HALF_H
                rx1 = max(0, int(round(pcx - roi_hw)))
                ry1 = max(int(table_min_y_ratio * h), int(round(pcy - roi_hh)))
                rx2 = min(w - 1, int(round(pcx + roi_hw)))
                ry2 = min(h - 1, int(round(pcy + roi_hh)))

            result = results[0]
            self.parent.model_view = result.plot(line_width=1, labels=False)

            if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
                print("[FALLEN-DETECT] No masks in inference output.")
                self.parent.target = None
                self.parent.debug_view = view
                self.post_failure()
                return

            mask_data = result.masks.data
            confs = []
            if result.boxes is not None and hasattr(result.boxes, "conf") and result.boxes.conf is not None:
                confs = result.boxes.conf.detach().cpu().numpy()

            roi_diag = math.hypot(max(rx2 - rx1, 1), max(ry2 - ry1, 1))
            prior = self.parent.prior_center_px
            candidates = []
            mask_layer = np.zeros_like(view)

            def overlaps_roi(x1, y1, x2, y2):
                return (min(float(x2), float(rx2)) > max(float(x1), float(rx1))) and (min(float(y2), float(ry2)) > max(float(y1), float(ry1)))

            for i in range(len(mask_data)):
                mask_small = mask_data[i].detach().cpu().numpy()
                mask_orig = map_mask_to_original(mask_small, pad_x, pad_y, new_w, new_h, w, h)
                mask_area = float(cv2.countNonZero(mask_orig))
                max_area = REDETECT_MAX_MASK_AREA_PX if self.parent.redetect_mode else MAX_MASK_AREA_PX
                if mask_area < MIN_MASK_AREA_PX or mask_area > max_area:
                    continue

                contours, _ = cv2.findContours(mask_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue

                cnt = max(contours, key=cv2.contourArea)
                if float(cv2.contourArea(cnt)) < MIN_MASK_AREA_PX:
                    continue

                rect = cv2.minAreaRect(cnt)
                (cx, cy), (rw, rh), _ = rect
                rw, rh = float(rw), float(rh)
                min_side = REDETECT_MIN_SIDE_PX if self.parent.redetect_mode else MIN_SIDE_PX
                if min(rw, rh) < min_side:
                    continue

                aspect = max(rw, rh) / max(min(rw, rh), 1.0)
                aspect_max = REDETECT_ASPECT_MAX if self.parent.redetect_mode else ASPECT_MAX
                aspect_min = REDETECT_ASPECT_MIN if self.parent.redetect_mode else ASPECT_MIN
                if aspect < aspect_min or aspect > aspect_max:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)
                if cy < table_min_y_ratio * h or not overlaps_roi(x, y, x + bw, y + bh):
                    continue

                det_conf = float(confs[i]) if i < len(confs) else 0.50
                prior_score = 0.50 if prior is None else clamp(1.0 - (math.hypot(cx - prior[0], cy - prior[1]) / max(roi_diag, 1.0)), 0.0, 1.0)
                conf_floor = REDETECT_PREDICT_CONF if self.parent.redetect_mode else PREDICT_CONF
                det_score = clamp((det_conf - conf_floor) / max(1.0 - conf_floor, 1e-6), 0.0, 1.0)
                shape_score = clamp(1.0 - (abs(aspect - 2.0) / 2.5), 0.0, 1.0)
                score = (W_PRIOR * prior_score) + (W_DETECT * det_score) + (W_SHAPE * shape_score)

                mask_layer[mask_orig > 0] = (255, 0, 0)
                box = cv2.boxPoints(rect).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(view, [box], isClosed=True, color=(0, 165, 255), thickness=1)

                candidates.append({
                    "score": float(score),
                    "cx": float(cx),
                    "cy": float(cy),
                    "rect": rect,
                    "det_conf": float(det_conf),
                    "mask_area": float(mask_area),
                    "aspect": float(aspect),
                })

            # FALLBACK logic - Accept first detected mask directly if shape filtering dropped it
            if len(candidates) == 0:
                print("[FALLEN-DETECT] Primary filters dropped candidate. Bypassing strict filters...")
                for i in range(len(mask_data)):
                    mask_small = mask_data[i].detach().cpu().numpy()
                    mask_orig = map_mask_to_original(mask_small, pad_x, pad_y, new_w, new_h, w, h)
                    mask_area = float(cv2.countNonZero(mask_orig))
                    if mask_area < MIN_MASK_AREA_PX:
                        continue
                    contours, _ = cv2.findContours(mask_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue
                    cnt = max(contours, key=cv2.contourArea)
                    rect = cv2.minAreaRect(cnt)
                    (cx, cy), _, _ = rect

                    best_fallback = {
                        "score": float(mask_area),
                        "cx": float(cx),
                        "cy": float(cy),
                        "rect": rect,
                        "det_conf": float(confs[i]) if i < len(confs) else 0.50,
                        "mask_area": float(mask_area),
                        "aspect": 1.5,
                    }
                    candidates.append(best_fallback)
                    break

            if len(candidates) == 0:
                print("[FALLEN-DETECT] FAILED: Zero valid fallen domino masks found.")
                self.parent.target = None
                self.parent.debug_view = view
                self.post_failure()
                return

            best = max(candidates, key=lambda c: c["score"])
            face_a_img, face_b_img, long_side = long_axis_from_rect(best["rect"])

            try:
                center_x_mm, center_y_mm = project_pixel_to_base(self.robot, best["cx"], best["cy"])
                face_a_x_mm, face_a_y_mm = project_pixel_to_base(self.robot, face_a_img[0], face_a_img[1])
                face_b_x_mm, face_b_y_mm = project_pixel_to_base(self.robot, face_b_img[0], face_b_img[1])
            except Exception as proj_ex:
                print(f"[FALLEN-DETECT] Projection error: {proj_ex}")
                self.parent.target = None
                self.parent.debug_view = view
                self.post_failure()
                return

            if math.hypot(face_a_x_mm, face_a_y_mm) <= math.hypot(face_b_x_mm, face_b_y_mm):
                pickup_x_mm, pickup_y_mm = face_a_x_mm, face_a_y_mm
            else:
                pickup_x_mm, pickup_y_mm = face_b_x_mm, face_b_y_mm

            inward_dx = float(center_x_mm - pickup_x_mm)
            inward_dy = float(center_y_mm - pickup_y_mm)
            inward_norm = max(math.hypot(inward_dx, inward_dy), 1e-6)
            inward_ux, inward_uy = inward_dx / inward_norm, inward_dy / inward_norm

            desired_heading_deg = wrap_deg(math.degrees(math.atan2(inward_uy, inward_ux)))
            pre_goal_x_mm = float(pickup_x_mm - ((PICKUP_FRONT_OFFSET_MM + FINAL_STICK_MM) * inward_ux))
            pre_goal_y_mm = float(pickup_y_mm - ((PICKUP_FRONT_OFFSET_MM + FINAL_STICK_MM) * inward_uy))

            target = {
                "confidence": float(best["score"]),
                "cx_px": float(best["cx"]),
                "cy_px": float(best["cy"]),
                "pickup_x_mm": float(pickup_x_mm),
                "pickup_y_mm": float(pickup_y_mm),
                "goal_heading_deg": float(desired_heading_deg),
                "pre_goal_x_mm": float(pre_goal_x_mm),
                "pre_goal_y_mm": float(pre_goal_y_mm),
            }

            print(f"[FALLEN-TARGET FOUND] Pick Loc: ({pickup_x_mm:.1f}, {pickup_y_mm:.1f})mm | Heading: {desired_heading_deg:.1f}°")
            self.parent.target = target
            self.parent.debug_view = view
            self.parent.prior_center_px = (target["cx_px"], target["cy_px"])
            self.post_data(target)

    class BuildMotionPlan(StateNode):
        def start(self, event=None):
            super().start(event)
            target = event.data if isinstance(event, DataEvent) else self.parent.target
            if target is None:
                self.post_failure()
                return

            plan = []
            side_moves = split_motion(float(target["pre_goal_y_mm"]), MAX_SIDE_MM, MIN_SIDE_MM)
            forward_moves = split_motion(float(target["pre_goal_x_mm"]), MAX_FORWARD_MM, MIN_FORWARD_MM)
            turn_moves = split_motion(float(target["goal_heading_deg"]), MAX_TURN_DEG, MIN_TURN_DEG)

            if len(side_moves) > MAX_SIDE_STEPS or len(forward_moves) > MAX_FORWARD_STEPS or len(turn_moves) > MAX_TURN_STEPS:
                print("[Build Motion] Motion step limits exceeded!")
                self.post_failure()
                return

            for mm in side_moves: plan.append({"cmd": "sideways", "distance_mm": float(mm)})
            for mm in forward_moves: plan.append({"cmd": "forward", "distance_mm": float(mm)})
            for deg in turn_moves: plan.append({"cmd": "turn", "angle_deg": float(deg)})

            self.parent.motion_plan = plan
            self.parent.motion_index = 0
            print(f"[Build Motion] Planned {len(plan)} motion steps.")
            self.post_completion()

    class BackupForRedetect(ActionNode):
        def start(self, event=None):
            super().start(event)
            if REDETECT_BACKUP_MM <= 0:
                self.post_completion()
                return
            print(f"[Backup] Backing up {REDETECT_BACKUP_MM}mm for close re-detection...")
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
            target = event.data if isinstance(event, DataEvent) else self.parent.target
            if target is None:
                self.parent.final_touch_step = {"cmd": "forward", "distance_mm": float(FINAL_STICK_MM)}
                self.post_failure()
                return

            forward_to_contact = clamp(float(target["pickup_x_mm"]) - float(PICKUP_FRONT_OFFSET_MM) - float(CONTACT_TOL_MM), 0.0, FINAL_TOUCH_MAX_MM)
            cmd_mm = clamp(forward_to_contact + float(FINAL_STICK_MM), 0.0, FINAL_TOUCH_MAX_MM)
            step = {"cmd": "forward", "distance_mm": float(cmd_mm), "label": "final touch"}
            self.parent.final_touch_step = step
            print(f"[Final Touch] Advancing {cmd_mm:.1f}mm for pickup contact.")
            self.post_data(step)

    class ShowTarget(StateNode):
        def start(self, event=None):
            super().start(event)
            if self.parent.debug_view is not None:
                imshow("domino_fallen_pickup", self.parent.debug_view)
            if self.parent.model_view is not None:
                imshow("domino_fallen_pickup_model", self.parent.model_view)
            self.post_completion()

    class DispatchMotion(StateNode):
        def start(self, event=None):
            super().start(event)
            if self.parent.motion_index >= len(self.parent.motion_plan):
                self.post_success()
                return
            step = self.parent.motion_plan[self.parent.motion_index]
            print(f"[Dispatch Motion] Step {self.parent.motion_index+1}/{len(self.parent.motion_plan)}: {step}")
            self.post_data(step)

    class ExecuteMotion(ActionNode):
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

    class AdvancePlan(StateNode):
        def start(self, event=None):
            super().start(event)
            self.parent.motion_index += 1
            self.post_completion()

    class ShouldCooldownAfterTurn(StateNode):
        def start(self, event=None):
            super().start(event)
            if self.parent.last_cmd == "turn":
                self.post_success()
            else:
                self.post_failure()

    class IncrementStep(StateNode):
        def start(self, event=None):
            super().start(event)
            cmd = self.parent.last_cmd
            if cmd == "turn":
                self.parent.turn_steps += 1
            elif cmd == "sideways":
                self.parent.side_steps += 1
            elif cmd == "forward":
                self.parent.forward_steps += 1
            self.post_completion()

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
        strike = SoftKick().set_name("strike").set_parent(self)
        retreat = self.RetreatAdjustment().set_name("retreat").set_parent(self)
        
        # Settle Pause to prevent actuator locking
        wait_after_kick = StateNode().set_name("wait_after_kick").set_parent(self)

        # --- Fallen Pickup Strategy Nodes ---
        prep_pickup = self.PreparePickupRun().set_name("prep_pickup").set_parent(self)
        detect = self.DetectTarget().set_name("detect").set_parent(self)
        show = self.ShowTarget().set_name("show").set_parent(self)
        build = self.BuildMotionPlan().set_name("build").set_parent(self)
        dispatch = self.DispatchMotion().set_name("dispatch").set_parent(self)
        exec_node = self.ExecuteMotion().set_name("exec_node").set_parent(self)
        cooldown_check = self.ShouldCooldownAfterTurn().set_name("cooldown_check").set_parent(self)
        cooldown_wait = StateNode().set_name("cooldown_wait").set_parent(self)
        inc = self.IncrementStep().set_name("inc").set_parent(self)
        advance = self.AdvancePlan().set_name("advance").set_parent(self)

        backup = self.BackupForRedetect().set_name("backup").set_parent(self)
        prep_redetect = self.PrepareRedetect().set_name("prep_redetect").set_parent(self)
        redetect = self.DetectTarget().set_name("redetect").set_parent(self)
        show2 = self.ShowTarget().set_name("show2").set_parent(self)
        touch = self.ComputeFinalTouch().set_name("touch").set_parent(self)
        touch_fallback = Print("Re-detect failed. Doing minimal stick touch.").set_name("touch_fallback").set_parent(self)
        touch_exec = self.ExecuteMotion().set_name("touch_exec").set_parent(self)
        pickup_done = Print(">>> FULL PIPELINE FINISHED! <<<").set_name("pickup_done").set_parent(self)

        no_target = Print("No valid fallen domino pickup target. Restarting loop...").set_name("no_target").set_parent(self)
        wait_scan = StateNode().set_name("wait_scan").set_parent(self)

        # --- KICK TRANSITIONS ---
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

        TimerTrans(0.1).add_sources(wait_step).add_destinations(strike)

        CompletionTrans().add_sources(strike).add_destinations(retreat)
        FailureTrans().add_sources(strike).add_destinations(wait_scan)

        # --- GUARANTEED RETREAT -> FALLEN SHIFT PIPELINE ---
        CompletionTrans().add_sources(retreat).add_destinations(wait_after_kick)
        SuccessTrans().add_sources(retreat).add_destinations(wait_after_kick)
        FailureTrans().add_sources(retreat).add_destinations(wait_after_kick)

        TimerTrans(1.5).add_sources(wait_after_kick).add_destinations(prep_pickup)

        # --- PICKUP TRANSITIONS ---
        CompletionTrans().add_sources(prep_pickup).add_destinations(detect)

        DataTrans().add_sources(detect).add_destinations(show)
        FailureTrans().add_sources(detect).add_destinations(no_target)

        CompletionTrans().add_sources(show).add_destinations(build)
        FailureTrans().add_sources(show).add_destinations(no_target)

        CompletionTrans().add_sources(build).add_destinations(dispatch)
        SuccessTrans().add_sources(build).add_destinations(backup)
        FailureTrans().add_sources(build).add_destinations(no_target)

        DataTrans().add_sources(dispatch).add_destinations(exec_node)
        SuccessTrans().add_sources(dispatch).add_destinations(backup)
        FailureTrans().add_sources(dispatch).add_destinations(no_target)

        CompletionTrans().add_sources(exec_node).add_destinations(cooldown_check)
        FailureTrans().add_sources(exec_node).add_destinations(no_target)

        SuccessTrans().add_sources(cooldown_check).add_destinations(cooldown_wait)
        FailureTrans().add_sources(cooldown_check).add_destinations(inc)
        TimerTrans(TURN_COOLDOWN_SEC).add_sources(cooldown_wait).add_destinations(inc)

        CompletionTrans().add_sources(inc).add_destinations(advance)
        CompletionTrans().add_sources(advance).add_destinations(dispatch)

        CompletionTrans().add_sources(backup).add_destinations(prep_redetect)
        CompletionTrans().add_sources(prep_redetect).add_destinations(redetect)

        DataTrans().add_sources(redetect).add_destinations(show2)
        FailureTrans().add_sources(redetect).add_destinations(touch_fallback)

        CompletionTrans().add_sources(show2).add_destinations(touch)
        FailureTrans().add_sources(show2).add_destinations(touch_fallback)

        DataTrans().add_sources(touch).add_destinations(touch_exec)
        FailureTrans().add_sources(touch).add_destinations(touch_exec)

        NullTrans().add_sources(touch_fallback).add_destinations(touch_exec)

        CompletionTrans().add_sources(touch_exec).add_destinations(pickup_done)

        NullTrans().add_sources(pickup_done).add_destinations(reset)
        NullTrans().add_sources(no_target).add_destinations(reset)

        return self