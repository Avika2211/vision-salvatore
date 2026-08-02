from __future__ import annotations

import os
import sys

os.environ["QT_LOGGING_RULES"] = (
    "qt.core.qobject.connect=false;"
    "qt.qpa.*=false;"
    "qt.scenegraph.*=false;"
    "*.debug=false"
)

import importlib
import math
import cv2
import numpy as np

from aim_fsm import *
from aim_fsm.worldmap import DominoObj, Pose
from aim_fsm.domino import (
    DominoWorldDetector,
    DominoObservation,
    normalize_axis_angle,
)

import aim_fsm.domino
importlib.reload(aim_fsm.domino)

KNOWN_LENGTH_MM = 48.0
FOCAL_LENGTH = 396.3


class modelling(StateMachineProgram):
    def __init__(self):
        super().__init__(
            launch_cam_viewer=True,
            launch_worldmap_viewer=True,
            launch_path_viewer=False,
            speech=False,
            aruco=False,
            domino=False,
            domino_labeling=False,
            force_annotation=True,
        )
        
        # Instantiate detector with all dual models configured
        detector = DominoWorldDetector(
            conf_threshold=0.35,
            standing_weights="bestieee.pt",
            fallen_weights="fallen.pt",
            standing_label_weights="different.pt",
            fallen_label_weights="fallenhalf.pt",
        )

        for attr in ["focal_length", "focal_length_px", "fx", "fy"]:
            if hasattr(detector, attr):
                setattr(detector, attr, FOCAL_LENGTH)

        self.robot.domino_detector = detector
        self._last_logged_objs = set()
        self._last_image = None

    def user_image(self, image, gray):
        detector = getattr(self.robot, "domino_detector", None)
        if detector is None or image is None:
            return

        self._last_image = image.copy()

        try:
            observations = detector.detect(image, frame_id=getattr(self.robot, "frame_count", None))

            # Fix: Use full RGB to BGR color conversion instead of converting to grayscale
            vis_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if len(image.shape) == 3 else image.copy()
            overlay = vis_frame.copy()

            for idx, obs in enumerate(observations):
                mask_pts = None

                if hasattr(obs, "quad") and obs.quad is not None and len(obs.quad) == 4:
                    mask_pts = np.array(obs.quad, dtype=np.int32).reshape((-1, 1, 2))

                # Green overlay for standing, Cyan/Yellow overlay for fallen
                poly_color = (255, 200, 0) if obs.is_fallen else (0, 230, 110)
                border_color = (255, 255, 0) if obs.is_fallen else (0, 255, 120)

                if mask_pts is not None:
                    cv2.fillPoly(overlay, [mask_pts], color=poly_color)
                    cv2.polylines(vis_frame, [mask_pts], isClosed=True, color=border_color, thickness=2, lineType=cv2.LINE_AA)

                cx, cy = obs.center_xy
                status_str = "FALLEN" if obs.is_fallen else "STANDING"
                label = f"Domino.{chr(ord('a') + idx)} ({status_str})"
                if obs.face_label:
                    label += f" [{obs.face_label}]"
                
                dist_str = f"{obs.distance_cm:.1f}cm"

                cv2.circle(vis_frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                cv2.putText(vis_frame, label, (int(cx) - 50, int(cy) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(vis_frame, dist_str, (int(cx) - 20, int(cy) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.addWeighted(overlay, 0.30, vis_frame, 0.70, 0, vis_frame)
            cv2.imshow("Live Domino Detector", vis_frame)
            cv2.waitKey(1)

        except Exception:
            pass

    def update_3d_snapshot(self):
        detector = getattr(self.robot, "domino_detector", None)
        image = getattr(self, "_last_image", None)
        world_map = getattr(self.robot, "world_map", None)

        if detector is None or image is None or world_map is None:
            print("[SNAPSHOT] Warning: Detector or Image Frame not available for snapshot.")
            return

        try:
            observations = detector.detect(image, frame_id=getattr(self.robot, "frame_count", None))

            world_map.objects.clear()
            
            if not observations:
                return

            for idx, obs in enumerate(observations):
                quad = np.array(obs.quad, dtype=np.float32) if (hasattr(obs, 'quad') and obs.quad is not None) else None
                cx, cy = obs.center_xy
                
                bottom_cy = float(np.max(quad[:, 1])) if quad is not None else cy
                hit, objpos = world_map.project_image_point_to_world(cx, bottom_cy)
                
                x_mm = float(objpos[0][0])
                y_mm = float(objpos[1][0])

                sorted_by_y = sorted(quad, key=lambda pt: pt[1], reverse=True)
                p_base1, p_base2 = sorted_by_y[0], sorted_by_y[1]
                if p_base1[0] > p_base2[0]:
                    p_base1, p_base2 = p_base2, p_base1

                hit1, _ = world_map.project_image_point_to_world(p_base1[0], p_base1[1])
                hit2, _ = world_map.project_image_point_to_world(p_base2[0], p_base2[1])

                dx_world = float(hit2[0][0] - hit1[0][0])
                dy_world = float(hit2[1][0] - hit1[1][0])
                
                local_yaw = math.atan2(dy_world, dx_world)
                world_yaw = normalize_axis_angle(self.robot.pose.theta + local_yaw)

                face_label = str(obs.face_label) if obs.face_label else "0-0"
                first_half, second_half = 0, 0
                if "-" in face_label:
                    try:
                        parts = face_label.split("-")
                        first_half, second_half = int(parts[0]), int(parts[1])
                    except ValueError:
                        pass
                elif getattr(obs, "half_counts", None):
                    first_half, second_half = int(obs.half_counts[0]), int(obs.half_counts[1])

                halves = [
                    {"count": first_half, "local_y": -12.0, "local_x": 0.0},
                    {"count": second_half, "local_y": 12.0, "local_x": 0.0}
                ]

                obj_id = f"domino_{face_label}"
                dup = 1
                while obj_id in world_map.objects:
                    obj_id = f"domino_{face_label}_{dup}"
                    dup += 1

                # Dynamic 3D Dimensions based on status
                if obs.is_fallen:
                    # Fallen flat on ground: Height = 8mm, Thickness = 8mm, Width = 24mm, Z = 4mm
                    z_mm = 4.0
                    height_3d = 8.0
                    width_3d = 24.0
                    thickness_3d = 8.0
                else:
                    # Upright Standing: Height = 24mm, Width = 24mm, Thickness = 8mm, Z = 12mm
                    z_mm = 12.0
                    height_3d = 24.0
                    width_3d = 24.0
                    thickness_3d = 8.0

                domino_obj = DominoObj(
                    id=obj_id,
                    x=x_mm,
                    y=y_mm,
                    z=z_mm,
                    theta=world_yaw,
                    face_label=face_label
                )
                
                setattr(domino_obj, "length", 48.0)
                setattr(domino_obj, "width", width_3d)
                setattr(domino_obj, "height", height_3d)
                setattr(domino_obj, "thickness", thickness_3d)
                setattr(domino_obj, "is_fallen", obs.is_fallen)
                setattr(domino_obj, "domino_halves", halves)

                world_map.objects[obj_id] = domino_obj

                status_str = "FALLEN" if obs.is_fallen else "STANDING"
                print(f"[TM SNAPSHOT] [{status_str}] {face_label} -> Pos: X={x_mm:.1f}mm, Y={y_mm:.1f}mm, Z={z_mm:.1f}mm @ Yaw={math.degrees(world_yaw):.1f}°")

        except Exception as e:
            print(f"[TM SNAPSHOT] Error: {e}")

    def stop(self):
        cv2.destroyAllWindows()
        super().stop()

    class ReportDominoes(StateNode):
        def start(self, event=None):
            super().start(event)
            if hasattr(self.parent, "update_3d_snapshot"):
                self.parent.update_3d_snapshot()

            dominoes = list(self.robot.world_map.objects.values())
            if not dominoes:
                print("\nNo dominoes registered in world map.")
            else:
                print("\n--- Current WorldMap Dominoes ---")
                for obj in dominoes:
                    print(obj)
            self.post_completion()

    def setup(self):
        intro = Print("\n[READY] Dual Perception Active (bestieee.pt + fallen.pt).\nType 'tm' in terminal for 3D snapshot.").set_name("intro").set_parent(self)
        loop = StateNode().set_name("loop").set_parent(self)
        report = self.ReportDominoes().set_name("report").set_parent(self)
        
        completiontrans1 = CompletionTrans().set_name("completiontrans1")
        completiontrans1.add_sources(intro).add_destinations(loop)
        
        textmsgtrans1 = TextMsgTrans().set_name("textmsgtrans1")
        textmsgtrans1.add_sources(loop).add_destinations(report)
        
        completiontrans2 = CompletionTrans().set_name("completiontrans2")
        completiontrans2.add_sources(report).add_destinations(loop)
        
        return self