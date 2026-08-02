import math
import cv2
import numpy as np
from ultralytics import YOLO


class Observation:
    """Holds detection metadata for world-map projection."""
    def __init__(self, center_xy, quad, axis_endpoints, divider_endpoints,
                 half_counts, half_image_centers, face_label, face_confidence,
                 confidence, mask_area):
        self.center_xy = center_xy
        self.quad = quad
        self.axis_endpoints = axis_endpoints
        self.divider_endpoints = divider_endpoints
        self.half_counts = half_counts
        self.half_image_centers = half_image_centers
        self.face_label = face_label
        self.face_confidence = face_confidence
        self.confidence = confidence
        self.mask_area = mask_area


class DominoWorldDetector:
    def __init__(self, conf_threshold=0.3, weights_path=None, label_provider=None):
        self.conf_threshold = conf_threshold
        # Load custom YOLO segmentation model and half-face model
        seg_weights = weights_path if weights_path else "bestieee.pt"
        self.seg_model = YOLO(seg_weights)
        self.half_model = YOLO("different.pt")
        self._latest_observations = []

    def detect(self, image, frame_id=None):
        if image is None:
            return []

        h, w = image.shape[:2]
        results = self.seg_model(image, conf=self.conf_threshold, verbose=False)
        observations = []

        if not results or len(results) == 0 or results[0].masks is None:
            self._latest_observations = []
            return []

        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes.conf.cpu().numpy()

        for idx, mask_raw in enumerate(masks):
            conf = float(boxes[idx])
            mask = (mask_raw > 0.5).astype(np.uint8) * 255
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            c = max(contours, key=cv2.contourArea)
            mask_area = float(cv2.contourArea(c))
            if mask_area < 200:
                continue

            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Extract oriented bounding patch for black dividing line search
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])

            # Detect black dividing line in domino mask
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
            
            # Find dark/black pixels along internal lines
            edges = cv2.Canny(masked_gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15, minLineLength=10, maxLineGap=5)

            divider_p0, divider_p1 = None, None
            if lines is not None:
                # Find line closest to mask centroid
                best_dist = float("inf")
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    dist = math.hypot(mx - cx, my - cy)
                    if dist < best_dist:
                        best_dist = dist
                        divider_p0 = (float(x1), float(y1))
                        divider_p1 = (float(x2), float(y2))

            # Define axis endpoints along long axis
            (width, height) = rect[1]
            angle = rect[2]
            if width < height:
                angle += 90

            rad = math.radians(angle)
            dx = math.cos(rad) * max(width, height) / 2.0
            dy = math.sin(rad) * max(width, height) / 2.0
            axis_p0 = (cx - dx, cy - dy)
            axis_p1 = (cx + dx, cy + dy)

            # Split domino into two half-faces along the dividing line/axis
            half1_center = (cx - dx / 2.0, cy - dy / 2.0)
            half2_center = (cx + dx / 2.0, cy + dy / 2.0)

            # Crop halves and classify via different.pt
            counts = []
            for hx, hy in [half1_center, half2_center]:
                x_min = max(0, int(hx - 20))
                x_max = min(w, int(hx + 20))
                y_min = max(0, int(hy - 20))
                y_max = min(h, int(hy + 20))
                
                half_crop = image[y_min:y_max, x_min:x_max]
                if half_crop.size == 0:
                    counts.append("?")
                    continue

                half_res = self.half_model(half_crop, verbose=False)
                if half_res and len(half_res[0].boxes) > 0:
                    cls_id = int(half_res[0].boxes.cls[0].cpu().numpy())
                    counts.append(str(cls_id))
                else:
                    counts.append("?")

            face_label = f"{counts[0]}-{counts[1]}"

            divider_endpoints = (divider_p0, divider_p1) if (divider_p0 and divider_p1) else None
            obs = Observation(
                center_xy=(cx, cy),
                quad=tuple((float(pt[0]), float(pt[1])) for pt in box),
                axis_endpoints=(axis_p0, axis_p1),
                divider_endpoints=divider_endpoints,
                half_counts=tuple(counts),
                half_image_centers=(half1_center, half2_center),
                face_label=face_label,
                face_confidence=conf,
                confidence=conf,
                mask_area=mask_area,
            )
            observations.append(obs)

        self._latest_observations = observations
        return observations

    def latest_observations(self):
        return self._latest_observations

    def latest_debug_contact_sheet(self, columns=2):
        return None