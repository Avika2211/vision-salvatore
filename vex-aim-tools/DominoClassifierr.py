from aim_fsm import *
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision import transforms
from PIL import Image


seg_model_path = r"bestieee.pt"
cls_model_path = r"different.pt"

NUM_CLASSES = 7
MIN_HALF_SIZE = 4
WHITE_THRESHOLD = 250
BORDER_MARGIN = 4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replaced with standalone utility transforms inside the class method
PYTORCH_TENSOR_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class DominoClassifierr(StateMachineProgram):

    def __init__(self):
        super().__init__()

        print("Loading segmentation model...")
        self.segmenter = YOLO(seg_model_path)

        print("Loading half-face classifier...")
        self.classifier = self._load_classifier(cls_model_path)

        # Initialize CLAHE block for uniform deployment
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        print("\n===== MODEL CHECK =====")
        print("Segmentation task:", self.segmenter.task)
        print("Classifier device:", DEVICE)
        print("Classifier classes:", NUM_CLASSES)
        print("=======================\n")

        self.result = None


    def _load_classifier(self, path):
        """
        Load the EfficientNet-B0 half-face classifier from a raw
        state_dict (NOT a YOLO model). Returns None if loading fails so
        the rest of the pipeline can degrade gracefully.
        """
        try:
            model = efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
            state_dict = torch.load(path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()
            print(f"Classifier loaded successfully from '{path}' on {DEVICE}")
            return model
        except Exception as exc:
            import traceback
            print(f"ERROR: failed to load classifier '{path}': {exc}")
            traceback.print_exc()
            return None


    def rotate_crop(self, image, obb):
        """
        Convert OBB domino to upright crop
        """
        points = obb.reshape(4, 2).astype(np.float32)
        rect = cv2.minAreaRect(points)
        center, size, angle = rect
        width, height = size

        if width <= 0 or height <= 0:
            return None, None

        if width < height:
            angle += 90
            width, height = height, width

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            M,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )

        x, y = int(center[0]), int(center[1])
        pad = 20

        x1 = max(0, x - int(width / 2) - pad)
        y1 = max(0, y - int(height / 2) - pad)
        x2 = min(image.shape[1], x + int(width / 2) + pad)
        y2 = min(image.shape[0], y + int(height / 2) + pad)

        crop = rotated[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None, None

        return crop, (x1, y1, x2, y2)


    def split_halves(self, image):
        """
        Split an upright domino crop vertically into left and right
        halves. Returns (None, None) if the crop is too small to split.
        """
        try:
            if image is None or image.size == 0:
                return None, None

            h, w = image.shape[:2]
            mid = w // 2

            if mid < MIN_HALF_SIZE or (w - mid) < MIN_HALF_SIZE:
                return None, None

            left = image[:, :mid].copy()
            right = image[:, mid:].copy()
            return left, right
        except Exception as exc:
            print(f"split_halves failed: {exc}")
            return None, None


    def remove_white_border(self, image, threshold=WHITE_THRESHOLD, margin=BORDER_MARGIN):
        """
        Trim near-white padding from the edges of a half-face crop,
        keeping a small margin around the remaining non-white content.
        """
        try:
            if image is None or image.size == 0:
                return image

            keep_mask = np.any(image < threshold, axis=2)
            if not np.any(keep_mask):
                return image

            ys, xs = np.where(keep_mask)
            y0 = max(0, int(ys.min()) - margin)
            y1 = min(image.shape[0], int(ys.max()) + margin + 1)
            x0 = max(0, int(xs.min()) - margin)
            x1 = min(image.shape[1], int(xs.max()) + margin + 1)

            trimmed = image[y0:y1, x0:x1]
            if trimmed is None or trimmed.size == 0:
                return image

            return trimmed
        except Exception as exc:
            print(f"remove_white_border failed: {exc}")
            return image


    def preprocess_half(self, half_bgr):
        """
        Exactly reproduce training preprocessing:
        BGR ➔ Grayscale ➔ Bilateral Filter ➔ CLAHE ➔ Triplicate Channels ➔ PyTorch Tensor
        """
        try:
            if half_bgr is None or half_bgr.size == 0:
                return None

            if half_bgr.shape[0] < MIN_HALF_SIZE or half_bgr.shape[1] < MIN_HALF_SIZE:
                return None

            # 1. Convert incoming robot BGR structure to Grayscale
            gray = cv2.cvtColor(half_bgr, cv2.COLOR_BGR2GRAY)

            # 2. Smooth out surface grain variations while locking edges
            filtered = cv2.bilateralFilter(gray, 5, 75, 75)

            # 3. Apply Local Contrast Normalization
            equalized = self.clahe.apply(filtered)

            # 4. Triplicate single-channel grayscale block to match expected 3-channel input
            three_channel = np.repeat(equalized[:, :, np.newaxis], 3, axis=2)

            # 5. Hand over to PIL and process spatial resizing/normalization tensors
            pil_image = Image.fromarray(three_channel)
            tensor = PYTORCH_TENSOR_TRANSFORMS(pil_image)
            tensor = tensor.unsqueeze(0)

            return tensor
        except Exception as exc:
            print(f"preprocess_half failed: {exc}")
            return None


    def predict_half(self, tensor):
        """
        Run the EfficientNet-B0 classifier on a single preprocessed
        half-face tensor. Returns (predicted_class, confidence).
        """
        try:
            if self.classifier is None:
                print("predict_half: classifier is not loaded (check startup errors above)")
                return None, 0.0

            if tensor is None:
                print("predict_half: input tensor is None (preprocessing must have failed)")
                return None, 0.0

            with torch.no_grad():
                tensor = tensor.to(DEVICE)
                logits = self.classifier(tensor)
                probs = torch.softmax(logits, dim=1)
                confidence, predicted = torch.max(probs, dim=1)

                pred_class = int(predicted.item())
                conf_value = float(confidence.item())

            if pred_class < 0 or pred_class >= NUM_CLASSES:
                print(f"predict_half: predicted class {pred_class} out of range")
                return None, 0.0

            return pred_class, conf_value
        except Exception as exc:
            import traceback
            print(f"predict_half failed: {exc}")
            traceback.print_exc()
            return None, 0.0


    class DetectDominoes(StateNode):

        def start(self, event=None):
            super().start(event)
            frame = self.robot.camera_image

            if frame is None:
                print("No camera frame available")
                self.parent.result = (None, [])
                self.post_data(self.parent.result)
                return

            result = self.parent.segmenter(
                frame,
                conf=0.35,
                verbose=False
            )[0]

            predictions = []
            obb = getattr(result, "obb", None)
            boxes = getattr(result, "boxes", None)
            quads = []

            if obb is not None and len(obb) > 0:
                print("Detected dominoes (OBB):", len(obb))
                quads = list(obb.xyxyxyxy.cpu().numpy())
            elif boxes is not None and len(boxes) > 0:
                print("Detected dominoes (axis-aligned):", len(boxes))
                xyxy = boxes.xyxy.cpu().numpy()
                for box in xyxy:
                    x1b, y1b, x2b, y2b = box
                    quad = np.array(
                        [
                            [x1b, y1b],
                            [x2b, y1b],
                            [x2b, y2b],
                            [x1b, y2b]
                        ],
                        dtype=np.float32
                    )
                    quads.append(quad)

            if not quads:
                print("No domino boxes detected")
                self.parent.result = (result, predictions)
                self.post_data(self.parent.result)
                return

            for obb in quads:
                crop, bbox = self.parent.rotate_crop(frame, obb)
                if crop is None or crop.size == 0:
                    continue

                print("Crop:", crop.shape)
                h, w, _ = crop.shape
                if h < 60 or w < 60:
                    print("Crop too small")
                    continue

                left_raw, right_raw = self.parent.split_halves(crop)
                if left_raw is None or right_raw is None:
                    print("Could not split crop into halves")
                    continue

                left_clean = self.parent.remove_white_border(left_raw)
                right_clean = self.parent.remove_white_border(right_raw)

                if (
                    left_clean is None or right_clean is None or
                    left_clean.size == 0 or right_clean.size == 0
                ):
                    print("Half crop invalid after border removal")
                    continue

                left_tensor = self.parent.preprocess_half(left_clean)
                right_tensor = self.parent.preprocess_half(right_clean)

                if left_tensor is None or right_tensor is None:
                    print("Half preprocessing failed")
                    continue

                left_pred, left_conf = self.parent.predict_half(left_tensor)
                right_pred, right_conf = self.parent.predict_half(right_tensor)

                if left_pred is None or right_pred is None:
                    print("Half prediction failed")
                    continue

                label = f"{left_pred}-{right_pred}"
                confidence = min(left_conf, right_conf)

                print("Prediction:", label, confidence)
                predictions.append(
                    {
                        "bbox": bbox,
                        "label": label,
                        "confidence": confidence,
                        "left_pred": left_pred,
                        "right_pred": right_pred,
                        "left_conf": left_conf,
                        "right_conf": right_conf
                    }
                )

            self.parent.result = (result, predictions)
            self.post_data(self.parent.result)


    class DisplayResults(StateNode):

        def start(self, event=None):
            super().start(event)
            result, predictions = (
                event.data
                if isinstance(event, DataEvent)
                else self.parent.result
            )

            if result is None:
                print("Nothing to display")
                return

            img = result.plot()
            for p in predictions:
                x1, y1, x2, y2 = p["bbox"]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f'{p["label"]} {p["confidence"]:.2f}',
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            imshow("domino", img)


    def setup(self):
        begin = StateNode().set_name("begin").set_parent(self)
        loop = Print("Type 'tm' to recognize a domino").set_name("loop").set_parent(self)
        detect = self.DetectDominoes().set_name("detect").set_parent(self)
        display = self.DisplayResults().set_name("display").set_parent(self)

        TimerTrans(2).add_sources(begin).add_destinations(loop)
        TextMsgTrans().add_sources(loop).add_destinations(detect)
        DataTrans().add_sources(detect).add_destinations(display)
        NullTrans().add_sources(display).add_destinations(loop)

        return self