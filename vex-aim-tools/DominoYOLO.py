from aim_fsm import *
from ultralytics import YOLO
import cv2
import numpy as np


seg_model_path = r"yolo_best_weights.pt"
cls_model_path = r"bestmerge.pt"


class DominoYOLO(StateMachineProgram):

    def __init__(self):
        super().__init__()

        print("Loading segmentation model...")
        self.segmenter = YOLO(seg_model_path)

        print("Loading classifier...")
        self.classifier = YOLO(cls_model_path)

        print("\n===== MODEL CHECK =====")
        print("Segmentation task:", self.segmenter.task)
        print("Classification task:", self.classifier.task)
        print("Classifier classes:", self.classifier.names)
        print("=======================\n")

        self.result = None


    def rotate_crop(self, image, obb):

        """
        Convert OBB domino to upright crop
        """

        points = obb.reshape(4,2).astype(np.float32)


        rect = cv2.minAreaRect(points)

        center, size, angle = rect


        width, height = size


        if width < height:
            angle += 90
            width, height = height, width


        M = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )


        rotated = cv2.warpAffine(
            image,
            M,
            (image.shape[1], image.shape[0])
        )


        x,y = int(center[0]), int(center[1])


        pad = 20


        x1 = max(0, x-int(width/2)-pad)
        y1 = max(0, y-int(height/2)-pad)

        x2 = min(
            image.shape[1],
            x+int(width/2)+pad
        )

        y2 = min(
            image.shape[0],
            y+int(height/2)+pad
        )


        crop = rotated[y1:y2, x1:x2]


        return crop, (x1,y1,x2,y2)



    class DetectDominoes(StateNode):

        def start(self,event=None):

            super().start(event)


            frame = self.robot.camera_image


            result = self.parent.segmenter(
                frame,
                conf=0.35,
                verbose=False
            )[0]


            predictions=[]


            # OBB model
            if result.obb is None or len(result.obb)==0:

                print("No domino boxes detected")

                self.parent.result=(result,predictions)
                self.post_data(self.parent.result)
                return



            print(
                "Detected dominoes:",
                len(result.obb)
            )


            obb_points = result.obb.xyxyxyxy.cpu().numpy()


            for obb in obb_points:


                crop,bbox = self.parent.rotate_crop(
                    frame,
                    obb
                )


                if crop.size==0:
                    continue


                print(
                    "Crop:",
                    crop.shape
                )


                h,w,_ = crop.shape


                if h < 60 or w < 60:
                    print("Crop too small")
                    continue



                cls_result = self.parent.classifier(
                    crop,
                    imgsz=224,
                    verbose=False
                )[0]


                probs = cls_result.probs


                top_indices = probs.top5


                label="unknown"
                confidence=0.0



                for idx in top_indices:

                    name = self.parent.classifier.names[int(idx)]


                    if name not in [
                        "Color_Images",
                        "Segmentation"
                    ]:

                        label=name

                        confidence=float(
                            probs.data[int(idx)]
                        )

                        break



                print(
                    "Prediction:",
                    label,
                    confidence
                )


                predictions.append(
                    {
                        "bbox":bbox,
                        "label":label,
                        "confidence":confidence
                    }
                )



            self.parent.result=(
                result,
                predictions
            )


            self.post_data(
                self.parent.result
            )




    class DisplayResults(StateNode):

        def start(self,event=None):

            super().start(event)


            result,predictions = (
                event.data
                if isinstance(event,DataEvent)
                else self.parent.result
            )


            img=result.plot()


            for p in predictions:

                x1,y1,x2,y2=p["bbox"]


                cv2.rectangle(
                    img,
                    (x1,y1),
                    (x2,y2),
                    (0,255,0),
                    2
                )


                cv2.putText(
                    img,
                    f'{p["label"]} {p["confidence"]:.2f}',
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2
                )


            imshow(
                "domino",
                img
            )




    def setup(self):

        begin = StateNode()\
            .set_name("begin")\
            .set_parent(self)


        loop = Print(
            "Type 'tm' to recognize a domino"
        )\
            .set_name("loop")\
            .set_parent(self)



        detect = self.DetectDominoes()\
            .set_name("detect")\
            .set_parent(self)



        display = self.DisplayResults()\
            .set_name("display")\
            .set_parent(self)



        TimerTrans(2)\
            .add_sources(begin)\
            .add_destinations(loop)


        TextMsgTrans()\
            .add_sources(loop)\
            .add_destinations(detect)


        DataTrans()\
            .add_sources(detect)\
            .add_destinations(display)


        NullTrans()\
            .add_sources(display)\
            .add_destinations(loop)


        return self