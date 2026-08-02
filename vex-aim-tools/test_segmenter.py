import os
import cv2

from ultralytics import YOLO

WEIGHTS_VERSION = "train" # or "train2" or "train3" or ...

model = YOLO(r'runs/segment/' + WEIGHTS_VERSION + r'/weights/best.pt')
model.eval()

TEST_IMAGE_DIR = 'data/test/images/'

for file in os.listdir(TEST_IMAGE_DIR):
  test_img = TEST_IMAGE_DIR + file
  res = model(test_img, conf=0.7)
  res0 = res[0]
  print('Result:', res0.verbose())
  cv2.imshow('res',res0.plot())
  cv2.waitKey(0)

cv2.destroyAllWindows()
print('Done.')