from ultralytics import YOLO

model = YOLO('yolo26s-seg.pt')

results = model.train(data=r"./data/data.yaml", epochs=20, imgsz=160)
