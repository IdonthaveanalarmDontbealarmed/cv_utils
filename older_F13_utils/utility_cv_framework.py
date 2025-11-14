# Various commands from Ultralytics YOLO lifecycle.
# Lazy people mana. Comment a block, uncomment another block. 

# SETTINGS
# import ultralytics
# print("ultralytics settings")
# from ultralytics import settings
# print(settings)

# BENCHMARK
# from ultralytics.utils.benchmarks import benchmark
# benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, half=False)
# benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, half=False, device="cpu")
# benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)

# TRAIN
# from ultralytics import YOLO
# model = YOLO("yolo11s.yaml").load("yolo11n.pt") # build from YAML and transfer weights
# results = model.train(data="C:\\python\\datasets\\ippt.yml", epochs=100, imgsz=640) # Train the model

# # RESUME TRAINING
from ultralytics import YOLO
model = YOLO("C:\\python\\game-of-drones\\runs\\detect\\train7\\weights\\best.pt")  # load a partially trained model
results = model.train(resume=True, data="C:\\python\\datasets\\ippt.yml", epochs=120, imgsz=640)

# EXPORT
# from ultralytics import YOLO
# model = YOLO("yolo11n_mavic.pt")
# model.export(format="ncnn", half = True)
# model.export(format="onnx")
# ov_model = YOLO("cv/yolo11n-best-ver3_openvino_model/")
# results = ov_model("https://ultralytics.com/images/bus.jpg")

# INTERFERENCE
# from ultralytics import YOLO
# model = YOLO('yolo11n_mavic.onnx')
# model = YOLO('yolo11n_mavic_ncnn_model')
# results = model(source=0, show = True, imgsz=320, conf = 0.6, save = True) #source = '...filename.xt' or =1 -webcam 
# model = YOLO('best-f1dpl.pt')
# results = model(source=0, show = True, conf = 0.4, save = True) #source = '...filename.xt' or =1 -webcam
# results = model.track(source=0, show=True, conf = 0.6, save=True)  # Tracking with default tracker

# EXPLORE THE RESULTS DATA OBJECT
# from ultralytics import YOLO
# model = YOLO('cv/yolo11n-f1dpl-best_openvino_model/')
# results = model.track(source=0, show=True, conf = 0.6)
# import pprint
# pprint.pprint(vars(results))
# pprint.pprint(vars(results[0].boxes))
# pprint.pprint(vars(results[0].masks))
# pprint.pprint(vars(results[0].probs))