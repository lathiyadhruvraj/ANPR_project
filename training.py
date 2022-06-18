
# import torch, detectron2
# TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
# CUDA_VERSION = torch.__version__.split("+")[-1]
# print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
# print("detectron2:", detectron2.__version__)

# Some basic setup:
# Setup detectron2 logger
import detectron2
# from detectron2.utils.logger import setup_logger
# setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog

import common
from yolov5 import train

def training(config_path):
	content = common.read_config(config_path)

	imgsz = content['train']['imgsz']
	save_dir = content['train']['save_dir']
	epochs = content['train']['epochs']
	batch_size = content['train']['batch_size']
	weights = content['train']['weights']
	data = content['train']['data']
	workers = content['train']['workers']

	print(save_dir, epochs, batch_size, weights, data)




	opt = train.run(img=imgsz, save_dir=save_dir, epochs=epochs, batch_size=batch_size, weights=weights, data=data,
	                workers=workers)
	train.main(opt=opt)