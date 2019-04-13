import threading
import time
from time import sleep
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt
import pyrealsense2 as rs

from PIL import Image
import numpy as np
from darkflow.net.build import TFNet
import cv2
from numpy import uint8


class Recognizer:
    def __init__(self):
        options: Dict[str, Any] = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}
        self.tfnet: TFNet = TFNet(options)

    def predict(self, x: np.ndarray) -> List[Dict[str, Any]]:
        return self.tfnet.return_predict(x)


class RealSenseReader:
    IMAGE_WIDTH: int = 640
    IMAGE_HEIGHT: int = 480

    def __init__(self):
        self.thread = threading.Thread(target=self.run_in_thread)
        self.config: rs.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.IMAGE_WIDTH, self.IMAGE_HEIGHT, rs.format.bgr8, 6)
        self.config.enable_stream(rs.stream.depth, self.IMAGE_WIDTH, self.IMAGE_HEIGHT, rs.format.z16, 6)
        self.pipeline: rs.pipeline = rs.pipeline()
        self.depth_buffer: np.ndarray = np.zeros(shape=(self.IMAGE_HEIGHT,
                                                        self.IMAGE_WIDTH),
                                                 dtype=uint8)
        self.color_buffer: np.ndarray = np.zeros(shape=(self.IMAGE_HEIGHT,
                                                        self.IMAGE_WIDTH,
                                                        3),
                                                 dtype=uint8)

    def start(self):
        self.pipeline.start(self.config)
        self.thread.start()

    def run_in_thread(self):
        while True:
            frames: rs.composite_frame = self.pipeline.wait_for_frames()
            dep_frame: rs.depth_frame = frames.get_depth_frame()
            video_frame: rs.video_frame = frames.get_color_frame()
            self.depth_buffer[:] = dep_frame.get_data()
            self.color_buffer[:] = video_frame.get_data()


if __name__ == '__main__':
    reader: RealSenseReader = RealSenseReader()
    reader.start()
    recoginzer: Recognizer = Recognizer()

    reco_target: np.ndarray = np.zeros(shape=(RealSenseReader.IMAGE_HEIGHT,
                                              RealSenseReader.IMAGE_WIDTH, 3),
                                       dtype=uint8)
    while True:
        reco_target[:] = reader.color_buffer
        plt.imshow(reco_target)
        plt.show()
        print(reco_target.dtype, reco_target.shape)
        print(recoginzer.predict(x=reco_target))


