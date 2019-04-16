import threading
import time
from time import sleep
from typing import Dict, Any, List, Tuple, Optional

import matplotlib.pyplot as plt
import pyrealsense2 as rs

from PIL import Image
import numpy as np
from darkflow.net.build import TFNet
import cv2
from numpy import uint8, uint16


class Recognizer:
    def __init__(self):
        options: Dict[str, Any] = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}
        self.tfnet: TFNet = TFNet(options)
        self.predict_result: Dict[str, Any] = {}

        self.thread: threading.Thread = threading.Thread(target=self.run_in_thread)
        self.target_image: Optional[np.ndarray] = None

    def run_in_thread(self):
        while True:
            if self.target_image is None:
                sleep(1.0)
                continue

            self.predict_result = self.predict(x=self.target_image)

    def predict(self, x: np.ndarray) -> List[Dict[str, Any]]:
        return self.tfnet.return_predict(x)

    def start(self):
        self.thread.start()


class RealSenseReader:
    IMAGE_WIDTH: int = 640
    IMAGE_HEIGHT: int = 480

    def __init__(self):
        self.thread = threading.Thread(target=self.run_in_thread)
        self.config: rs.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.IMAGE_WIDTH, self.IMAGE_HEIGHT, rs.format.bgr8, 6)
        self.config.enable_stream(rs.stream.depth, self.IMAGE_WIDTH, self.IMAGE_HEIGHT, rs.format.z16, 6)
        self.pipeline: rs.pipeline = rs.pipeline()
        self.profile: rs.pipeline_profile = self.pipeline.start(self.config)
        self.device: rs.device = self.profile.get_device()
        self.depth_sensor: rs.depth_sensor = self.device.first_depth_sensor()
        self.depth_scale: float = self.depth_sensor.get_depth_scale()
        self.video_frame: Optional[rs.video_frame] = None
        self.depth_frame: Optional[rs.depth_frame] = None
        self.depth_buffer: np.ndarray = np.zeros(shape=(self.IMAGE_HEIGHT,
                                                        self.IMAGE_WIDTH),
                                                 dtype=float)
        self.color_buffer: np.ndarray = np.zeros(shape=(self.IMAGE_HEIGHT,
                                                        self.IMAGE_WIDTH,
                                                        3),
                                                 dtype=uint16)

    def start(self):
        self.thread.start()

    def run_in_thread(self):
        while True:
            frames: rs.composite_frame = self.pipeline.wait_for_frames()
            self.depth_frame: rs.depth_frame = frames.get_depth_frame()
            self.video_frame: rs.video_frame = frames.get_color_frame()
            self.depth_buffer[:] = self.depth_frame.get_data()
            self.color_buffer[:] = self.video_frame.get_data()


def calculate_center(start: int, end: int) -> int:
    diff: int = end - start
    return int(start + (diff / 2))


if __name__ == '__main__':
    reader: RealSenseReader = RealSenseReader()
    reader.start()

    recognizer: Recognizer = Recognizer()
    recognizer.start()

    reco_target: np.ndarray = np.zeros(shape=(RealSenseReader.IMAGE_HEIGHT,
                                              RealSenseReader.IMAGE_WIDTH, 3),
                                       dtype=uint16)
    depth_target: np.ndarray = np.zeros(shape=(RealSenseReader.IMAGE_HEIGHT,
                                               RealSenseReader.IMAGE_WIDTH, 1),
                                        dtype=float)
    bg_removed: np.ndarray = np.zeros(shape=reco_target.shape,
                                      dtype=uint8)
    clipping_meter: float = 1 / reader.depth_scale

    while True:
        reco_target[:] = reader.color_buffer
        depth_target[:] = np.reshape(a=reader.depth_buffer,
                                     newshape=(RealSenseReader.IMAGE_HEIGHT,
                                               RealSenseReader.IMAGE_WIDTH, 1))
        plt.imshow(reco_target)

        recognizer.target_image = reco_target

        for item in recognizer.predict_result:
            confidence: float = item['confidence']
            if confidence < 0.3:
                continue
            x: int = calculate_center(start=item['topleft']['x'], end=item['bottomright']['x'])
            y: int = calculate_center(start=item['topleft']['y'], end=item['bottomright']['y'])
            z: int = reader.depth_frame.get_distance(x=x, y=y)
            plt.scatter(x=x, y=y, s=50, label='{0}({1:.2f}) ({2:.2f}m)'.format(item['label'], item['confidence'], z))
        plt.legend()
        plt.show()

        sleep(1.0)


