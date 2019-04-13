import time
from typing import Dict, Any

import cv2
import numpy as np
from darkflow.net.build import TFNet

if __name__ == '__main__':
    """
    YOLOのパフォーマンスチェック用
    """

    options: Dict[str, Any] = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}

    tfnet: TFNet = TFNet(options)

    imgcv = cv2.imread("./640x480-1.jpg")
    print(type(imgcv), imgcv.shape)
    times: np.ndarray = np.zeros(shape=50, dtype=float)
    for idx in range(50):
        start_at: float = time.perf_counter()
        result = tfnet.return_predict(imgcv)
        times[idx] = time.perf_counter() - start_at
        print(result)
        exit()

    print(times.mean(), times.max(), times.min())

