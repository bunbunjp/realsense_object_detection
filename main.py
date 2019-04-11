import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    config: rs.config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline: rs.pipeline = rs.pipeline()
    # pipeline.wait_for_frames(timeout_ms=1000)
    pipeline.start(config)
    print(pipeline.get_active_profile())
    depth_buffer: np.ndarray = np.zeros(shape=(480, 640), dtype=int)
    color_buffer: np.ndarray = np.zeros(shape=(480, 640, 3), dtype=int)
    while True:
        frames: rs.composite_frame = pipeline.wait_for_frames()
        dep_frame: rs.depth_frame = frames.get_depth_frame()
        video_frame: rs.video_frame = frames.get_color_frame()
        depth_buffer[:] = dep_frame.get_data()
        color_buffer[:] = video_frame.get_data()
        print(depth_buffer[240,320])
        plt.imshow(color_buffer)
        plt.show()


