import cv2

import os
import cv2
import numpy as np

from config import Configs
from detector import Detector  # Face Detector
from video_helper import VideoHelper
from multiple_object_controller import MultipleObjectController
from acceptor import Acceptor


def run():
    # Step 1: Initialization
    # 视频： video_helper.py
    # 检测： detector.py
    # 结果接收：acceptor.py
    # 参数配置: config.py
    # 总控： multiple_object_controller.py
    configs = Configs()
    detector = Detector(configs)
    acceptor = Acceptor(configs)
    video_helper = VideoHelper(configs)
    object_controller = MultipleObjectController(configs, video_helper)

    # step 2: 总体流程：main loop
    # A: 对物体，每帧检测，不要跟踪 （可以要平滑）
    # B: 对物体，要跟踪： a. 此帧有检测 (+observation correction)
    #                  b. 此帧无检测（只跟踪，pure predicton）
    cur_frame_counter = 0
    detection_loop_counter = 0
    while video_helper.not_finished(cur_frame_counter):
        # 0. get frame
        frame = video_helper.get_frame()

        # 1.1 每帧都检测
        if not configs.NUM_JUMP_FRAMES:
            detects = detector.detect(frame)
            object_controller.update(detects)
        else:
            # 1.2 隔帧检测
            # 1.2.1 此帧有检测
            if detection_loop_counter % configs.NUM_JUMP_FRAMES == 0:
                detection_loop_counter = 0
                detects = detector.detect(frame)
                object_controller.update(detects)                   # 核心
            # 1.2.2 此帧无检测
            else:
                object_controller.update_without_detection()        # 核心

        # deal with acceptor
        # ask acceptor do something
        cur_frame_counter += 1
        detection_loop_counter += 1


if __name__ == "__main__":
    run()
