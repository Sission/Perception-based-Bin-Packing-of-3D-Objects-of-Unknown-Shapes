#!/usr/bin/env python3
import copy
import sys
import os
import time

import rospy

src_folder = os.path.dirname(os.path.abspath(__file__))


# Function to recursively add subdirectories to the Python path
def add_subdirectories_to_path(folder):
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        if os.path.isdir(item_path):
            sys.path.append(item_path)
            add_subdirectories_to_path(item_path)


# Add 'src' folder to the Python path
sys.path.append(src_folder)

# Add all subdirectories of 'src' to the Python path
add_subdirectories_to_path(src_folder)

from AruCo import detection
from segment import segmentation
from point_cloud import pc_estimation
from robot import nesting_motion_test
from moveit_control import *


def maintain_main():
    # target_area in image pixel coordinates
    target_area = {"x_top_min": 210, "x_top_max": 1000, "x_bottom_min": 370, "x_bottom_max": 840,
                   "y_min": 100, "y_max": 700}
    fetch = detection.FetchRS(target_area=target_area, serial=241122305823,
                              width=1280, height=720, marker_size=36.5, stream=False)

    while True:
        fetch.get_3d()
        # detected pose is accurate, use its pixel value and rotation
        timestamp = fetch.timestamp
        segment = segmentation.Segment(timestamp, target_area=target_area, maintain=True, grid_size=18, show_all=True,
                                       preprocess_image=True)
        res = segment.execute()
        if res:
            break



if __name__ == "__main__":
    rospy.init_node("franka_motion")
    rospy.Rate(100)
    maintain_main()
