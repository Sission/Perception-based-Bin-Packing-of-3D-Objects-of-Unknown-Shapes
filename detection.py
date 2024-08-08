#!/usr/bin/env python3
import copy
import os
import shutil

import pyrealsense2 as rs
import rospy
import numpy as np
import cv2
from cv2 import aruco
from datetime import datetime
import open3d as o3d
import json
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import time


def time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Capture the start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Capture the end time
        print(f"Function {func.__name__} took {end_time - start_time:.6f} seconds to execute.")
        return result

    return wrapper


# 4/15 3：40 version


class FetchRS:
    def __init__(self, target_area, maintain=True, serial=128422270676,
                 width=1280, height=720, marker_size=90, stream=False, fps=30):
        # Format the date and time as a string in the desired format
        self.aruco_data = {}
        self.timestamp = datetime.now().strftime('%Y%m%d%H%M')
        print('Time Stamp:', self.timestamp)
        self.target_area = target_area
        self.width = width
        self.height = height
        self.marker_size = marker_size
        self.stream = stream
        self.marker_num = 0

        self.maintain = maintain
        self.pcd = o3d.geometry.PointCloud()

        self.pipeline = rs.pipeline()

        self.config = rs.config()
        self.config.enable_device(str(serial))
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_device(str(serial))

        self.profile = self.pipeline.start(self.config)

        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()

        self.intrinsic, self.k, self.distort, self.timestamp_folder = self.info(
            self.profile.get_stream(rs.stream.depth))

        print('intrinsic: ', self.intrinsic)
        print('k: ', self.k)
        print('distort: ', self.distort)

        # wait for 1s to maker sure color images arrive
        rospy.sleep(1)

    def info(self, st_profile):
        intrinsic = st_profile.as_video_stream_profile().get_intrinsics()
        k = np.array(
            [[intrinsic.fx, 0, intrinsic.ppx], [0, intrinsic.fy, intrinsic.ppy], [0, 0, 1]]
            , dtype=np.float32)
        distort = np.array(intrinsic.coeffs, dtype=np.float32)

        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_file_dir)
        timestamp_folder = os.path.join(parent_dir, 'data', self.timestamp)
        # Check if the folder exists
        if os.path.exists(timestamp_folder):
            # If it exists, delete it
            shutil.rmtree(timestamp_folder)
        os.makedirs(timestamp_folder)

        # camera_info = {'k': k.tolist(), 'distort': distort.tolist(),
        #                'depth_scale': self.depth_scale, 'width': self.width, 'height': self.height}
        #
        # camera_info_path = os.path.join(timestamp_folder, 'camera_info.json')
        #
        # with open(camera_info_path, "w") as json_file:
        #     json.dump(camera_info, json_file)

        return intrinsic, k, distort, timestamp_folder

    def set_config(self, config):
        if config not in ['Default', 'High Accuracy', 'High Density']:
            config = 'Default'

        # use preset configuration
        preset_range = self.depth_sensor.get_option_range(rs.option.visual_preset)
        for i in range(int(preset_range.max)):
            visual_preset = self.depth_sensor.get_option_value_description(rs.option.visual_preset, i)
            if visual_preset == config:
                self.depth_sensor.set_option(rs.option.visual_preset, i)

    def get_rgbd(self):

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The 'align_to' is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        align = rs.align(align_to)

        try:
            while True:
                # Wait for the next set of frames
                frames = self.pipeline.wait_for_frames()

                # Check if both depth and color frames are available
                # Align the depth frame to color frame
                aligned_frames = align.process(frames)
                # Get aligned frames
                aligned_depth_frame = aligned_frames.get_depth_frame()

                color_frame = frames.get_color_frame()

                if not aligned_frames:
                    print('Depth frame is not available.')
                    continue  # Skip this iteration of the loop and try again

                if not color_frame:
                    print('Color frame is not available.')
                    continue

                # depth image is 1 channel (16U1), color is 3 channels (8U3)
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                self.detect_markers_and_plot_axes(color_image)

                if self.marker_num != 0 and not self.stream:
                    cv2.imwrite(os.path.join(self.timestamp_folder, 'color_image.png'), color_image)
                    cv2.imwrite(os.path.join(self.timestamp_folder, 'depth_image.png'), depth_image)
                    cv2.imwrite(os.path.join(self.timestamp_folder, 'depth_colormap.png'), depth_colormap)

                    np_cloud, np_cloud_filtered = self.depth_1d_to_3d(color_image, depth_image)
                    pcd = o3d.geometry.PointCloud()
                    pcd_filtered = o3d.geometry.PointCloud()

                    pcd.points = o3d.utility.Vector3dVector(np_cloud)
                    pcd_filtered.points = o3d.utility.Vector3dVector(np_cloud_filtered)

                    pcd.transform([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                    pcd_filtered.transform([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

                    o3d.visualization.draw_geometries([pcd_filtered])

                    point_cloud_path = os.path.join(self.timestamp_folder, 'point_cloud.pcd')
                    o3d.io.write_point_cloud(point_cloud_path, pcd)

                    return

                else:
                    cv2.imshow('Color Image', color_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            # Stop pipeline and clean up
            pass
            # self.pipeline.stop()
            # cv2.destroyAllWindows()

    @time_decorator
    def get_3d(self):
        # ... (your existing code)
        align_to = rs.stream.color
        align = rs.align(align_to)
        vis = o3d.visualization.Visualizer()
        try:
            while True:
                start_time = time.time()
                num_frames = 0
                total_depth_image = np.zeros((self.height, self.width), dtype=np.float32)
                total_color_image = np.zeros((self.height, self.width, 3), dtype=np.float32)

                # while time.time() - start_time < 2:
                # print('Time:', time.time() - start_time)
                # Wait for the next set of frames
                frames = self.pipeline.wait_for_frames()

                # Check if both depth and color frames are available
                # Align the depth frame to color frame
                aligned_frames = align.process(frames)
                # Get aligned frames
                aligned_depth_frame = aligned_frames.get_depth_frame()

                color_frame = frames.get_color_frame()

                if not aligned_depth_frame or not color_frame:
                    print('Depth or color frame is not available.')
                    return

                # depth image is 1 channel (16U1), color is 3 channels (8U3)
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                total_depth_image += depth_image
                total_color_image += color_image
                num_frames += 1
                # Calculate average depth and color images
                depth_image = total_depth_image / num_frames
                color_image = (total_color_image / num_frames).astype(np.uint8)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                cv2.imwrite(os.path.join(self.timestamp_folder, 'color_image.png'), color_image)

                self.detect_markers_and_plot_axes(color_image)

                if not self.maintain:
                    filtered_points, all_points = self.depth_1d_to_3d(color_image, depth_image)
                    self.pcd.points = o3d.utility.Vector3dVector(np.array(all_points))

                    return

                if self.marker_num != 0 and not self.stream:
                    vis.create_window()

                    cv2.imwrite(os.path.join(self.timestamp_folder, 'depth_image.png'), depth_image)
                    cv2.imwrite(os.path.join(self.timestamp_folder, 'depth_colormap.png'), depth_colormap)

                    # Create a point cloud from the color image
                    filtered_pcd = o3d.geometry.PointCloud()
                    pcd = o3d.geometry.PointCloud()

                    filtered_points, all_points = self.depth_1d_to_3d(color_image, depth_image)

                    filtered_pcd.points = o3d.utility.Vector3dVector(np.array(filtered_points)[:, :3])
                    filtered_pcd.colors = o3d.utility.Vector3dVector(np.array(filtered_points)[:, 3:])

                    pcd.points = o3d.utility.Vector3dVector(np.array(all_points))

                    vis.add_geometry(filtered_pcd)

                    point_cloud_path = os.path.join(self.timestamp_folder, 'point_cloud.pcd')
                    o3d.io.write_point_cloud(point_cloud_path, pcd)

                    for i in range(self.marker_num):
                        frame_offset = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                        ht = np.array(self.aruco_data['ht'][i])

                        frame_offset.translate((ht[0, 3],
                                                ht[1, 3],

                                                ht[2, 3]))
                        rotation_matrix = ht[:3, :3]  # Extract the 3x3 rotation matrix
                        frame_offset.rotate(rotation_matrix,
                                            center=(ht[0, 3], ht[1, 3], ht[2, 3]))
                        vis.add_geometry(frame_offset)  # Add the offset coordinate frame

                    frame_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    vis.add_geometry(frame_origin)

                    view_control = vis.get_view_control()
                    view_control.set_front([0, 0, -1])  # Set the front vector to [0, 0, -1] (along negative z-axis)
                    view_control.set_up([0, -1, 0])  # Set the up vector to [0, -1, 0] (along negative y-axis)
                    view_control.set_lookat([0, 0, 0])  # Set the look-at point to [0, 0, 0] (origin)
                    view_control.set_zoom(0.2)  # Set the zoom level
                    # Rotate the camera view 180 degrees around the y-axis
                    vis.run()

                    return

                else:
                    cv2.imshow('Color Image', color_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            pass

    @time_decorator
    def depth_1d_to_3d(self, color_image, depth_image):
        # depth_3d_cam is the position of points based on camera coordinate frame.
        filtered_points = []
        all_points = []
        for y in range(self.height):
            for x in range(self.width):
                z = depth_image[y, x]
                # if z > 0:
                pt_3d = rs.rs2_deproject_pixel_to_point(self.intrinsic, [x, y], z)
                scaled_pt_3d = [coord * self.depth_scale for coord in pt_3d]
                color = color_image[y, x] / 255.0  # Normalize color values to [0, 1]
                all_points.append([scaled_pt_3d[0], scaled_pt_3d[1], scaled_pt_3d[2]])
                if self.target_area['x_top_min'] < x < self.target_area['x_top_max'] and \
                        self.target_area['y_min'] - 100 < y < self.target_area['y_max']:
                    filtered_points.append(
                        [scaled_pt_3d[0], scaled_pt_3d[1], scaled_pt_3d[2], color[2], color[1], color[0]])

        return filtered_points, all_points

    @time_decorator
    def detect_markers_and_plot_axes(self, color_image):
        self.marker_num = 0
        arucoDict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        arucoParams = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(arucoDict, arucoParams)
        corners, ids, rejected = detector.detectMarkers(color_image)

        img_w_frame = copy.deepcopy(color_image)

        if ids is not None:
            self.marker_num = len(ids)
            self.aruco_data = {'ids': [], 'ht': [], 'centroid': [], 'corners': []}
            for i in range(self.marker_num):
                # Get the center as the coordinate frame center
                if ids[i] == 49:
                    marker_size = 100
                else:
                    marker_size = self.marker_size
                m = marker_size / 2
                objPoints = np.array([[-m, m, 0], [m, m, 0], [m, -m, 0], [-m, -m, 0]], dtype=np.float32).reshape(
                    (4, 1, 3))
                valid, rvec, tvec = cv2.solvePnP(objPoints, corners[i], np.reshape(self.k, (3, 3)),
                                                 np.array(self.distort))
                if valid:
                    ht = np.identity(4)
                    rot, _ = cv2.Rodrigues(rvec)
                    ht[:3, 3] = (tvec / 1000.0).reshape(3)  ## unit: meter
                    ht[:3, :3] = rot

                    centroid = [int(np.mean(corners[i][0][:, 0])), int(np.mean(corners[i][0][:, 1]))]

                    self.aruco_data['ids'].append(int(ids[i][0]))
                    self.aruco_data['ht'].append(ht.tolist())
                    self.aruco_data['centroid'].append(centroid)
                    self.aruco_data['corners'].append(corners[i][0].tolist())

                    if self.stream:
                        img_w_frame = cv2.circle(img_w_frame, tuple(centroid), radius=10, color=(255, 255, 255),
                                                 thickness=-1)
                        aruco.drawDetectedMarkers(img_w_frame, corners)
                        cv2.drawFrameAxes(color_image, self.k, self.distort, rvec, tvec, 100)
                else:
                    self.marker_num -= 1

            if not self.maintain:
                return

            aruco_info_path = os.path.join(self.timestamp_folder, 'aruco_data.json')
            with open(aruco_info_path, "w") as json_file:
                json.dump(self.aruco_data, json_file)


def main():
    w = FetchRS(serial=241122305823, width=1280, height=720, stream=True)
    # w = FetchRS(serial=128422270676, width=1280, height=720, stream=False)
    w.get_rgbd()


if __name__ == '__main__':
    main()
