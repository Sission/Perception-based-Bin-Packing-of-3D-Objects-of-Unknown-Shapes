#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import io
import cv2
import json
import open3d as o3d
from EfficientSAM.efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
import zipfile

from segment_anything.utils.amg import (
    batched_mask_to_box,
    calculate_stability_score,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
)
from torchvision.ops.boxes import batched_nms, box_area
from matplotlib.patches import Circle
import math
from preprocess_image import read_ndarray_from_file, save_list_to_file, create_mask_from_points

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time


def time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Capture the start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Capture the end time
        print(f"Function {func.__name__} took {end_time - start_time:.6f} seconds to execute.")
        return result

    return wrapper


# Mask Shape: (720, 1280)
# Height: 720, Width: 1280
# Figure size: (1280, 720)

def calculate_subplot_dimensions(num_plots):
    num_cols = math.ceil(math.sqrt(num_plots))
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows needed
    return num_rows, num_cols


class Segment:
    def __init__(self, timestamp, target_area, maintain=True,
                 grid_size=8, show_all=True, preprocess_image=True):

        self.current_file_dir = os.path.dirname(os.path.abspath(__file__))
        self.maintain = maintain
        if self.maintain:
            self.timestamp = timestamp
            aruco_data_path = os.path.join(self.current_file_dir, f"../data/{self.timestamp}", 'aruco_data.json')
            with open(aruco_data_path, "r") as json_file:
                self.aruco_data = json.load(json_file)
            print('Read aruco_data.json')

            point_cloud_dir = os.path.join(self.current_file_dir, f"../data/{self.timestamp}", 'point_cloud.pcd')
            self.pcd = o3d.io.read_point_cloud(point_cloud_dir)

        else:
            self.timestamp = timestamp.timestamp
            self.aruco_data = timestamp.aruco_data
            self.pcd = timestamp.pcd

        self.show_all = show_all
        self.target_area = [
            [target_area['x_top_max'], target_area['y_max']],
            [target_area['x_top_min'], target_area['y_max']],
            [target_area['x_bottom_min'], target_area['y_min']],
            [target_area['x_bottom_max'], target_area['y_min']]
        ]
        self.base_area = [
            [target_area['x_top_max'] - 10, target_area['y_max'] - 10],
            [target_area['x_top_min'] + 10, target_area['y_max'] - 10],
            [target_area['x_bottom_min'] + 10, target_area['y_min'] + 10],
            [target_area['x_bottom_max'] - 10, target_area['y_min'] + 10]
        ]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: ", self.device)
        self.grid_size = grid_size
        # Get the current file directory

        # Load the image
        self.image = None
        self.image_tensor = self.process_image(preprocess_image)
        print("Images loaded")

        _, self.image_h, self.image_w = self.image_tensor.shape
        print("Image height:", self.image_h, 'Image width:', self.image_w)
        self.mask_data = {'ids': [], 'ht': [], 'centroid': [], 'corners': [], 'mask': [],
                          'base_info': {'mask': None, 'ids': None, 'ht': None, 'centroid': None, 'corners': None}}

        self.model = build_efficient_sam_vits()
        self.model.eval()
        print('SAM_VITS model loaded')

    @time_decorator
    def process_image(self, preprocess_image):
        image_path = os.path.join(self.current_file_dir, f"../data/{self.timestamp}", 'color_image.png')
        self.image = cv2.imread(image_path)

        if preprocess_image:
            # points = [(400, 200), (800, 200), (900, 720), (250, 720)]
            points = [tuple(point) for point in self.target_area]
            image = np.ones((720, 1280, 3), dtype=np.uint8) * 255
            mask = create_mask_from_points(image.shape, points)
            #
            # file_path = os.path.join(self.current_file_dir, "data", 'processed_mask.txt')
            # mask = read_ndarray_from_file(file_path)

            # Resize mask to match the shape of the image if necessary
            if self.image.shape[:2] != mask.shape:
                mask = cv2.resize(mask, (self.image.shape[1], self.image.shape[0]))

            # Convert mask to uint8 type (0 or 255) if it's a boolean mask
            if mask.dtype == bool:
                mask = mask.astype(np.uint8) * 255

            self.image[np.logical_not(mask)] = [0, 0, 0]

        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        # Save the processed image
        # print("Image shape:", self.image.shape)
        return ToTensor()(self.image).to(self.device)

    @time_decorator
    def segment(self):
        self.model.to(self.device)

        points, num_points = self.generate_grid()
        print("Grid points generated")

        input_points = points.reshape(1, num_points, 1, 2).to(self.device)
        input_labels = torch.ones((1, num_points, 1), device=self.device)

        with torch.no_grad():
            print('Running model...')
            predicted_masks, predicted_iou = \
                self.get_predictions(self.image_tensor, input_points, input_labels)
            print('Model run complete')

        rle = [mask_to_rle_pytorch(m[0:1]) for m in predicted_masks]
        predicted_masks = self.process_small_region(rle)
        print("Masks processed")
        return predicted_masks

    @time_decorator
    def generate_grid(self):
        step_x = self.image_w / self.grid_size
        step_y = self.image_h / self.grid_size

        # Generate grid indices
        indices_x = torch.arange(0.5, self.image_w, step_x)
        indices_y = torch.arange(0.5, self.image_h, step_y)

        # Create grid by taking outer product of indices
        xx, yy = torch.meshgrid(indices_x, indices_y, indexing='ij')

        # Reshape into a single tensor
        xy = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        return xy, xy.shape[0]

    @time_decorator
    def get_predictions(self, img_tensor, points, point_labels):
        predicted_masks, predicted_iou = self.model(img_tensor.unsqueeze(0), points, point_labels)
        sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
        predicted_iou_scores = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
        predicted_masks = torch.take_along_dim(predicted_masks, sorted_ids[..., None, None], dim=2)
        predicted_masks = predicted_masks[0]
        iou = predicted_iou_scores[0, :, 0]
        index_iou = iou > 0.7
        iou_ = iou[index_iou]
        masks = predicted_masks[index_iou]
        score = calculate_stability_score(masks, 0.0, 1.0)
        score = score[:, 0]
        index = score > 0.9
        score_ = score[index]
        masks = masks[index]
        iou_ = iou_[index]
        masks = torch.ge(masks, 0.0)
        return masks, iou_

    @time_decorator
    def process_small_region(self, rles):
        new_masks = []
        scores = []
        min_area = 100
        nms_thresh = 0.7
        for rle in rles:
            mask = rle_to_mask(rle[0])

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have cha nged
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                rles[i_mask] = mask_to_rle_pytorch(mask_torch)
        masks = [rle_to_mask(rles[i][0]) for i in keep_by_nms]
        return masks

    @time_decorator
    def plot(self, masks):
        all_plot_num = len(masks)
        target_plot_num = len(self.mask_data['mask'])
        print("Number of all masks:", all_plot_num)
        print("Number of target masks:", target_plot_num)
        px = 1 / plt.rcParams['figure.dpi']

        num_rows, num_cols = 1, 2
        fig, ax = plt.subplots(num_rows, num_cols, figsize=(640 * px * num_cols, 360 * px * num_rows))
        ax = ax.flatten()

        annotated_image_dir = os.path.join(self.current_file_dir, f"../data/{self.timestamp}")
        os.makedirs(annotated_image_dir, exist_ok=True)
        # Save the image
        annotated_image_path = os.path.join(annotated_image_dir, 'annotated_image.png')
        plt.savefig(annotated_image_path, bbox_inches='tight', pad_inches=0)


        # Save the original image
        original_image_path = os.path.join(self.current_file_dir, f"../data/{self.timestamp}", 'original_image.png')
        cv2.imwrite(original_image_path, self.image)

        # Convert the annotated image to BGR format for saving with OpenCV
        annotated_image = self.image.copy()
        alpha_mask = img[:, :, 3]
        for c in range(3):
            annotated_image[:, :, c] = annotated_image[:, :, c] * (1 - alpha_mask) + img[:, :, c] * alpha_mask * 255

        annotated_image_path = os.path.join(self.current_file_dir, f"../data/{self.timestamp}", 'annotated_image.png')
        cv2.imwrite(annotated_image_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        # Optionally, display the plot if needed
        plt.show()

        # Optionally, close the figure if you are done with it
        plt.close(fig)
        #
        # if all_plot_num + 1 < num_rows * num_cols:
        #     for j in range(all_plot_num + 1, num_rows * num_cols):
        #         fig.delaxes(ax.flat[j])
        #
        # ax[0].imshow(self.image)
        # ax[0].title.set_text("Original")
        # ax[0].axis('off')
        #
        # plt.tight_layout()
        # fig.suptitle("Comparison of Original Image and EfficientSAM Segmentation")
        # plt.show()
        # extent = ax[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        # image_path = os.path.join(self.current_file_dir, f"../data/{self.timestamp}", 'segment_filtered.png')
        #
        # fig.savefig(image_path, bbox_inches=extent)

        if self.show_all:
            # Plot the whole image with aruco detected frames
            num_rows, num_cols = calculate_subplot_dimensions(all_plot_num + 1)
            fig, ax = plt.subplots(num_rows, num_cols, figsize=(640 * px * num_cols, 360 * px * num_rows))
            ax = ax.flatten()
            self.show_anns_separate(masks, ax)
            if all_plot_num + 1 < num_rows * num_cols:
                for j in range(all_plot_num + 1, num_rows * num_cols):
                    fig.delaxes(ax.flat[j])

            ax[0].imshow(self.image)
            ax[0].title.set_text("Original")
            ax[0].axis('off')

            for i in range(len(self.aruco_data['ids'])):
                centroid = self.aruco_data['centroid'][i]
                corners = self.aruco_data['corners'][i]
                centroid_circle = Circle((centroid[0], centroid[1]), radius=2, color='red', fill=True)
                ax[0].add_patch(centroid_circle)

                # Adding circles at corners
                for corner in corners:
                    circle = Circle((corner[0], corner[1]), radius=1, color='blue', fill=True)
                    ax[0].add_patch(circle)

            for i in self.target_area:
                ax[0].add_patch(Circle((i[0], i[1]), radius=10, color='red', fill=True))

            plt.tight_layout()
            fig.suptitle("Comparison of Original Image and EfficientSAM Segmentation")
            plt.show()

        if self.mask_data['mask']:
            num_cols = int(math.sqrt(target_plot_num))
            num_rows = (target_plot_num + num_cols - 1) // num_cols  # Calculate the number of rows needed
            fig, ax = plt.subplots(num_rows, num_cols, figsize=(640 * px * num_cols, 360 * px * num_rows))
            ax = np.array(ax).flatten()
            self.show_anns_separate(self.mask_data['mask'], ax)
            if target_plot_num < num_rows * num_cols:
                for j in range(target_plot_num, num_rows * num_cols):
                    fig.delaxes(ax.flat[j])
            plt.tight_layout()
            fig.suptitle("Target Masks")
            plt.show()

    @time_decorator
    def show_anns_separate(self, masks, ax):
        for idx, mask in enumerate(masks):
            if len(masks) != len(self.mask_data['mask']):
                idx += 1
            ax[idx].imshow(self.image)
            ax[idx].set_autoscale_on(False)
            img = np.ones((self.image_h, self.image_w, 4))
            img[:, :, 3] = 0
            color_mask = np.concatenate([[0, 1, 0], [1]])  # Red color with alpha 0.5
            # if idx == 8:
            #     color_mask = np.concatenate([[0, 1, 0], [1]])
            img[mask] = color_mask
            ax[idx].imshow(img)
            ax[idx].set_title(f"Masks {idx}")
            ax[idx].axis('off')

    @time_decorator
    def show_anns(self, masks, ax):
        ax[1].imshow(self.image)
        ax[1].set_autoscale_on(False)

        img = np.ones((masks[0].shape[0], masks[0].shape[1], 4))
        img[:, :, 3] = 0
        for i, ann in enumerate(masks):
            color_mask = np.concatenate([np.random.random(3), [0.9]])
            img[ann] = color_mask
        ax[1].imshow(img)
        ax[1].title.set_text("EfficientSAM")
        ax[1].axis('off')
        return img

    @time_decorator
    def analyze_masks(self, masks):
        print("Analyzing masks...")

    @time_decorator
    def execute(self):
        masks = self.segment()
        self.analyze_masks(masks)
        self.plot(masks)

        if len(self.mask_data['ids']) == 0:
            print("No masks found")
            return False
        else:
            print("Masks found")
            if self.maintain:
                data_path = os.path.join(self.current_file_dir, f"../data/{self.timestamp}", 'mask_data.json')
                with open(data_path, "w") as json_file:
                    json.dump(self.mask_data, json_file)
                print('Saved mask_data.json')
            return True

# Path where you want to save the file
