import numpy as np
# required to plot a representation of Bin and contained items
from matplotlib.patches import Rectangle, Circle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


def rotation_matrix(cos_value, sin_value):
    return [
        [cos_value, -sin_value],
        [sin_value, cos_value]
    ]


class Painter:
    def __init__(self, pgrp, box):
        ''' '''
        self.bin_groups = pgrp
        self.x_min, self.x_max = 0, 0
        self.y_min, self.y_max = 0, 0
        self.box = box
        self.box_height = 55000000

    def construct_object(self, item):
        bottom_shape = item.transformed_shape().contour

        bottom_shape_coords = [(point.x, point.y) for point in bottom_shape]
        height = 50000000  # Height of the object scaled down
        top_shape = [(point.x, point.y, height) for point in bottom_shape]

        vertices = []
        for i in range(len(bottom_shape_coords)):
            side = [
                (bottom_shape_coords[i][0], bottom_shape_coords[i][1], 0),
                (bottom_shape_coords[(i + 1) % len(bottom_shape_coords)][0],
                 bottom_shape_coords[(i + 1) % len(bottom_shape_coords)][1], 0),
                (top_shape[(i + 1) % len(bottom_shape_coords)][0],
                 top_shape[(i + 1) % len(bottom_shape_coords)][1], height),
                (top_shape[i][0], top_shape[i][1], height)
            ]
            vertices.append(side)

        side_vertices = vertices[:len(bottom_shape_coords)]
        top_bottom_vertices = [
            [(point[0], point[1], 0) for point in bottom_shape_coords],
            [(point[0], point[1], height) for point in top_shape]
        ]

        return side_vertices, top_bottom_vertices

    def create_object_collections(self, side_vertices, top_bottom_vertices, color, edge_width=0.5, alpha=0.2):
        side_collection = Poly3DCollection(side_vertices, color=color, linewidths=edge_width, alpha=alpha)
        side_collection.set_facecolor(color)

        top_bottom_collection = Poly3DCollection(top_bottom_vertices, color=color, linewidths=edge_width, alpha=0.2)
        top_bottom_collection.set_facecolor(color)

        centroid = [
            sum(v[dim] for side in side_vertices for v in side) / (len(side_vertices) * len(side_vertices[0]))
            for dim in range(3)]

        return side_collection, top_bottom_collection, centroid

    def box_vertices(self, min_vals):
        box_bottom_shape = [
            (min_vals[0], min_vals[1]),
            (min_vals[0] + self.box.width, min_vals[1]),
            (min_vals[0], min_vals[1] + self.box.height),
            (min_vals[0] + self.box.width, min_vals[1] + self.box.height)
        ]

        # Define the height of the box
        height = self.box_height

        # Create the vertices for the sides of the box
        box_vertices = [
            [
                (box_bottom_shape[0][0], box_bottom_shape[0][1], 0),
                (box_bottom_shape[1][0], box_bottom_shape[1][1], 0),
                (box_bottom_shape[1][0], box_bottom_shape[1][1], height),
                (box_bottom_shape[0][0], box_bottom_shape[0][1], height)
            ],
            [
                (box_bottom_shape[0][0], box_bottom_shape[0][1], 0),
                (box_bottom_shape[2][0], box_bottom_shape[2][1], 0),
                (box_bottom_shape[2][0], box_bottom_shape[2][1], height),
                (box_bottom_shape[0][0], box_bottom_shape[0][1], height)
            ],
            [
                (box_bottom_shape[2][0], box_bottom_shape[2][1], 0),
                (box_bottom_shape[3][0], box_bottom_shape[3][1], 0),
                (box_bottom_shape[3][0], box_bottom_shape[3][1], height),
                (box_bottom_shape[2][0], box_bottom_shape[2][1], height)
            ],
            [
                (box_bottom_shape[1][0], box_bottom_shape[1][1], 0),
                (box_bottom_shape[3][0], box_bottom_shape[3][1], 0),
                (box_bottom_shape[3][0], box_bottom_shape[3][1], height),
                (box_bottom_shape[1][0], box_bottom_shape[1][1], height)
            ],
            [
                (box_bottom_shape[0][0], box_bottom_shape[0][1], 0),
                (min_vals[0], min_vals[1], 0),  # Adding the line connecting (min_x, min_y) to the box
                (min_vals[0], min_vals[1], height),
                (box_bottom_shape[1][0], box_bottom_shape[1][1], height)
            ]
        ]
        return box_vertices

    def plot_object(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Initialize min and max values for x, y, and z
        min_vals = [float('inf'), float('inf'), float('inf')]
        max_vals = [float('-inf'), float('-inf'), float('-inf')]

        colors = ['cyan', 'magenta', 'yellow', 'green', 'blue']  # Add more colors if needed
        color_index = 0

        for bin_group in self.bin_groups:  # Assuming PackGroup is iterable
            for i, item in enumerate(bin_group):  # Assuming each bin group is also iterable
                side_vertices, top_bottom_vertices = self.construct_object(item)

                side_col, top_bottom_col, centroid = self.create_object_collections(side_vertices, top_bottom_vertices,
                                                                                    colors[color_index % len(colors)],
                                                                                    edge_width=0.2, alpha=0.1)
                ax.add_collection3d(side_col)
                ax.add_collection3d(top_bottom_col)
                ax.text(*centroid, str(i), color='black', ha='center', va='center')

                color_index += 1

                # Calculate the centroid of the object for placing text

                # Update min and max values for x, y, and z based on current vertices
                for dim in range(3):
                    coords = [v[dim] for side in side_vertices for v in side]
                    min_vals[dim] = min(min_vals[dim], min(coords))
                    max_vals[dim] = max(max_vals[dim], max(coords))

        # Plot the box
        box_vertices = self.box_vertices(min_vals)
        box_edge_collection = Line3DCollection(box_vertices, colors='black')
        ax.add_collection3d(box_edge_collection)

        # Update min and max values for x, y, and z based on box vertices
        box_coords = [v[dim] for side in box_vertices for v in side for dim in range(3)]
        for dim in range(3):
            min_vals[dim] = min(min_vals[dim], min(box_coords[dim::3]))
            max_vals[dim] = max(max_vals[dim], max(box_coords[dim::3]))

        # Set axes labels and limits
        ax.set_title('3D bin nesting')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        ax.set_xlim([min_vals[0], max_vals[0]])
        ax.set_ylim([min_vals[1], max_vals[1]])
        ax.set_zlim([min_vals[2], max_vals[2]])

        plt.show()
