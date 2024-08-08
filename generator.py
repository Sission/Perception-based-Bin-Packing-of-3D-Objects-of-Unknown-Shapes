import cv2
import cv2.aruco as aruco
import numpy as np

# Parameters
marker_id = 3  # Marker ID
marker_size_mm = 40  # Marker size in 100 millimeters
margin_size_mm = 5  # Margin size in 10 millimeters
border_size_mm = 25  # Border size in millimeters
pixels_per_mm = 10  # Pixels per millimeter (adjust based on your requirements)

# Convert sizes from millimeters to pixels
marker_size = int(marker_size_mm * pixels_per_mm)
margin_size = int(margin_size_mm * pixels_per_mm)
border_size = int(border_size_mm * pixels_per_mm)

# Create dictionary and draw marker
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
marker_image = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

# Add white margin
marker_with_margin = np.ones((marker_size + margin_size * 2, marker_size + margin_size * 2), dtype=np.uint8) * 255
marker_with_margin[margin_size:marker_size + margin_size, margin_size:marker_size + margin_size] = marker_image

# Add black border
marker_with_border = np.ones((marker_with_margin.shape[0] + border_size * 2, marker_with_margin.shape[1] + border_size * 2), dtype=np.uint8) * 0
marker_with_border[border_size:border_size + marker_with_margin.shape[0], border_size:border_size + marker_with_margin.shape[1]] = marker_with_margin

# Display the marker with margin and border
cv2.imshow('ArUco Marker with Margin and Border', marker_with_border)
cv2.imwrite(f'aruco_border_id_{marker_id}.png', marker_with_border)  # Save marker image
cv2.waitKey(0)
cv2.destroyAllWindows()
