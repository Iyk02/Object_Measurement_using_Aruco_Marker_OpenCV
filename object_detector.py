import cv2
import numpy as np

class HomogeneousBgDetector():
    def __init__(self):
        pass

    def aruco(self, img):
        # Load Aruco detector
        parameters = cv2.aruco.DetectorParameters_create()
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)

        # Get Aruco marker
        corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

        # Draw polygon around the marker
        int_corners = np.int0(corners)
        cv2.polylines(img, int_corners, True, (0, 100, 255), 5)

        # Aruco Perimeter
        aruco_perimeter = cv2.arcLength(corners[0], True)
        print(aruco_perimeter)

        # Pixel to cm ratio
        pixel_cm_ratio = aruco_perimeter / 20

        return pixel_cm_ratio

    def detect_objects(self, frame):
        # Convert Image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a Mask with adaptive threshold
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 5)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #cv2.imshow("mask", mask)
        objects_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 4000:
                #cnt = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
                objects_contours.append(cnt)

        return objects_contours
