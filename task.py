import cv2
import numpy as np

def detect_yellow_lines(image_path):
    image = cv2.imread(image_path)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    yellow_hues = np.arange(10, 35)
    yellow_mask = np.zeros_like(hsv_image[:, :, 0])

    for hue in yellow_hues:
        lower_yellow_hsv = np.array([hue - 10, 50, 100])
        upper_yellow_hsv = np.array([hue + 10, 255, 255])

        hue_mask = cv2.inRange(hsv_image, lower_yellow_hsv, upper_yellow_hsv)

        _, thresh_mask = cv2.threshold(hue_mask, 127, 255, cv2.ADAPTIVE_THRESH_MEAN_C)

        yellow_mask |= thresh_mask

    kernel = np.ones((5, 5), np.uint8)

    yellow_mask_opened = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    yellow_mask_closed = cv2.morphologyEx(yellow_mask_opened, cv2.MORPH_CLOSE, kernel)

    return yellow_mask_closed

def extract_ship_silhouette(image_path):
    image = cv2.imread(image_path)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_blue_hsv = np.array([90, 50, 50])
    upper_blue_hsv = np.array([130, 255, 255])

    blue_mask_hsv = cv2.inRange(hsv_image, lower_blue_hsv, upper_blue_hsv)

    kernel = np.ones((5, 5), np.uint8)
    
    water_mask = cv2.morphologyEx(blue_mask_hsv, cv2.MORPH_CLOSE, kernel)
    non_water_mask = 255 - water_mask

    result = cv2.bitwise_and(image, image, mask=non_water_mask)

    return result

if __name__ == "__main__":

    road_image_path = 'images/road.jpg'
    detected_yellow_lines = detect_yellow_lines(road_image_path)

    ship_image_path = 'images/ship.png'
    extracted_ship_silhouette = extract_ship_silhouette(ship_image_path)

    cv2.imshow('Road', detected_yellow_lines)
    cv2.imshow('Ship', extracted_ship_silhouette)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
