# %%
import cv2
import sys
import numpy as np
from matplotlib import pyplot

# %%
def create_mask(img_hsv):
    # lower mask (0-10)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    # find the colors within the boundaries
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    # find the colors within the boundaries
    mask1 = cv2.inRange(img_hsv, lower_red2, upper_red2)

    # join masks
    mask = mask0 + mask1

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask

# %%
def add_centeroid(frame, mask):
    # Segment only the detected region
    segmented_img = cv2.bitwise_and(frame, frame, mask=mask)

    # convert image to grayscale image
    gray_image = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)

    # convert the grayscale image to binary image
    ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Moment obrazu
    M = cv2.moments(thresh)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # put text and highlight the center
    cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(frame, "red ball", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

# %%
def find_color(frame):
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = create_mask(img_hsv)

    #calculate and add centroid
    frame = add_centeroid(frame, mask)

    # Find contours from the mask
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contour on image
    output = cv2.drawContours(img_hsv.copy(), contours, -1, (0, 0, 255), 3)

    return output


# %%
cap = cv2.VideoCapture('Files/movingball.mp4')

if not cap.isOpened():
    print("Error opening video file")

while cap.isOpened():
    cv2.startWindowThread()
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    det_frame = find_color(frame)
    cv2.imshow('frame, click q to quit', det_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


