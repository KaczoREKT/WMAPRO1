# %%
import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt

# %%
def print_image(image, title=""):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title(title)
    plt.show()

# %%
# Wczytanie obrazu
img = cv2.imread("Files/ball.png")
if img is None:
    sys.exit("Could not read the image.")
print_image(img, "Original Image")

# %%
# Konwersja do przestrzeni HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Dolna maska dla czerwonego koloru (0-10)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)

# Górna maska dla czerwonego koloru (170-180)
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])
mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)

# Połączenie obu masek
mask = mask1 + mask2
print_image(mask, "Red Color Mask")

# %%
# Usunięcie szumów
kernel = np.ones((7, 7), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
print_image(mask, "Mask after Noise Removal")

# %%
# Segmentacja – zastosowanie maski na oryginalnym obrazie
segmented_img = cv2.bitwise_and(img, img, mask=mask)
print_image(segmented_img, "Segmented Image")

# %%
# Konwersja obrazu na skalę szarości
gray_image = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
print_image(gray_image, "Grayscale Image")

# %%
# Konwersja obrazu na obraz binarny
ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print_image(thresh, "Threshold Image")

# %%
# Obliczanie momentów obrazu
M = cv2.moments(thresh)
if M["m00"] != 0:
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
else:
    cX, cY = 0, 0

# %%
# Rysowanie środka masy
cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
cv2.putText(img, "red ball", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
print_image(img, "Image with Center Marked")

# %%
# Znajdowanie konturów
contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Rysowanie konturów
output = cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), 3)
print_image(output, "Contours")
