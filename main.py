import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Load the image and convert to RGB ---
image = cv2.imread(r'C:\Users\DELL\OneDrive\Desktop\Filter App\Images\CR7.jpeg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --- Rotation ---
center = (image_rgb.shape[1] // 2, image_rgb.shape[0] // 2)
angle = 30
scale = 1
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
rotated_image = cv2.warpAffine(image_rgb, rotation_matrix, (image_rgb.shape[1], image_rgb.shape[0]))

# --- Translation ---
tx, ty = 100, 70
translation_matrix = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
translated_image = cv2.warpAffine(image_rgb, translation_matrix, (image_rgb.shape[1], image_rgb.shape[0]))

# --- Canny Edge Detection ---
edges = cv2.Canny(image_rgb, 100, 700)

# --- Strong Gaussian Blur ---
blurred = cv2.GaussianBlur(image, (51, 51), 0)
blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

# --- Cartoon Effect ---
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_blurred = cv2.medianBlur(gray, 5)
edges_cartoon = cv2.adaptiveThreshold(gray_blurred, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 9, 9)
color = cv2.bilateralFilter(image, 9, 300, 300)
cartoon = cv2.bitwise_and(color, color, mask=edges_cartoon)
cartoon_rgb = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)

# --- Plotting all results ---
fig, axs = plt.subplots(2, 3, figsize=(15, 8))

axs[0, 0].imshow(image_rgb)
axs[0, 0].set_title('Original Image')

axs[0, 1].imshow(rotated_image)
axs[0, 1].set_title('Rotated Image')

axs[0, 2].imshow(translated_image)
axs[0, 2].set_title('Translated Image')

axs[1, 0].imshow(edges, cmap='gray')
axs[1, 0].set_title('Edge Detection')

axs[1, 1].imshow(blurred_rgb)
axs[1, 1].set_title('Blurred Image')

axs[1, 2].imshow(cartoon_rgb)
axs[1, 2].set_title('Cartoon Effect')

# --Remove axis ticks--
for ax in axs.flatten():
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
