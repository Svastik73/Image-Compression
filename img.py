import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def compress_image(image_path, n_colors):
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 3)

    # Apply KMeans to find n_colors clusters
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Replace each pixel value with its corresponding center value
    compressed_pixels = centers[labels]
    compressed_image = compressed_pixels.reshape(image.shape).astype(np.uint8)

    return  compressed_image

# Parameters
image_path = 'lotus.jpg'  # Since the image is in the same directory
n_colors = int(input("Enter the number of clusters/colors: "))

# Compress the image
original_img=cv2.imread(image_path)  #fetch original image
compressed_image = compress_image(image_path, n_colors)

# Display the original and compressed images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Compressed Image with {n_colors} Colors")
plt.imshow(compressed_image)
plt.axis('off')

plt.show()
