import cv2
import os
from datetime import datetime

# Step 1: Initialize the HOG descriptor/person detector
print("Initializing HOG descriptor...")
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
print("HOG descriptor initialized successfully.\n")

# Step 2: Load the input image
image_path = 'C:/Users/singh/Downloads/people_walking.jpg'
print(f"Loading image from {image_path}...")
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print("Error: Image not found or unable to load!")
    exit()

print("Image loaded successfully.\n")

# Step 3: Resize the image for faster processing (without imutils)
print("Resizing image...")
original_height, original_width = image.shape[:2]
resize_width = min(800, original_width)
resize_height = int(original_height * (resize_width / original_width))
image = cv2.resize(image, (resize_width, resize_height))
print(f"Image resized to {resize_width}x{resize_height} pixels.\n")

# Step 4: Detect people in the image
print("Detecting pedestrians...")
regions, _ = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)

# Step 5: Draw bounding boxes around detected people
print(f"Number of pedestrians detected: {len(regions)}")
for i, (x, y, w, h) in enumerate(regions):
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    print(f"Pedestrian {i+1}: Location [x={x}, y={y}, width={w}, height={h}]")

print("\nAll pedestrians have been marked with bounding boxes.\n")

# Step 6: Save the output image
output_dir = 'C:/Users/singh/Downloads/'
output_filename = f"pedestrian_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
output_path = os.path.join(output_dir, output_filename)
cv2.imwrite(output_path, image)
print(f"Output image saved as {output_filename} in {output_dir}.\n")

# Step 7: Display the output image with bounding boxes
cv2.imshow("Pedestrian Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Script execution completed successfully.")
