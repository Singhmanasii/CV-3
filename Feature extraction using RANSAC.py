import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
img1_color = cv2.imread("C:/Users/singh/Downloads/cube 1.jpeg")
img2_color = cv2.imread("C:/Users/singh/Downloads/cubw 2.png")
img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
height, width = img2.shape

# Detect ORB keypoints and descriptors
orb_detector = cv2.ORB_create(5000)
kp1, d1 = orb_detector.detectAndCompute(img1, None)
kp2, d2 = orb_detector.detectAndCompute(img2, None)

# Match descriptors
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(d1, d2)
matches = sorted(matches, key=lambda x: x.distance)
matches = matches[:int(len(matches) * 0.9)]  # Keep 90% of the best matches
no_of_matches = len(matches)

# Draw matches to visualize
img_matches = cv2.drawMatches(img1_color, kp1, img2_color, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Show the matches
plt.figure(figsize=(20, 10))
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title("Feature Matches")
plt.axis('off')
plt.show()

# Extract the matched keypoints
p1 = np.zeros((no_of_matches, 2))
p2 = np.zeros((no_of_matches, 2))
for i in range(len(matches)):
    p1[i, :] = kp1[matches[i].queryIdx].pt
    p2[i, :] = kp2[matches[i].trainIdx].pt

# Compute the homography matrix using RANSAC
homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)  # Adjust the RANSAC threshold here (5.0 is a common value)

# Apply the homography to the original image
transformed_img = cv2.warpPerspective(img1_color, homography, (width, height))

# Save and display the transformed image
cv2.imwrite('output.jpg', transformed_img)
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
axs[0].imshow(cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original Image')
axs[0].axis('off')
axs[1].imshow(cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB))
axs[1].set_title('Transformed Image')
axs[1].axis('off')
plt.show()
