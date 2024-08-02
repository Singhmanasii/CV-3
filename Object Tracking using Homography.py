import cv2
import numpy as np

# Load the image and video
img = cv2.imread("C:/Users/singh/Downloads/fd img.jpg", cv2.IMREAD_GRAYSCALE)
cap = cv2.VideoCapture("C:/Users/singh/Downloads/fd.mp4")

if img is None or not cap.isOpened():
    print("Error loading image or video.")
    exit()

# Initialize SIFT and FLANN
sift = cv2.SIFT_create()
kp_img, desc_img = sift.detectAndCompute(img, None)
flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), {})

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, desc_frame = sift.detectAndCompute(gray_frame, None)
    matches = flann.knnMatch(desc_img, desc_frame, k=2)

    good_points = [m for m, n in matches if m.distance < 0.8 * n.distance]
    if len(good_points) > 10:
        src_pts = np.float32([kp_img[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = img.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        frame = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)

    # Display results
    all_matches = cv2.drawMatches(img, kp_img, frame, kp_frame, [m[0] for m in matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Matches', all_matches)
    cv2.imshow('Homography', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
