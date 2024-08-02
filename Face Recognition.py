import cv2

img = cv2.imread('C:/Users/singh/Downloads/image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if not face_cascade.empty():
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)

font = cv2.FONT_HERSHEY_SIMPLEX
text = 'I am Manasi Singh'
font_scale = 1
font_thickness = 1
text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
text_x = 20
text_y = 30 + text_size[1]
cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

scale_percent = 20
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

cv2.imshow("Face Detection", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
