from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

def shift_image(img, depth_img, shift_amount=10):
    img = img.convert("RGBA")
    depth_img = ImageOps.grayscale(depth_img).resize(img.size, Image.Resampling.LANCZOS)
    
    data = np.array(img)
    depth_data = np.array(depth_img)
    
    shifted_data = np.zeros_like(data)
    height, width = data.shape[:2]
    
    for y in range(height):
        for x in range(width):
            shift = int((depth_data[y, x] / 255.0) * shift_amount)
            new_x = min(max(x + shift, 0), width - 1)
            shifted_data[y, new_x] = data[y, x]
    
    shifted_image = Image.fromarray(shifted_data)
    return shifted_image

# Load images
img = Image.open("C:/Users/singh/Downloads/cube 1.jpeg")
depth_img = Image.open("C:/Users/singh/Downloads/cubw 2.png")

# Apply the shift
shifted_img = shift_image(img, depth_img, shift_amount=10)

# Display images
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(img)
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(shifted_img)
axs[1].set_title('Shifted Image')
axs[1].axis('off')

plt.show()
