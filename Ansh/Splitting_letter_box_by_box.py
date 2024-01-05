from PIL import Image
import pytesseract
import cv2
import numpy as np

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Open the image file
im = Image.open('Remove Background (1).jpg')

# Use pytesseract to get bounding boxes for each character
boxes_data = pytesseract.image_to_boxes(im)
boxes = [box.split() for box in boxes_data.splitlines()]

# Convert the image to a numpy array
im_array = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

# Loop through each bounding box
for box in boxes:
    x, y, w, h = int(box[1]), int(box[2]), int(box[3]), int(box[4])

    # Crop the character from the image
    char_image = im_array[y:h, x:w]

    # Create a PIL image from the cropped character
    char_image_pil = Image.fromarray(cv2.cvtColor(char_image, cv2.COLOR_BGR2RGB))

    # Save the character to a file
    char_image_pil.save(f'char_{x}_{y}.png')

print("Character extraction complete")
