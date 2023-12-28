import cv2
import pytesseract
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('image sample.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to get binary image
_, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Set the Tesseract path (Windows only)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Extracted letters and their coordinates
extracted_letters = []

for contour in contours:
    # Get bounding box for each contour
    x, y, w, h = cv2.boundingRect(contour)

    # Extract each letter from the bounding box
    letter_image = gray_image[y:y+h, x:x+w]

    # Use Tesseract to do OCR on the letter image
    letter_text = pytesseract.image_to_string(letter_image, config='--psm 10 --oem 3')

    # Filter out non-alphabetic characters
    letter_text = ''.join(filter(str.isalpha, letter_text))

    # Save the letter and its bounding box coordinates
    if letter_text:
        extracted_letters.append({'letter': letter_text, 'coordinates': (x, y, x+w, y+h)})
        
        # Draw a green rectangle around the letter in the original image
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Write the detected letter on the image
        cv2.putText(image, letter_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save the image with bounding boxes and detected letters
cv2.imwrite('output_image_with_letters.jpg', image)

# Display the image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Image with Bounding Boxes and Detected Letters')
plt.show()

# Print extracted letters and their coordinates
print(extracted_letters)
