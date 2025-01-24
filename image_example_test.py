import cv2
import pytesseract
from PIL import Image
import numpy as np

def detect_word_in_image(image_path, target_word):
    """
    Detect and return bounding boxes of a specific word in an image.
    
    Args:
        image_path (str): Path to the input image
        target_word (str): Word to search for in the image
    
    Returns:
        list: List of bounding boxes (x, y, width, height) for the word's occurrences
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return []
        
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply some preprocessing to improve OCR accuracy
    # You can adjust these steps based on your specific image characteristics
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray, 3)
    
    # Use Tesseract to do OCR and get detailed information
    custom_config = r'--oem 3 --psm 11'
    try:
        ocr_details = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractError as e:
        print(f"Tesseract Error: {e}")
        return []
    
    # Print OCR results for debugging
    print("Detected words and their confidence scores:")
    for i, word in enumerate(ocr_details['text']):
        if word.strip():  # Only print non-empty words
            print(f"Word: '{word}', Confidence: {ocr_details['conf'][i]}")
    
    # Find bounding boxes for the target word
    word_boxes = []
    for i, word in enumerate(ocr_details['text']):
        # Convert to lowercase for case-insensitive matching
        if word.lower() == target_word.lower():
            # Check confidence to filter out low-quality detections
            if ocr_details['conf'][i] > 60:  # Confidence threshold
                x = ocr_details['left'][i]
                y = ocr_details['top'][i]
                w = ocr_details['width'][i]
                h = ocr_details['height'][i]
                word_boxes.append((x, y, w, h))
    
    return word_boxes

def visualize_word_boxes(image_path, word_boxes):
    """
    Visualize the detected word bounding boxes on the original image.
    
    Args:
        image_path (str): Path to the input image
        word_boxes (list): List of bounding boxes to draw
    
    Returns:
        numpy.ndarray: Image with bounding boxes drawn
    """
    # Read the image
    img = cv2.imread(image_path)
    
    # Draw rectangles for each bounding box
    for (x, y, w, h) in word_boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return img

# Example usage
def main():
    # Path to your input image
    image_path = 'sample4.jpg'
    
    # Word to search for
    target_word = 'bounding'
    
    # Detect word bounding boxes
    word_boxes = detect_word_in_image(image_path, target_word)
    
    # Print detected bounding boxes
    print(f"Found {len(word_boxes)} occurrences of '{target_word}':")
    for box in word_boxes:
        print(f"Bounding Box: x={box[0]}, y={box[1]}, width={box[2]}, height={box[3]}")
    
    # Optionally visualize the results
    if word_boxes:
        result_image = visualize_word_boxes(image_path, word_boxes)
        cv2.imshow('Word Detections', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()