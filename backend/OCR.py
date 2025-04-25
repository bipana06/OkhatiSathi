import pytesseract
import re
import cv2

def perform_ocr(image_path, lang='eng', contrast=1.5, denoise=True):
    try:      
        # Fast image loading and preprocessing
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Quick contrast adjustment
        if contrast != 1:
            img = cv2.convertScaleAbs(img, alpha=contrast, beta=0)
        
        # Fast denoising (only if needed)
        if denoise:
            img = cv2.fastNlMeansDenoising(img, h=7, templateWindowSize=7, searchWindowSize=21)
        
        # Fast thresholding
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR with minimal config for speed
        text = pytesseract.image_to_string(
            img,
            config='--oem 1 --psm 6', 
            lang=lang
        )

        # Basic cleaning
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = text.replace('.', '')      # Remove all full stops
        
        return text
    
    except Exception as e:
        print("Error occured while trying to process the image. Please try again.")
        return f"OCR Error: {e}", 0

# if __name__ == "__main__":
#     image_path = "images/test7.png" 
    
#     text = ocr(image_path)
#     print(text)