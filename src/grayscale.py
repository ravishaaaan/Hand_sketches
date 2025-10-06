import cv2

def convert_to_grayscale(image_path, save_path="datasets/gray_output.png"):
    """Convert an image to grayscale and save it."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(save_path, gray)
    return save_path
