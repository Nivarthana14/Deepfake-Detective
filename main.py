import sys
import cv2

def load_image(image_path):
    # Read image
    img = cv2.imread(image_path)

    # Check if image exists
    if img is None:
        print("❌ Error: Image not found or invalid path")
        return

    # Print success info
    print("✅ Image loaded successfully!")
    print("📏 Image Shape:", img.shape)

if __name__ == "__main__":
    # Check if user gave input
    if len(sys.argv) < 2:
        print("⚠️ Usage: python main.py <image_path>")
    else:
        image_path = sys.argv[1]
        load_image(image_path)
        