import cv2
import os

# Create a folder to save images if it doesn't exist
save_folder = "raw_images"
os.makedirs(save_folder, exist_ok=True)

cap = cv2.VideoCapture(0)  # 0 is usually the default laptop webcam
count = 0

print("Press 's' to save an image. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Data Collection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        # Save the image when 's' is pressed
        img_name = os.path.join(save_folder, f"sign_{count}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"Saved {img_name}")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()