import cv2
from ultralytics import YOLO

model = YOLO('best_people_detector.pt')

video_url = "http://10.217.210.229:8080/video"

cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    print(f"Error: Could not open video stream at {video_url}")
    print("Please check if the URL is correct and if your phone and computer are on the same Wi-Fi network.")
    exit()

print("Successfully connected to camera. Press 'q' on the keyboard to quit.")

while True:
    # Read one frame from the video stream.
    success, frame = cap.read()

    # If a frame was successfully read...
    if success:
        # Run YOLOv8 inference on the frame to detect objects.
        results = model(frame)

        # Visualize the results on the frame.
        # This draws the bounding boxes and labels on the image.
        annotated_frame = results[0].plot()

        # Display the annotated frame in a window.
        cv2.imshow("YOLOv8 Live People Detection", annotated_frame)

        # Wait for 1 millisecond, and break the loop if the 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # If the stream ends or a frame can't be read, break the loop.
        print("Video stream ended or frame could not be read.")
        break

# Clean up: release the video capture object and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()