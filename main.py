import cv2
import numpy as np
from waste_classifier import getPrediction  # Import your prediction function

# Initialize the video feed from the phone camera (replace with your phone IP and port)
video_url = 'http://100.91.248.9:8080/video'  # Change this to your phone's video feed URL
cap = cv2.VideoCapture(video_url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from camera.")
        break

    # Resize the frame to 180x180 for your model input
    resized_frame = cv2.resize(frame, (180, 180))

    # Convert the frame to a file format (since getPrediction expects a file path)
    # For real-time processing, save the image temporarily
    temp_img_path = '/Users/abhisheksingal/PycharmProjects/MCP100 Project/resources/tmp_frame.jpg'
    cv2.imwrite(temp_img_path, resized_frame)

    # Get prediction from the temporary file
    label, confidence, _ = getPrediction(temp_img_path)
    if label == "Recycle":
        label = "Dry Waste"
    if label == "Organic":
        label = "Wet Waste"


    # Change font color and size
    font_color = (0, 255, 242)  # Blue color (BGR format)
    font_size = 1               # Font size (adjust as necessary)
    thickness = 2               # Line thickness

    # Display the prediction on the video feed
    cv2.putText(frame, f"{label}: {confidence}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, thickness)

    # Show the video feed with predictions
    cv2.imshow("Waste Classification", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
