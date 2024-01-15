import cv2
import numpy as np

# Initialize the video capture object (you can replace 0 with your video file path)
# cap = cv2.VideoCapture('car2.mp4')
# cap = cv2.VideoCapture('cars7.webm')
# cap = cv2.VideoCapture('video.mp4')
cap = cv2.VideoCapture('People.mp4')


# Parameters
threshold_value = 25  # Adjust this threshold as needed
min_contour_area = 50  # Adjust this threshold as needed

# Initialize the AKAZE detector
akaze = cv2.AKAZE_create()

# Initialize the background model with the first frame
ret, prev_frame = cap.read()
if not ret:
    exit()

# Convert the first frame to grayscale

prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Create an empty mask for drawing key points
mask = np.zeros_like(prev_frame)

# Initialize a counter for the number of moving objects
moving_objects_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # frame = cv2.resize(frame,(600,500))
    # Convert the current frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the pixel-wise absolute difference (energy function)
    energy_function = cv2.absdiff(prev_frame_gray, frame_gray)

    # Apply thresholding to obtain a binary mask
    threshold_value = 25  # Adjust this threshold as needed
    _, binary_mask = cv2.threshold(energy_function, threshold_value, 255, cv2.THRESH_BINARY)

    binary_mask = cv2.dilate(binary_mask, None, iterations=6)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    moving_objects_count = 0

    # Draw rectangles around detected objects
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:  # Adjust this threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'motion detected', (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            moving_objects_count += 1

    # Display the original frame with rectangles around objects
    cv2.imshow('Moving Object Detection', frame)
    cv2.imshow('binary image', binary_mask)

    # Update the previous frame
    prev_frame_gray = frame_gray.copy()

    # Press 'q' to exit the loop
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Print the number of moving objects
print("Number of Moving Objects:", moving_objects_count)

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
