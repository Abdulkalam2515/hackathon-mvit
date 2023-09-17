import cv2
from flask import Flask, render_template, Response

# Initialize Flask app
app = Flask(__name__)

# Initialize the video capture (0 for default camera)
cap = cv2.VideoCapture(0)

# Initialize variables
previous_frame = None
motion_detected = False

@app.route('/')
def index():
    return render_template('index.html')

# Create a generator function to yield frames
def generate_frames():
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if previous_frame is None:
            previous_frame = gray_frame
            continue

        # Compute the absolute difference between the current and previous frame
        frame_delta = cv2.absdiff(previous_frame, gray_frame)

        # Apply a threshold to the frame delta to create a binary image
        threshold = 30  # You can adjust this threshold value
        _, thresh_frame = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate over the contours to detect motion
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # You can adjust this area threshold
                motion_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Convert the processed frame to bytes for displaying in the web page
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
