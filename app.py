from flask import Flask, render_template, Response, request
import cv2
import numpy as np

app = Flask(__name__, static_folder='static')

try:
    net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def generate_frames(detect_faces, detect_objects):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        h, w = resized_frame.shape[:2]

        if detect_faces:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if detect_objects:
            blob = cv2.dnn.blobFromImage(resized_frame, 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.3:  # Adjust confidence threshold as needed (e.g., 0.3, 0.4, 0.5)
                    class_id = int(detections[0, 0, i, 1])
                    label = CLASSES[class_id]
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(resized_frame, (startX, startY), (endX, endY), COLORS[class_id], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(resized_frame, label + ": {:.2f}%".format(confidence * 100), (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[class_id], 2)

        ret, buffer = cv2.imencode('.jpg', resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    detect_faces = request.args.get('detect_faces', default=False, type=lambda v: v.lower() == 'true')
    detect_objects = request.args.get('detect_objects', default=False, type=lambda v: v.lower() == 'true')
    return Response(generate_frames(detect_faces, detect_objects), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
