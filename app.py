from flask import Flask, render_template, Response, request
import cv2

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

def generate_frames(detect_faces, detect_objects):  # Correct parameters!
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
            # ... (rest of object detection code)

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
