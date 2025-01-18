# Real-Time Webcam Object and Face Detection

This project demonstrates real-time object and face detection using a webcam, powered by OpenCV and a pre-trained MobileNet SSD model. It provides a web interface built with Flask for easy interaction.

## Features

*   Real-time object detection using MobileNet SSD.
*   Real-time face detection using Haar cascades.
*   Interactive web interface with Flask.
*   Toggle face and object detection on/off via checkboxes.
*   Futuristic and engaging user interface.

## Technologies Used

*   Python
*   Flask (web framework)
*   OpenCV (computer vision library)
*   NumPy (numerical computing)
*   HTML, CSS, JavaScript (front-end)
*   MobileNet SSD (object detection model)
*   Haar Cascades (face detection)


1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Model Files:**

    Download the `MobileNetSSD_deploy.prototxt` and `MobileNetSSD_deploy.caffemodel` files. You can find them in various online repositories or tutorials related to MobileNet SSD. Place these files in the root directory of your project (the same directory as `app.py`).

3.  **Run the Flask app:**

    ```bash
    python app.py
    ```

5.  **Open in Browser:**

    Open your web browser and go to `http://127.0.0.1:5000/`.

## Deployment (If Applicable)

[If you have deployed your app, add deployment instructions here. For example, for PythonAnywhere:]

This application cannot be fully deployed to static hosting services like GitHub Pages because it requires a Python backend. For online hosting with the webcam functionality, consider services like PythonAnywhere, Heroku, or other cloud platforms.

## Project Structure

```
webcam-object-face-detection/
├── app.py              # Flask application
├── templates/
│   └── index.html      # HTML template
├── static/
│   └── images/
│       └── background.jpg # Background image (optional)
├── MobileNetSSD_deploy.prototxt
├── MobileNetSSD_deploy.caffemodel
├── requirements.txt
└── README.md
```

## How It Works

This application uses the following process:

1.  The user opens the web page in their browser.
2.  The browser makes a request to the Flask server.
3.  The Flask server renders the `index.html` template, which contains the webcam display and controls.
4.  When the page loads, JavaScript makes a request to the `/video_feed` route.
5.  The `/video_feed` route calls the `generate_frames` function.
6.  `generate_frames` captures frames from the webcam using OpenCV.
7.  If face detection is enabled, Haar cascades are used to detect faces in the frames.
8.  If object detection is enabled, the MobileNet SSD model is used to detect objects.
9.  The processed frames are encoded as JPEG images and sent back to the browser as a multipart stream.
10. The browser displays the streamed images, creating the real-time video feed.

## Credits

Developed by CHOLLANGI YUVARAJ (yuvarajprince341091@gmail.com) and DASARI VINAYBABU
