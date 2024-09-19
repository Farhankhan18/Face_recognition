

Face Recognition System Using OpenCV
This project implements a simple Face Recognition System using OpenCV in Python. The system captures face images from a live camera feed, trains an LBPH (Local Binary Patterns Histogram) face recognizer model, and performs real-time face recognition.

Features:
Face Detection with Haar Cascade:

Uses Haar Cascade Classifier (haarcascade_frontalface_default.xml), a pre-trained model provided by OpenCV, to detect faces in a live video stream.
The detected faces are captured and stored for each person to be used for training the recognition model.
Capture Faces:

Captures face images from the camera for a predefined list of people.
Stores 10 face images per person.
Displays a bounding box around each detected face and allows capturing the face using the 'c' key or skipping the person with the 'q' key.
Train the LBPH Face Recognizer:

After capturing face data, the system trains the LBPH face recognizer using the captured images.
LBPH is effective in face recognition due to its use of local features and grayscale images.
Real-time Face Recognition:

Recognizes faces from a live camera feed in real-time using the trained LBPH model.
Displays the recognized person's name or labels the person as "Unknown" if the confidence level is above a set threshold.
Draws bounding boxes around detected faces and labels them.
Requirements:
OpenCV 4.10.0 or higher
NumPy
How to Run:
Install the necessary libraries:

pip install opencv-python numpy
Capture Faces and Train the Model:

Run the script to start capturing images from the camera.
Use 'c' to capture face images and 'q' to skip to the next person.
Once enough images are captured, the LBPH model will be trained on the data.
Face Recognition:

After training, the system will switch to face recognition mode and display names of recognized people or "Unknown" for unrecognized faces.
Files:
main.py: Core logic for face detection, capturing, training, and recognition.
Haar Cascade XML File: The project uses OpenCV's pre-trained Haar Cascade Classifier (haarcascade_frontalface_default.xml) to detect faces.
Improvements and Customization:
Add more people by modifying the person_names list.
Tune the detection and recognition parameters, such as the confidence threshold and face image size.
Extend the system to support different classifiers or face recognition algorithms.
Usage Example:

During execution, press 'c' to capture a photo or 'q' to skip. After capturing and training, the system will automatically switch to real-time face recognition.

This version now properly highlights the use of the Haar Cascade Classifier for face detection, which is an important aspect of the system.



