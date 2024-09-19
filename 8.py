import cv2
import numpy as np

# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# List of person names
person_names = ["Farhan", "Siddu", "Ratan"]

# Initialize data and labels
faces_data = []
labels = []

# Define a fixed size for the face images
FACE_SIZE = (100, 100)

def capture_and_store_faces():
    # Try accessing the default camera with index 0
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera couldn't be opened. Please check your camera or permissions.")
        return
    
    for idx, person_name in enumerate(person_names):
        print(f"Capturing images for {person_name}. Press 'c' to capture a photo, and 'q' to skip.")

        captured_faces = 0
        while captured_faces < 10:  # Capture 10 face images for each person
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from camera.")
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_roi = gray[y:y + h, x:x + w]
                face_roi_resized = cv2.resize(face_roi, FACE_SIZE)

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.imshow('Capturing Faces', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    faces_data.append(face_roi_resized)
                    labels.append(idx)
                    captured_faces += 1
                    print(f"Captured {captured_faces} images for {person_name}.")
                elif key == ord('q'):
                    print(f"Skipping {person_name}.")
                    break

            if captured_faces >= 10:
                break

    cap.release()
    cv2.destroyAllWindows()

# Step 1: Capture faces
capture_and_store_faces()

# Convert the labels to a numpy array
labels = np.array(labels)

# Train the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces_data, labels)

# Set a confidence threshold for recognizing unknown faces
THRESHOLD = 80

def recognize_faces_in_video():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera couldn't be opened for face recognition.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y + h, x:x + w]
            face_roi_resized = cv2.resize(face_roi, FACE_SIZE)

            # Recognize face
            label, confidence = recognizer.predict(face_roi_resized)

            if confidence < THRESHOLD:
                match_name = person_names[label]
            else:
                match_name = "Unknown"

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, match_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Step 2: Recognize faces from video
recognize_faces_in_video()
