import cv2
import dlib
from scipy.spatial import distance as dist

# Load Haar cascade for face detection and dlib's shape predictor for facial landmarks
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Path to the dlib facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Eye aspect ratio threshold for blink detection
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 2
blink_counter = 0
blinks = 0

def eye_aspect_ratio(eye):
    # Compute the distances between the vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Compute the distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Define eye landmark indices (using dlib's 68-point model)
LEFT_EYE_LANDMARKS = list(range(36, 42))
RIGHT_EYE_LANDMARKS = list(range(42, 48))

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray, 0)

    for face in faces:
        # Predict facial landmarks
        shape = predictor(gray, face)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        # Extract the left and right eye landmarks
        left_eye = [shape[i] for i in LEFT_EYE_LANDMARKS]
        right_eye = [shape[i] for i in RIGHT_EYE_LANDMARKS]

        # Calculate the eye aspect ratio for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Draw the eyes on the frame
        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Check if EAR is below threshold to detect a blink
        if ear < EYE_AR_THRESH:
            blink_counter += 1
        else:
            # If a blink was detected, count it
            if blink_counter >= EYE_AR_CONSEC_FRAMES:
                blinks += 1
                cv2.putText(frame, "Blink Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            blink_counter = 0

    # Determine if face is real or fake based on blink count
    if blinks > 1:
        cv2.putText(frame, "Real Face Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No Blink Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Liveness Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
