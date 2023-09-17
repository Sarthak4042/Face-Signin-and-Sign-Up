import face_recognition
import cv2

# Load an image of the user's face for sign-up
user_image = face_recognition.load_image_file("/Users/sarthakmore/Downloads/My_photo.png")
user_face_encoding = face_recognition.face_encodings(user_image)[0]

# Create an empty list for storing user face encodings
known_face_encodings = []
known_face_names = []

# Add the user's face encoding and name to the lists
known_face_encodings.append(user_face_encoding)
known_face_names.append("User")

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a single frame from the webcam
    ret, frame = video_capture.read()

    # Find face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Initialize matches as False
    matches = []

    # Loop through each face found in the frame
    for face_encoding in face_encodings:
        # Check if the face matches the user's face
        match = face_recognition.compare_faces(known_face_encodings, face_encoding)
        matches.extend(match)

    if True in matches:
        # If a match is found, the user is signed up
        print("Face Sign-Up Successful!")
        break

# Release the webcam and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
