import cv2
import dlib
import numpy as np
from fer import FER  
from imutils import face_utils
from scipy.spatial import distance as dist

def initialize_detectors():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    fer_detector = FER(mtcnn=True)
    return detector, predictor, fer_detector

def main(video_path):
    # Initialize detectors
    detector, predictor, fer_detector = initialize_detectors()

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video details
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create a VideoWriter object to save the output video
    output_path = video_path.split('.')[0] + '_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Emotion detection
        result = fer_detector.detect_emotions(frame)
        if result:
            # Display the detected emotions and draw a rectangle around detected faces
            for face in result:
                emotions = face['emotions']
                top_emotion = max(emotions, key=emotions.get)
                bounding_box = face['box']
                
                # Draw a rectangle around the face
                (x, y, w, h) = bounding_box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Annotate the frame with the detected emotion
                cv2.putText(frame, f"Emotion: {top_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Write the annotated frame to the output video
            out.write(frame)

    # Release the video capture and writer objects
    cap.release()
    out.release()

    print(f"Output video saved to {output_path}")

# Example usage
main('/Users/dhyan/Desktop/folder2/happy.mp4')
