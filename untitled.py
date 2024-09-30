import cv2
import numpy as np
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
from fer import FER
import random

def initialize_detectors():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    fer_detector = FER()
    return detector, predictor, fer_detector

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_eyeball_movement(landmarks):
    left_eye = landmarks[42:48]
    right_eye = landmarks[36:42]
    left_eye_center = left_eye.mean(axis=0).astype("int")
    right_eye_center = right_eye.mean(axis=0).astype("int")
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    ear = (left_ear + right_ear) / 2.0
    return left_eye_center, right_eye_center, ear

def analyze_clip(cap, start_frame, end_frame, detector, predictor, fer_detector):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    movement_counts = []
    blink_counts = 0
    emotion_counts = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
    prev_left_eye_center = None
    prev_right_eye_center = None

    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            left_eye_center, right_eye_center, ear = calculate_eyeball_movement(shape)

            if prev_left_eye_center is not None and prev_right_eye_center is not None:
                left_movement = dist.euclidean(prev_left_eye_center, left_eye_center)
                right_movement = dist.euclidean(prev_right_eye_center, right_eye_center)
                movement_counts.append(left_movement + right_movement)

                if ear < 0.21:  # Threshold for blinking
                    blink_counts += 1

            prev_left_eye_center = left_eye_center
            prev_right_eye_center = right_eye_center

        # Emotion detection
        result = fer_detector.detect_emotions(frame)
        for face in result:
            emotions = face['emotions']
            top_emotion = max(emotions, key=emotions.get)
            emotion_counts[top_emotion] += 1

    return movement_counts, blink_counts, emotion_counts

def main(video_path):
    # Initialize detectors
    detector, predictor, fer_detector = initialize_detectors()

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video details
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    clip_duration = random.randint(5, 7)  # Duration in seconds
    clip_length = int(clip_duration * fps)

    # Initialize cumulative variables
    total_movement_counts = []
    total_blink_counts = 0
    total_emotion_counts = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}

    # Process random clips
    num_clips = 12  # Number of random clips to analyze
    for _ in range(num_clips):
        start_frame = random.randint(0, total_frames - clip_length)
        end_frame = min(start_frame + clip_length, total_frames)

        movement_counts, blink_counts, emotion_counts = analyze_clip(cap, start_frame, end_frame, detector, predictor, fer_detector)
        
        total_movement_counts.extend(movement_counts)
        total_blink_counts += blink_counts
        for emotion, count in emotion_counts.items():
            total_emotion_counts[emotion] += count

    # Calculate cumulative statistics
    avg_movement_rate = np.mean(total_movement_counts) if total_movement_counts else 0
    total_frames_analyzed = sum(total_emotion_counts.values())
    emotion_percentages = {emotion: (count / total_frames_analyzed) * 100 for emotion, count in total_emotion_counts.items()} if total_frames_analyzed > 0 else {emotion: 0 for emotion in total_emotion_counts}
    
    sorted_emotions = sorted(emotion_percentages.items(), key=lambda x: x[1], reverse=True)
    dominant_emotion, dominant_emotion_percentage = sorted_emotions[0] if sorted_emotions else ('neutral', 0)
    second_dominant_emotion, second_dominant_emotion_percentage = sorted_emotions[1] if len(sorted_emotions) > 1 else ('neutral', 0)

    # Generate detailed description based on dominant emotion
    description = ""
    confidence_score = 8  # Default confidence score

    if dominant_emotion == 'neutral':
        description = (
            f"The candidate's predominant emotional state is neutral, accounting for {dominant_emotion_percentage:.2f}% of the video. "
            f"However, there were also noticeable frames where the second most dominant emotion, {second_dominant_emotion}, "
            f"was observed with {second_dominant_emotion_percentage:.2f}% presence. This indicates that while the overall demeanor "
            f"appears neutral, there are instances where other emotions were also present."
        )
    elif dominant_emotion in ['fear', 'disgust']:
        description = (
            f"The candidate appears quite uneasy with a dominant emotional state of {dominant_emotion} "
            f"at {dominant_emotion_percentage:.2f}%. There is some evidence of nervousness "
            f"indicated by eye movements and blinking. Overall, the candidate's demeanor suggests a higher "
            f"level of anxiety."
        )
        confidence_score -= 2
    elif dominant_emotion == 'happy':
        description = (
            f"The candidate seems positive with a dominant emotional state of {dominant_emotion} "
            f"at {dominant_emotion_percentage:.2f}%. The eye movement and blinking are within normal ranges, "
            f"indicating that the candidate is calm and composed throughout the video."
        )
        confidence_score += 2
    elif dominant_emotion == 'sad':
        description = (
            f"The candidate appears to be experiencing a dominant emotional state of sadness at {dominant_emotion_percentage:.2f}%. "
            f"Eye movement and blink rate are moderate, which might indicate that the candidate is feeling down or reflective."
        )
        confidence_score -= 1
    elif dominant_emotion == 'surprise':
        description = (
            f"The candidate shows a dominant emotional state of surprise at {dominant_emotion_percentage:.2f}%. "
            f"Eye movements and blink rate might be heightened, reflecting a state of astonishment or unexpectedness."
        )
    else:
        description = (
            f"The candidate shows a mixed emotional state with no overwhelmingly dominant emotion. Eye movement and blink rate "
            f"are moderate, suggesting that while the candidate might not be extremely anxious, there are "
            f"slight signs of nervousness. The overall demeanor is neutral."
        )

    if avg_movement_rate > 2 or total_blink_counts > (20 * (total_frames_analyzed / fps / 60)):
        description += (
            " The eye movement and blink count suggest some level of nervousness, adding to the overall "
            "impression of anxiety or unease."
        )
        confidence_score -= 1

    # Ensure confidence score is within 0-10 range
    confidence_score = max(0, min(10, confidence_score))

    # Output results
    print(f"Average Eye Movement Rate: {avg_movement_rate:.2f}")
    print(f"Total Blink Count: {total_blink_counts}")
    print(f"Emotion Counts: {total_emotion_counts}")
    print(f"Overall Description: {description}")
    print(f"Confidence Score: {confidence_score}/10")

if __name__ == "__main__":
    # Example usage with a sample video path
    video_path = "/Users/dhyan/Desktop/folder2/Update _ Interview Confirmation _ Technical Round 2_ Power BI Developer _ Gemini Solutions!-20240726_171447-Meeting Recording.mp4"
    main(video_path)
