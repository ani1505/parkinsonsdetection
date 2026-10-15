import cv2
import numpy as np
import os

def preprocess_video(video_path, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    prev_frame = None
    prev_frame_time = 0

    # Create the frames folder if it doesn't exist
    frames_folder = os.path.join('dataset', 'frames')
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate the current time of the frame
        current_frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Check if the current frame time exceeds the previous frame time by the frame rate (in seconds)
        if current_frame_time - prev_frame_time >= frame_rate:
            # Convert frame to grayscale and resize
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_resized = cv2.resize(frame_gray, (224, 224))
            
            # Check if the frame is significantly different from the previous frame
            if prev_frame is None or cv2.norm(frame_resized, prev_frame, cv2.NORM_L2) > 1000.0:
                # Save frame as an image file in the frames folder
                frame_path = os.path.join(frames_folder, f'{os.path.splitext(os.path.basename(video_path))[0]}_frame_{int(current_frame_time * 1000)}.jpg')
                cv2.imwrite(frame_path, frame_resized)
                
                # Add frame to the list
                frames.append(frame_resized)
                
                # Update the previous frame and frame time
                prev_frame = frame_resized
                prev_frame_time = current_frame_time

    cap.release()
    return frames

def load_frames(frames_folder):
    frames = []
    for filename in os.listdir(frames_folder):
        frame_path = os.path.join(frames_folder, filename)
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        frames.append(frame)
    return frames
