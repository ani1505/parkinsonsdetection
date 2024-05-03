# feature_extraction.py
import cv2
import numpy as np

def extract_optical_flow(video_frames, max_length):
    prev_frame = video_frames[0]
    flow_vectors = []

    for frame in video_frames[1:]:
        # Calculate optical flow between consecutive frames
        flow = cv2.calcOpticalFlowFarneback(prev_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Compute magnitude and angle of the optical flow vectors
        magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1])
        
        # Use only magnitude as a feature
        flow_vectors.append(magnitude)
        
        prev_frame = frame

    # Concatenate flow vectors into a single feature vector
    features = np.concatenate(flow_vectors)

    # Pad or truncate features to ensure consistent length
    if len(features) < max_length:
        features = np.pad(features, (0, max_length - len(features)), mode='constant')
    elif len(features) > max_length:
        features = features[:max_length]

    return features
