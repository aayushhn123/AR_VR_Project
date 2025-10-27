import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# Page configuration
st.set_page_config(
    page_title="AR Face Filters",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 20px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .filter-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1 {
        color: white;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .sidebar .sidebar-content {
        background: rgba(255,255,255,0.95);
    }
</style>
""", unsafe_allow_html=True)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.filter_type = "none"
    
    def set_filter(self, filter_type):
        self.filter_type = filter_type
    
    def apply_dog_filter(self, frame, landmarks):
        h, w = frame.shape[:2]
        
        # Get key facial landmarks
        nose_tip = landmarks[4]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        
        # Calculate positions
        nose_x, nose_y = int(nose_tip.x * w), int(nose_tip.y * h)
        left_eye_x, left_eye_y = int(left_eye.x * w), int(left_eye.y * h)
        right_eye_x, right_eye_y = int(right_eye.x * w), int(right_eye.y * h)
        
        # Calculate ear positions
        eye_distance = abs(right_eye_x - left_eye_x)
        ear_size = int(eye_distance * 0.8)
        
        # Draw dog ears (triangles)
        left_ear = np.array([
            [left_eye_x - ear_size//2, left_eye_y - ear_size],
            [left_eye_x - ear_size, left_eye_y + ear_size//2],
            [left_eye_x, left_eye_y]
        ], np.int32)
        
        right_ear = np.array([
            [right_eye_x + ear_size//2, right_eye_y - ear_size],
            [right_eye_x + ear_size, right_eye_y + ear_size//2],
            [right_eye_x, right_eye_y]
        ], np.int32)
        
        cv2.fillPoly(frame, [left_ear], (139, 90, 43))
        cv2.polylines(frame, [left_ear], True, (80, 50, 20), 2)
        cv2.fillPoly(frame, [right_ear], (139, 90, 43))
        cv2.polylines(frame, [right_ear], True, (80, 50, 20), 2)
        
        # Draw dog nose
        nose_size = int(eye_distance * 0.2)
        cv2.ellipse(frame, (nose_x, nose_y), (nose_size, int(nose_size*0.7)), 
                   0, 0, 360, (0, 0, 0), -1)
        
        # Draw tongue
        tongue_y = nose_y + int(eye_distance * 0.3)
        cv2.ellipse(frame, (nose_x, tongue_y), (int(nose_size*0.8), int(nose_size*1.5)), 
                   0, 0, 360, (100, 100, 255), -1)
        
        return frame
    
    def apply_cat_filter(self, frame, landmarks):
        h, w = frame.shape[:2]
        
        nose_tip = landmarks[4]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        
        nose_x, nose_y = int(nose_tip.x * w), int(nose_tip.y * h)
        left_eye_x, left_eye_y = int(left_eye.x * w), int(left_eye.y * h)
        right_eye_x, right_eye_y = int(right_eye.x * w), int(right_eye.y * h)
        
        eye_distance = abs(right_eye_x - left_eye_x)
        ear_size = int(eye_distance * 0.6)
        
        # Draw cat ears (triangles)
        left_ear = np.array([
            [left_eye_x - ear_size//3, left_eye_y - ear_size],
            [left_eye_x - ear_size, left_eye_y],
            [left_eye_x, left_eye_y - ear_size//3]
        ], np.int32)
        
        right_ear = np.array([
            [right_eye_x + ear_size//3, right_eye_y - ear_size],
            [right_eye_x + ear_size, right_eye_y],
            [right_eye_x, right_eye_y - ear_size//3]
        ], np.int32)
        
        cv2.fillPoly(frame, [left_ear], (200, 200, 200))
        cv2.polylines(frame, [left_ear], True, (150, 150, 150), 2)
        # Inner ear
        inner_left = left_ear.copy()
        inner_left