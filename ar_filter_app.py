import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import tempfile
import time

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
    .stCamera > div {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize MediaPipe Face Mesh
@st.cache_resource
def load_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

face_mesh = load_face_mesh()

def apply_dog_filter(frame, landmarks):
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

def apply_cat_filter(frame, landmarks):
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
    inner_left = (left_ear * 0.7 + np.array([left_eye_x, left_eye_y]) * 0.3).astype(np.int32)
    cv2.fillPoly(frame, [inner_left], (255, 192, 203))
    
    inner_right = (right_ear * 0.7 + np.array([right_eye_x, right_eye_y]) * 0.3).astype(np.int32)
    cv2.fillPoly(frame, [inner_right], (255, 192, 203))
    
    cv2.fillPoly(frame, [right_ear], (200, 200, 200))
    cv2.polylines(frame, [right_ear], True, (150, 150, 150), 2)
    
    # Draw cat nose
    nose_size = int(eye_distance * 0.15)
    pts = np.array([
        [nose_x, nose_y - nose_size//2],
        [nose_x - nose_size//2, nose_y + nose_size//2],
        [nose_x + nose_size//2, nose_y + nose_size//2]
    ], np.int32)
    cv2.fillPoly(frame, [pts], (255, 105, 180))
    
    # Draw whiskers
    whisker_length = int(eye_distance * 0.8)
    whisker_positions = [
        (nose_x, nose_y - nose_size//4),
        (nose_x, nose_y),
        (nose_x, nose_y + nose_size//4)
    ]
    
    for wx, wy in whisker_positions:
        cv2.line(frame, (wx, wy), (wx - whisker_length, wy), (0, 0, 0), 2)
        cv2.line(frame, (wx, wy), (wx + whisker_length, wy), (0, 0, 0), 2)
    
    return frame

def apply_glasses_filter(frame, landmarks):
    h, w = frame.shape[:2]
    
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    
    left_eye_x, left_eye_y = int(left_eye.x * w), int(left_eye.y * h)
    right_eye_x, right_eye_y = int(right_eye.x * w), int(right_eye.y * h)
    
    eye_distance = abs(right_eye_x - left_eye_x)
    glass_width = int(eye_distance * 0.4)
    glass_height = int(glass_width * 0.8)
    
    # Draw left lens
    cv2.ellipse(frame, (left_eye_x, left_eye_y), (glass_width, glass_height), 
               0, 0, 360, (0, 0, 0), 3)
    cv2.ellipse(frame, (left_eye_x, left_eye_y), (glass_width-3, glass_height-3), 
               0, 0, 360, (200, 200, 255), 2)
    
    # Draw right lens
    cv2.ellipse(frame, (right_eye_x, right_eye_y), (glass_width, glass_height), 
               0, 0, 360, (0, 0, 0), 3)
    cv2.ellipse(frame, (right_eye_x, right_eye_y), (glass_width-3, glass_height-3), 
               0, 0, 360, (200, 200, 255), 2)
    
    # Draw bridge
    cv2.line(frame, 
            (left_eye_x + glass_width, left_eye_y),
            (right_eye_x - glass_width, right_eye_y),
            (0, 0, 0), 3)
    
    # Draw arms
    cv2.line(frame, (left_eye_x - glass_width, left_eye_y),
            (left_eye_x - glass_width - 50, left_eye_y), (0, 0, 0), 3)
    cv2.line(frame, (right_eye_x + glass_width, right_eye_y),
            (right_eye_x + glass_width + 50, right_eye_y), (0, 0, 0), 3)
    
    return frame

def apply_crown_filter(frame, landmarks):
    h, w = frame.shape[:2]
    
    forehead = landmarks[10]
    left_temple = landmarks[234]
    right_temple = landmarks[454]
    
    forehead_x, forehead_y = int(forehead.x * w), int(forehead.y * h)
    left_x = int(left_temple.x * w)
    right_x = int(right_temple.x * w)
    
    crown_width = abs(right_x - left_x)
    crown_height = int(crown_width * 0.4)
    crown_y = forehead_y - crown_height - 20
    
    # Draw crown base
    base_pts = np.array([
        [left_x, crown_y + crown_height],
        [left_x, crown_y + crown_height - 20],
        [right_x, crown_y + crown_height - 20],
        [right_x, crown_y + crown_height]
    ], np.int32)
    cv2.fillPoly(frame, [base_pts], (255, 215, 0))
    cv2.polylines(frame, [base_pts], True, (218, 165, 32), 2)
    
    # Draw crown points
    num_points = 5
    for i in range(num_points):
        point_x = left_x + (crown_width * i) // (num_points - 1)
        point_y = crown_y if i % 2 == 0 else crown_y + 20
        
        triangle = np.array([
            [point_x, point_y],
            [point_x - 15, crown_y + crown_height - 20],
            [point_x + 15, crown_y + crown_height - 20]
        ], np.int32)
        cv2.fillPoly(frame, [triangle], (255, 215, 0))
        cv2.polylines(frame, [triangle], True, (218, 165, 32), 2)
        
        # Add jewels
        cv2.circle(frame, (point_x, point_y + 10), 5, (255, 0, 0), -1)
    
    return frame

def process_image(image, filter_type):
    """Process image with selected filter"""
    # Convert PIL to OpenCV
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    if filter_type == "none":
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        if filter_type == "dog":
            img = apply_dog_filter(img, landmarks)
        elif filter_type == "cat":
            img = apply_cat_filter(img, landmarks)
        elif filter_type == "glasses":
            img = apply_glasses_filter(img, landmarks)
        elif filter_type == "crown":
            img = apply_crown_filter(img, landmarks)
    
    # Convert back to PIL
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Main app
st.title("🎭 AR Face Filters")
st.markdown("### Snapchat-style filters powered by AI")

# Sidebar for filter selection
with st.sidebar:
    st.markdown("## 🎨 Choose Your Filter")
    
    filter_options = {
        "None": "none",
        "🐶 Dog": "dog",
        "🐱 Cat": "cat",
        "👓 Glasses": "glasses",
        "👑 Crown": "crown"
    }
    
    selected_filter = st.radio(
        "Select a filter:",
        list(filter_options.keys()),
        index=0
    )
    
    st.markdown("---")
    st.markdown("""
    ### 📝 Instructions
    1. Click "Take Photo" below
    2. Allow camera access
    3. Select a filter from the list
    4. Take a photo to see the filter!
    
    ### 💡 Tips
    - Ensure good lighting
    - Keep face centered
    - Try different expressions!
    """)

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<div class="filter-card">', unsafe_allow_html=True)
    
    # Camera input
    camera_photo = st.camera_input("📸 Take a photo", key="camera")
    
    if camera_photo is not None:
        # Read image
        image = Image.open(camera_photo)
        
        # Process with filter
        filter_type = filter_options[selected_filter]
        
        with st.spinner(f"Applying {selected_filter} filter..."):
            processed_image = process_image(image, filter_type)
        
        # Display result
        st.image(processed_image, caption=f"With {selected_filter} Filter", use_container_width=True)
        
        # Download button
        buf = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        processed_image.save(buf.name, format='PNG')
        
        with open(buf.name, 'rb') as file:
            st.download_button(
                label="📥 Download Photo",
                data=file,
                file_name=f"filtered_photo_{int(time.time())}.png",
                mime="image/png"
            )
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### 🎯 Active Filter")
    st.info(f"**{selected_filter}**")
    
    if selected_filter == "🐶 Dog":
        st.markdown("Woof! You're a good boy! 🦴")
    elif selected_filter == "🐱 Cat":
        st.markdown("Meow! So purrfect! 🐾")
    elif selected_filter == "👓 Glasses":
        st.markdown("Looking smart! 🤓")
    elif selected_filter == "👑 Crown":
        st.markdown("Royalty detected! 👸")
    else:
        st.markdown("No filter applied")
    
    st.markdown("---")
    st.markdown("### 📊 Features")
    st.success("✅ Real-time face detection")
    st.success("✅ Multiple filter options")
    st.success("✅ Download your photos")
    st.success("✅ Easy to use")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white;'>
    <p>Built with ❤️ using Streamlit, OpenCV & MediaPipe</p>
    <p style='font-size: 12px;'>📱 Works on mobile and desktop!</p>
</div>
""", unsafe_allow_html=True)
