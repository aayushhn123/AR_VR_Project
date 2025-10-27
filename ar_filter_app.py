import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import tempfile
import time
from math import atan2, degrees, radians, sin, cos

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

def rotate_point(px, py, cx, cy, angle):
    rad = radians(angle)
    dx = px - cx
    dy = py - cy
    return (
        cx + dx * cos(rad) - dy * sin(rad),
        cy + dx * sin(rad) + dy * cos(rad)
    )

def apply_dog_filter(frame, landmarks, face_angle):
    h, w = frame.shape[:2]
    
    nose_tip = landmarks[4]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    
    nose_x, nose_y = int(nose_tip.x * w), int(nose_tip.y * h)
    left_eye_x, left_eye_y = int(left_eye.x * w), int(left_eye.y * h)
    right_eye_x, right_eye_y = int(right_eye.x * w), int(right_eye.y * h)
    
    eye_distance = int(np.sqrt((right_eye_x - left_eye_x)**2 + (right_eye_y - left_eye_y)**2))
    ear_size = int(eye_distance * 1.0)  # Increased size for better proportion
    
    # Improved dog ears with more points for rounded shape
    left_ear_base = (left_eye_x - ear_size // 4, left_eye_y - ear_size // 2)
    left_ear_pts = [
        [left_ear_base[0], left_ear_base[1] - ear_size],
        [left_ear_base[0] - ear_size // 2, left_ear_base[1] - ear_size // 2],
        [left_ear_base[0] - ear_size // 4, left_ear_base[1]],
        [left_ear_base[0] + ear_size // 4, left_ear_base[1]],
        [left_ear_base[0] + ear_size // 2, left_ear_base[1] - ear_size // 2]
    ]
    
    # Rotate left ear around base
    rotated_left_ear = [rotate_point(p[0], p[1], *left_ear_base, face_angle) for p in left_ear_pts]
    left_ear_poly = np.array(rotated_left_ear, np.int32)
    
    right_ear_base = (right_eye_x + ear_size // 4, right_eye_y - ear_size // 2)
    right_ear_pts = [
        [right_ear_base[0], right_ear_base[1] - ear_size],
        [right_ear_base[0] + ear_size // 2, right_ear_base[1] - ear_size // 2],
        [right_ear_base[0] + ear_size // 4, right_ear_base[1]],
        [right_ear_base[0] - ear_size // 4, right_ear_base[1]],
        [right_ear_base[0] - ear_size // 2, right_ear_base[1] - ear_size // 2]
    ]
    
    # Rotate right ear around base
    rotated_right_ear = [rotate_point(p[0], p[1], *right_ear_base, face_angle) for p in right_ear_pts]
    right_ear_poly = np.array(rotated_right_ear, np.int32)
    
    # Draw ears with fur-like border
    cv2.fillPoly(frame, [left_ear_poly], (139, 90, 43))
    cv2.polylines(frame, [left_ear_poly], True, (80, 50, 20), 3)
    cv2.fillPoly(frame, [right_ear_poly], (139, 90, 43))
    cv2.polylines(frame, [right_ear_poly], True, (80, 50, 20), 3)
    
    # Dog nose with shine
    nose_size = int(eye_distance * 0.25)
    cv2.ellipse(frame, (nose_x, nose_y), (nose_size, int(nose_size*0.8)), face_angle, 0, 360, (0, 0, 0), -1)
    # Add shine
    cv2.circle(frame, (nose_x - nose_size // 4, nose_y - nose_size // 4), nose_size // 6, (255, 255, 255), -1)
    
    # Improved tongue with curve
    tongue_y = nose_y + int(eye_distance * 0.4)
    tongue_width = int(nose_size * 1.2)
    tongue_height = int(nose_size * 2.0)
    cv2.ellipse(frame, (nose_x, tongue_y), (tongue_width // 2, tongue_height // 2), face_angle, 0, 360, (100, 100, 255), -1)
    # Add tongue line
    tongue_line_start = (nose_x, tongue_y - tongue_height // 4)
    tongue_line_end = (nose_x, tongue_y + tongue_height // 4)
    rotated_start = rotate_point(*tongue_line_start, nose_x, tongue_y, face_angle)
    rotated_end = rotate_point(*tongue_line_end, nose_x, tongue_y, face_angle)
    cv2.line(frame, (int(rotated_start[0]), int(rotated_start[1])), (int(rotated_end[0]), int(rotated_end[1])), (50, 50, 200), 2)
    
    return frame

def apply_cat_filter(frame, landmarks, face_angle):
    h, w = frame.shape[:2]
    
    nose_tip = landmarks[4]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    
    nose_x, nose_y = int(nose_tip.x * w), int(nose_tip.y * h)
    left_eye_x, left_eye_y = int(left_eye.x * w), int(left_eye.y * h)
    right_eye_x, right_eye_y = int(right_eye.x * w), int(right_eye.y * h)
    
    eye_distance = int(np.sqrt((right_eye_x - left_eye_x)**2 + (right_eye_y - left_eye_y)**2))
    ear_size = int(eye_distance * 0.8)
    
    # Improved cat ears with pointed shape
    left_ear_base = (left_eye_x, left_eye_y - ear_size // 3)
    left_ear_pts = [
        [left_ear_base[0], left_ear_base[1] - ear_size],
        [left_ear_base[0] - ear_size, left_ear_base[1]],
        [left_ear_base[0] + ear_size // 3, left_ear_base[1]]
    ]
    
    rotated_left_ear = [rotate_point(p[0], p[1], *left_ear_base, face_angle) for p in left_ear_pts]
    left_ear_poly = np.array(rotated_left_ear, np.int32)
    
    right_ear_base = (right_eye_x, right_eye_y - ear_size // 3)
    right_ear_pts = [
        [right_ear_base[0], right_ear_base[1] - ear_size],
        [right_ear_base[0] + ear_size, right_ear_base[1]],
        [right_ear_base[0] - ear_size // 3, right_ear_base[1]]
    ]
    
    rotated_right_ear = [rotate_point(p[0], p[1], *right_ear_base, face_angle) for p in right_ear_pts]
    right_ear_poly = np.array(rotated_right_ear, np.int32)
    
    cv2.fillPoly(frame, [left_ear_poly], (200, 200, 200))
    cv2.polylines(frame, [left_ear_poly], True, (150, 150, 150), 2)
    
    cv2.fillPoly(frame, [right_ear_poly], (200, 200, 200))
    cv2.polylines(frame, [right_ear_poly], True, (150, 150, 150), 2)
    
    # Inner ears
    inner_scale = 0.6
    left_inner_pts = [ (left_ear_base[0] + inner_scale * (p[0] - left_ear_base[0]), left_ear_base[1] + inner_scale * (p[1] - left_ear_base[1])) for p in left_ear_pts ]
    rotated_left_inner = [rotate_point(p[0], p[1], *left_ear_base, face_angle) for p in left_inner_pts]
    left_inner_poly = np.array(rotated_left_inner, np.int32)
    cv2.fillPoly(frame, [left_inner_poly], (255, 192, 203))
    
    right_inner_pts = [ (right_ear_base[0] + inner_scale * (p[0] - right_ear_base[0]), right_ear_base[1] + inner_scale * (p[1] - right_ear_base[1])) for p in right_ear_pts ]
    rotated_right_inner = [rotate_point(p[0], p[1], *right_ear_base, face_angle) for p in right_inner_pts]
    right_inner_poly = np.array(rotated_right_inner, np.int32)
    cv2.fillPoly(frame, [right_inner_poly], (255, 192, 203))
    
    # Cat nose
    nose_size = int(eye_distance * 0.15)
    nose_pts = [
        [nose_x, nose_y - nose_size // 2],
        [nose_x - nose_size // 2, nose_y + nose_size // 2],
        [nose_x + nose_size // 2, nose_y + nose_size // 2]
    ]
    rotated_nose = [rotate_point(p[0], p[1], nose_x, nose_y, face_angle) for p in nose_pts]
    nose_poly = np.array(rotated_nose, np.int32)
    cv2.fillPoly(frame, [nose_poly], (255, 105, 180))
    
    # Improved whiskers with multiple thin lines
    whisker_length = int(eye_distance * 1.0)
    whisker_positions = [
        (nose_x - nose_size // 2, nose_y - nose_size // 3),
        (nose_x - nose_size // 2, nose_y + nose_size // 3),
        (nose_x + nose_size // 2, nose_y - nose_size // 3),
        (nose_x + nose_size // 2, nose_y + nose_size // 3)
    ]
    rad = radians(face_angle)
    for wx, wy in whisker_positions:
        for offset in [-5, 0, 5]:  # Multiple strands
            wy_off = wy + offset
            for direction in [-1, 1]:
                vx = direction * whisker_length
                vy = 0
                rvx = vx * cos(rad) - vy * sin(rad)
                rvy = vx * sin(rad) + vy * cos(rad)
                end_x = wx + rvx
                end_y = wy_off + rvy
                cv2.line(frame, (wx, wy_off), (int(end_x), int(end_y)), (0, 0, 0), 1)
    
    return frame

def apply_glasses_filter(frame, landmarks, face_angle):
    h, w = frame.shape[:2]
    
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    
    left_eye_x, left_eye_y = int(left_eye.x * w), int(left_eye.y * h)
    right_eye_x, right_eye_y = int(right_eye.x * w), int(right_eye.y * h)
    
    eye_distance = int(np.sqrt((right_eye_x - left_eye_x)**2 + (right_eye_y - left_eye_y)**2))
    glass_width = int(eye_distance * 0.55)  # Adjusted for better fit
    glass_height = int(glass_width * 0.6)
    
    # Draw left lens with gradient fill simulation
    cv2.ellipse(frame, (left_eye_x, left_eye_y), (glass_width, glass_height), 
                face_angle, 0, 360, (0, 0, 0), 3)
    cv2.ellipse(frame, (left_eye_x, left_eye_y), (glass_width-3, glass_height-3), 
                face_angle, 0, 360, (150, 150, 255), -1)  # Solid fill for lens
    
    # Draw right lens
    cv2.ellipse(frame, (right_eye_x, right_eye_y), (glass_width, glass_height), 
                face_angle, 0, 360, (0, 0, 0), 3)
    cv2.ellipse(frame, (right_eye_x, right_eye_y), (glass_width-3, glass_height-3), 
                face_angle, 0, 360, (150, 150, 255), -1)
    
    # Bridge already tilted due to positions
    cv2.line(frame, 
             (left_eye_x + int(glass_width * cos(radians(face_angle))), left_eye_y + int(glass_width * sin(radians(face_angle)))),
             (right_eye_x - int(glass_width * cos(radians(face_angle))), right_eye_y - int(glass_width * sin(radians(face_angle)))),
             (0, 0, 0), 3)
    
    # Arms with rotation
    arm_length = int(eye_distance * 0.6)
    rad = radians(face_angle)
    
    # Left arm
    left_arm_start_x = left_eye_x - glass_width * cos(rad) + glass_height * sin(rad)  # Approximate side point
    left_arm_start_y = left_eye_y - glass_width * sin(rad) - glass_height * cos(rad)
    vx = -arm_length
    vy = 0
    rvx = vx * cos(rad) - vy * sin(rad)
    rvy = vx * sin(rad) + vy * cos(rad)
    left_arm_end_x = left_arm_start_x + rvx
    left_arm_end_y = left_arm_start_y + rvy
    cv2.line(frame, (int(left_arm_start_x), int(left_arm_start_y)),
             (int(left_arm_end_x), int(left_arm_end_y)), (0, 0, 0), 3)
    
    # Right arm
    right_arm_start_x = right_eye_x + glass_width * cos(rad) + glass_height * sin(rad)
    right_arm_start_y = right_eye_y + glass_width * sin(rad) - glass_height * cos(rad)
    vx = arm_length
    vy = 0
    rvx = vx * cos(rad) - vy * sin(rad)
    rvy = vx * sin(rad) + vy * cos(rad)
    right_arm_end_x = right_arm_start_x + rvx
    right_arm_end_y = right_arm_start_y + rvy
    cv2.line(frame, (int(right_arm_start_x), int(right_arm_start_y)),
             (int(right_arm_end_x), int(right_arm_end_y)), (0, 0, 0), 3)
    
    return frame

def apply_crown_filter(frame, landmarks, face_angle):
    h, w = frame.shape[:2]
    
    # Better landmarks for head top
    forehead = landmarks[9]  # Better center
    left_temple = landmarks[162]
    right_temple = landmarks[389]
    
    forehead_x, forehead_y = int(forehead.x * w), int(forehead.y * h)
    left_x, left_y = int(left_temple.x * w), int(left_temple.y * h)
    right_x, right_y = int(right_temple.x * w), int(right_temple.y * h)
    
    crown_width = int(np.sqrt((right_x - left_x)**2 + (right_y - left_y)**2) * 1.2)
    crown_height = int(crown_width * 0.35)
    crown_y = forehead_y - crown_height - 30  # Adjusted position
    
    crown_center_x = (left_x + right_x) // 2
    crown_center_y = crown_y + crown_height // 2
    
    # Crown base as rectangle, rotated
    base_half_width = crown_width // 2
    base_half_height = 20
    base_pts = [
        [crown_center_x - base_half_width, crown_center_y - base_half_height],
        [crown_center_x - base_half_width, crown_center_y + base_half_height],
        [crown_center_x + base_half_width, crown_center_y + base_half_height],
        [crown_center_x + base_half_width, crown_center_y - base_half_height]
    ]
    rotated_base = [rotate_point(p[0], p[1], crown_center_x, crown_center_y, face_angle) for p in base_pts]
    base_poly = np.array(rotated_base, np.int32)
    cv2.fillPoly(frame, [base_poly], (255, 215, 0))
    cv2.polylines(frame, [base_poly], True, (218, 165, 32), 2)
    
    # Crown points with more detail
    num_points = 7  # More points for finer look
    point_height = crown_height * 0.8
    for i in range(num_points):
        base_x = crown_center_x - base_half_width + (crown_width * i) // (num_points - 1)
        base_y = crown_center_y - base_half_height
        peak_height = point_height if i % 2 == 0 else point_height * 0.7
        peak_x = base_x
        peak_y = base_y - peak_height
        
        triangle_pts = [
            [peak_x, peak_y],
            [base_x - 20, base_y],
            [base_x + 20, base_y]
        ]
        rotated_triangle = [rotate_point(p[0], p[1], crown_center_x, crown_center_y, face_angle) for p in triangle_pts]
        triangle_poly = np.array(rotated_triangle, np.int32)
        cv2.fillPoly(frame, [triangle_poly], (255, 215, 0))
        cv2.polylines(frame, [triangle_poly], True, (218, 165, 32), 2)
        
        # Jewel
        jewel_x, jewel_y = peak_x, peak_y + 15
        rotated_jewel = rotate_point(jewel_x, jewel_y, crown_center_x, crown_center_y, face_angle)
        cv2.circle(frame, (int(rotated_jewel[0]), int(rotated_jewel[1])), 6, (255, 0, 0), -1)
    
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
        
        # Calculate face angle and eye positions for distance
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        left_eye_x, left_eye_y = int(left_eye.x * img.shape[1]), int(left_eye.y * img.shape[0])
        right_eye_x, right_eye_y = int(right_eye.x * img.shape[1]), int(right_eye.y * img.shape[0])
        face_angle = degrees(atan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x))
        
        if filter_type == "dog":
            img = apply_dog_filter(img, landmarks, face_angle)
        elif filter_type == "cat":
            img = apply_cat_filter(img, landmarks, face_angle)
        elif filter_type == "glasses":
            img = apply_glasses_filter(img, landmarks, face_angle)
        elif filter_type == "crown":
            img = apply_crown_filter(img, landmarks, face_angle)
    
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
