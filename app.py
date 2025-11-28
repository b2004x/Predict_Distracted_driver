import streamlit as st
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from pathlib import Path
import time
# ==== CONFIG ====
IMG_SIZE = 224
FRAME_INTERVAL = 2
ALERT_THRESHOLD = 5

CLASS_NAMES = ['other_activities', 'safe_driving', 'talking_phone', 'texting_phone', 'turning']
DISTRACTED_CLASSES = {"other_activities", "texting_phone", "talking_phone", "turning"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== LOAD MODEL ====
@st.cache_resource
def load_model(model_path: Path):
    model = models.efficientnet_b0(pretrained= False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, len(CLASS_NAMES))

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def play_alert_sound():
    import base64
    sound = open("alert.mp3", "rb").read()
    b64 = base64.b64encode(sound).decode()

    audio_html = f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# ==== PREDICT IMAGE ====
def predict_image(model, frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    tensor = transform(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        _, preds = torch.max(outputs, 1)

    return CLASS_NAMES[preds.item()]


# ==== STREAMLIT UI ====
st.title("🚗 Driver Behavior Monitoring (Streamlit)")
st.write("AI model detects distracted driving actions frame-by-frame.")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
if uploaded_file:
    st.success("Files uploaded! You can now run the model.")

    model = load_model(Path("EfficientNet_B0.pth"))

    # Save uploaded video temporarily
    temp_video = "temp_input_video.mp4"
    with open(temp_video, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("▶ Start Processing"):
        st_frame = st.empty()
        st_label = st.empty()
        st_alert = st.empty()

        cap = cv2.VideoCapture(temp_video)
        frame_count = 0
        alert_count = 0
        alerted = False
        last_sound_time = 0
        alert_interval = 2.0  # seconds
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display_frame = frame_rgb.copy()

            label = None
            if frame_count % FRAME_INTERVAL == 0:
                label = predict_image(model, frame_rgb)

                # ---- ALERT LOGIC ----
                if label in DISTRACTED_CLASSES:
                    alert_count += 1
                    if alert_count >= ALERT_THRESHOLD and not alerted:
                        alerted = True
                        st_alert.error("🚨 **ALERT: Distracted behavior detected repeatedly!**")

                now = time.time()
                if now - last_sound_time >= alert_interval:
                    play_alert_sound()
                    last_sound_time = now
                else:
                    alert_count = 0
                    alerted = False
                    st_alert.empty()
                    

            # ---- Draw label on frame ----
            if label:
                color = (255, 0, 0) if label in DISTRACTED_CLASSES else (0, 255, 0)
                cv2.putText(display_frame,
                            f"Behavior: {label}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                st_label.write(f"**Prediction:** {label}")

            # ---- DISPLAY FRAME ----
            st_frame.image(display_frame, channels="RGB")

            frame_count += 1

        cap.release()
        st.success("✔ Video processing completed!")
else:
    st.info("Upload a video and model file to start.")
