import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
from time import time
from PIL import Image
import gdown
from tqdm import tqdm
import io

MODEL_ID = '1FMXOk9ifEoZDl4c7NzpANiP2o_Ednt7P'
MODEL_PATH = 'FINAL_MODEL.h5'

# Load model
@st.cache_resource
def load_detection_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.info("Downloading model from Google Drive...")
            url = f'https://drive.google.com/uc?id={MODEL_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
        
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_detection_model()

# Define class names
class_names = ['buah', 'karbohidrat', 'minuman', 'protein', 'sayur']

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return np.expand_dims(image, axis=0)

def get_region_proposals(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > 1000:
            pad_w = int(w * 0.2)
            pad_h = int(h * 0.2)
            x = max(0, x - pad_w)
            y = max(0, y - pad_h)
            w = min(image.shape[1] - x, w + 2 * pad_w)
            h = min(image.shape[0] - y, h + 2 * pad_h)
            regions.append((x, y, w, h))
    
    if not regions:
        regions.append((0, 0, image.shape[1], image.shape[0]))
    
    return regions

def detect_objects_in_region(image, box, model, min_confidence=0.5):
    x, y, w, h = box
    region = image[y:y+h, x:x+w]
    
    if region.shape[0] < 32 or region.shape[1] < 32:
        return None
        
    processed_region = preprocess_image(region)
    predictions = model.predict(processed_region, verbose=0)
    confidence = float(np.max(predictions[0]))
        
    if confidence >= min_confidence:
        class_idx = np.argmax(predictions[0])
        return {
            'box': box,
            'confidence': confidence,
            'class': class_names[class_idx],
        }
    return None

def apply_nms(detections, nms_threshold=0.5):
    if not detections:
        return []
        
    boxes = [d['box'] for d in detections]
    scores = [d['confidence'] for d in detections]
    boxes_nms = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in boxes]
    indices = cv2.dnn.NMSBoxes(boxes_nms, scores, 0.5, nms_threshold)
    
    if len(indices) > 0:
        return [detections[i] for i in indices.flatten()]
    return []

def draw_detection(image, detection):
    colors = {
        'buah': (255, 0, 0),
        'karbohidrat': (0, 255, 0),
        'minuman': (0, 0, 255),
        'protein': (255, 255, 0),
        'sayur': (255, 0, 255)
    }
    
    x, y, w, h = detection['box']
    class_name = detection['class']
    confidence = detection['confidence']
    color = colors.get(class_name, (0, 255, 0))
    
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    label = f'{class_name}: {confidence:.2f}'
    (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(image, (x, y - label_h - 10), (x + label_w + 10, y), color, -1)
    cv2.putText(image, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def process_frame(frame, model, max_regions=10):
    regions = get_region_proposals(frame)
    regions = regions[:max_regions]
    
    detections = []
    for box in regions:
        result = detect_objects_in_region(frame, box, model)
        if result:
            detections.append(result)
    
    detections = apply_nms(detections)
    
    for det in detections:
        draw_detection(frame, det)
    
    return frame

def main():
    st.title("Food Detection App")
    
    model = load_detection_model()
    
    st.write("Choose detection mode:")
    detection_mode = st.radio("Select Detection Mode:", ["Image Upload", "Real-time Camera"])
    
    if detection_mode == "Image Upload":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            image_bytes = uploaded_file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            start_time = time()
            processed_image = process_frame(image.copy(), model)
            processing_time = time() - start_time
            
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="Processed Image")
            st.write(f"Processing time: {processing_time:.2f} seconds")
    
    else:
        st.write("Real-time detection mode")
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            st.error("Camera not found or not accessible. Please check your camera setup.")
        else:
            run = st.checkbox('Start/Stop')
            FRAME_WINDOW = st.image([])
            
            while run:
                ret, frame = camera.read()
                if not ret:
                    st.error("Unable to read frame from camera.")
                    break

                processed_frame = process_frame(frame, model)
                FRAME_WINDOW.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
            
            camera.release()

if __name__ == "__main__":
    main()
