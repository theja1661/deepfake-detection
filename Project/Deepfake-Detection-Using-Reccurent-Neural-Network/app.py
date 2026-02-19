import os
import pickle
import torch
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
from flask_ngrok import run_with_ngrok
from torchvision import transforms
from model import Model
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Flask app
app = Flask(__name__)
run_with_ngrok(app)

# Create static/uploads folder if it doesn't exist
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Paths for text-based model
TEXT_MODEL_PATH = "/home/Project/Deepfake-Detection-Using-Reccurent-Neural-Network/models/"
VECTORIZER_FILE = os.path.join(TEXT_MODEL_PATH, "lstm.pkl")
MODEL_FILE = os.path.join(TEXT_MODEL_PATH, "debert.pkl")

# Load text classification models
try:
    with open(VECTORIZER_FILE, "rb") as vf:
        vectorizer = pickle.load(vf)

    with open(MODEL_FILE, "rb") as mf:
        text_model = pickle.load(mf)

    print("Text models loaded successfully!")
except Exception as e:
    print("Error loading text models:", str(e))
    exit(1)

# Define transformation for video frames
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load the deepfake detection model
video_model = Model(2).cuda()
video_model.load_state_dict(torch.load(
    '/home/Project/Deepfake-Detection-Using-Reccurent-Neural-Network/Models/model_97_acc_100_frames_FF_data.pt'))
video_model.eval()

def frame_extract(path):
    """Extract frames from video"""
    vidObj = cv2.VideoCapture(path)
    success = 1
    while success:
        success, image = vidObj.read()
        if success:
            yield image

def validate_video(vid_path):
    """Check if the video is corrupted and get frames"""
    count = 20
    frames = []
    a = int(100 / count)
    first_frame = np.random.randint(0, a)
    for i, frame in enumerate(frame_extract(vid_path)):
        frames.append(train_transforms(frame))
        if len(frames) == count:
            break
    frames = torch.stack(frames)
    frames = frames[:count]
    return frames

def predict_video(model, video_path):
    """Predict if the video is FAKE or REAL"""
    frames = validate_video(video_path)
    frames = frames.unsqueeze(0).cuda()
    output = model(frames)
    _, prediction = torch.max(output[1], 1)
    confidence = torch.nn.Softmax(dim=1)(output[1])[0][prediction].item()
    result = "REAL" if prediction.item() == 1 else "FAKE"
    return result, confidence

def predict_text(text):
    """Predict if the news text is FAKE or REAL"""
    text_vectorized = vectorizer.transform([text])
    prediction = text_model.predict(text_vectorized)[0]
    return "FAKE" if prediction == 1 else "REAL"

@app.route("/", methods=["GET"])
def index():
    return render_template('index.html')

@app.route("/predict_text", methods=["POST"])
def predict_text_route():
    if 'text' in request.form and request.form['text'].strip():
        text = request.form['text']
        result = predict_text(text)
        return render_template('result.html', result=result, type="text")
    else:
        return render_template("index.html", error="Please enter news text.")

@app.route("/predict_video", methods=["POST"])
def predict_video_route():
    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template("index.html", error="Please upload a video.")

    file = request.files['file']
    video_filename = file.filename
    saved_path = os.path.join(UPLOAD_FOLDER, video_filename)
    file.save(saved_path)

    try:
        result, confidence = predict_video(video_model, saved_path)

        # Path used in HTML (must begin with /static/)
        video_path = f"/static/uploads/{video_filename}"

        return render_template('result.html',
                               result=result,
                               confidence=round(confidence * 100, 2),
                               type="video",
                               video_path=video_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        pass  # Don't delete the file if you want to show it in result.html

if __name__ == "__main__":
    app.run()
  