from flask import Flask, request, render_template
import os
import pickle

app = Flask(__name__)

# Load models
MODEL_PATH = "app/FakeNewsDetectorAPI/models/"
VECTORIZER_FILE = os.path.join(MODEL_PATH, "vectorizer_model.pkl")
MODEL_FILE = os.path.join(MODEL_PATH, "nb_model.pkl")

try:
    with open(VECTORIZER_FILE, "rb") as vf:
        vectorizer = pickle.load(vf)

    with open(MODEL_FILE, "rb") as mf:
        model = pickle.load(mf)

    print("Models loaded successfully!")
except Exception as e:
    print("Error loading models:", str(e))
    exit(1)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text")

    if not text:
        return render_template("index.html", error="Please enter news text.")

    # Convert text to vector
    text_vectorized = vectorizer.transform([text])

    # Predict
    prediction = model.predict(text_vectorized)[0]
    result = "FAKE" if prediction == 1 else "REAL"

    return render_template("result.html", result=result)  

if __name__ == "__main__":
    app.run(debug=True)
