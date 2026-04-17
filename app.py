from flask import Flask, request, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# 🔥 IMPORTANT for deployment (no GUI)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load model safely
MODEL_PATH = "model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("model.h5 not found!")

model = load_model(MODEL_PATH)

# Upload folder
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs("static", exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence_percent = None
    image_path = None
    leukemia_percent = None
    normal_percent = None
    chart = None

    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename != "":
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            image_path = filepath.replace("static/", "")

            # Read image
            img = cv2.imread(filepath)

            if img is None:
                return render_template("index.html", result="❌ Invalid Image")

            # Preprocess
            img = cv2.resize(img, (128, 128))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            # Predict
            prediction = model.predict(img)[0][0]

            leukemia_prob = float(prediction)
            normal_prob = 1 - leukemia_prob

            leukemia_percent = round(leukemia_prob * 100, 2)
            normal_percent = round(normal_prob * 100, 2)

            # Result
            if prediction > 0.5:
                result = "⚠️ Leukemia Detected"
                confidence_percent = leukemia_percent
            else:
                result = "✅ Normal"
                confidence_percent = normal_percent

            # 🔥 PIE CHART
            labels = ['Normal', 'Leukemia']
            sizes = [normal_percent, leukemia_percent]

            plt.figure()
            plt.pie(sizes, labels=labels, autopct='%1.1f%%')
            plt.title('Prediction Distribution')

            chart_path = os.path.join("static", "pie.png")
            plt.savefig(chart_path)
            plt.close()

            chart = "pie.png"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence_percent,
        image=image_path,
        leukemia=leukemia_percent,
        normal=normal_percent,
        chart=chart
    )

# 🚀 REQUIRED FOR RENDER
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)