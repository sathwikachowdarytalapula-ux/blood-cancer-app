from flask import Flask, request, render_template_string
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
model = load_model("model.h5")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Blood Cancer Detection</title>
    <style>
        body {
            font-family: Arial;
            text-align: center;
            background: #f4f6f9;
        }
        .container {
            margin-top: 50px;
        }
        h1 {
            color: #333;
        }
        .box {
            background: white;
            padding: 30px;
            border-radius: 10px;
            width: 400px;
            margin: auto;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
        }
        .result {
            font-size: 20px;
            margin-top: 20px;
        }
        img {
            margin-top: 20px;
            max-width: 300px;
        }
    </style>
</head>
<body>

<div class="container">
    <div class="box">
        <h1>🧬 Blood Cancer Detection</h1>

        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" required><br><br>
            <input type="submit" value="Analyze">
        </form>

        {% if result %}
            <div class="result">
                <p><b>{{ result }}</b></p>
                <p>Confidence: {{ confidence }}%</p>
            </div>
        {% endif %}

        <h3>Training Accuracy Graph</h3>
        <img src="/static/accuracy.png">

    </div>
</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence_percent = None

    if request.method == "POST":
        file = request.files["file"]

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)[0][0]

        confidence_percent = round(float(prediction) * 100, 2)

        if prediction > 0.5:
            result = "⚠️ Leukemia Detected"
        else:
            result = "✅ Normal"

    return render_template_string(
        HTML,
        result=result,
        confidence=confidence_percent
    )

if __name__ == "__main__":
    app.run(debug=True)