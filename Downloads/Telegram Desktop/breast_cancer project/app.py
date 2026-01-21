from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("model/breast_cancer_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        radius = float(request.form["radius"])
        texture = float(request.form["texture"])
        perimeter = float(request.form["perimeter"])
        area = float(request.form["area"])
        compactness = float(request.form["compactness"])

        features = np.array([[radius, texture, perimeter, area, compactness]])
        features = scaler.transform(features)

        result = model.predict(features)[0]
        prediction = "Malignant" if result == 1 else "Benign"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
