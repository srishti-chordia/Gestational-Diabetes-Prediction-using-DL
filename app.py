from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)
model = load_model("gdm_model.h5")  # Load your trained model
scaler = joblib.load("scaler.joblib")  # Load the fitted scaler


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Get data from form using case-sensitive keys to match the dataset
    AP = float(request.form["AP"])
    ICP = float(request.form["ICP"])
    TD = float(request.form["TD"])
    Eclampsia = float(request.form["Eclampsia"])
    Age = float(request.form["Age"])
    BMI = float(request.form["BMI"])
    ALT = float(request.form["ALT"])
    AST = float(request.form["AST"])
    GGT = float(request.form["GGT"])
    ALP = float(request.form["ALP"])
    TBA = float(request.form["TBA"])
    UREA = float(request.form["UREA"])
    CREA = float(request.form["CREA"])
    UA = float(request.form["UA"])
    BMG = float(request.form["BMG"])
    A1MG = float(request.form["A1MG"])
    CysC = float(request.form["CysC"])
    FPG = float(request.form["FPG"])

    # Combine clinical and biochemical data
    clinical_data = np.array([AP, ICP, TD, Eclampsia, Age, BMI])
    biochemical_data = np.array(
        [
            ALT,
            AST,
            GGT,
            ALP,
            TBA,
            UREA,
            CREA,
            UA,
            BMG,
            A1MG,
            CysC,
            FPG,
        ]
    )

    # Combine clinical and biochemical data
    input_data = np.concatenate((clinical_data, biochemical_data), axis=0).reshape(
        1, -1
    )

    # Standardize the input data using the same scaler
    input_data_scaled = scaler.transform(input_data)

    # Predict GDM probability
    prediction = model.predict(input_data_scaled)
    gdm_probability = float(prediction[0][0])

    # Return prediction as JSON response
    return jsonify({"gdm_probability": gdm_probability})


if __name__ == "__main__":
    app.run(debug=True)
