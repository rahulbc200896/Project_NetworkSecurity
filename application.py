import os
import sys
import pandas as pd

from flask import Flask, request, render_template, redirect, url_for, send_file
from flask_cors import CORS
from dotenv import load_dotenv

from networksecurity.exception.exception import Network_Security_Exception
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_obj
from networksecurity.utils.ml_utils.metric.estimator import NetworkModel

# Load env
load_dotenv()

# Flask app
application = Flask(__name__)
app = application
CORS(app)

# ===============================
# HOME
# ===============================
@app.route("/")
def home():
    return redirect("/train")

# ===============================
# TRAIN ROUTE
# ===============================
@app.route("/train", methods=["GET"])
def train():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return "Training completed successfully"
    except Exception as e:
        raise Network_Security_Exception(e, sys)

# ===============================
# PREDICT ROUTE
# ===============================
@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        if request.method == "POST":
            file = request.files["file"]

            if not file:
                return "No file uploaded"

            df = pd.read_csv(file)

            # Load model & preprocessor
            preprocessor = load_obj("final_model/preprocessor.pkl")
            model = load_obj("final_model/model.pkl")

            network_model = NetworkModel(
                preprocessor=preprocessor,
                model=model
            )

            predictions = network_model.Predict(df)
            df["predicted_column"] = predictions

            # Save output
            os.makedirs("prediction_output", exist_ok=True)
            output_path = "prediction_output/output.csv"
            df.to_csv(output_path, index=False)

            return render_template(
                "table_flask.html",
                table=df.to_html(classes="table table-striped", index=False)
            )

        return render_template("upload.html")

    except Exception as e:
        raise Network_Security_Exception(e, sys)

# ===============================
# MAIN
# ===============================
#if __name__ == "__main__":
    #app.run(host="0.0.0.0", port=8000, debug=True)

if __name__ =='__main__':
    port= int(os.environ.get("PORT",8080))
    app.run(host="0.0.0.0",port=port, debug=True)
