from flask import Flask,render_template,request
import joblib
import pandas as pd
import numpy as np

app=Flask(__name__)

km=joblib.load("model/rfm_kmeans_model.pkl")
scaler = joblib.load("model/rfm_scaler.pkl")
segment_names = joblib.load("model/segment_names.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():

    recency=float(request.form['recency'])
    monetary=float(request.form['monetary'])
    frequency=float(request.form['frequency'])

    if recency < 0 or frequency < 1 or monetary <= 0:
        return render_template("index.html", error="Please enter valid positive values.")

    data=pd.DataFrame(
        [[recency,frequency,monetary]],
        columns=['recency','frequency','monetary']
    )

    data_log=np.log1p(data)

    data_scaled=scaler.transform(data_log)

    prediction=int(km.predict(data_scaled)[0])
    segment=segment_names[prediction]

    return render_template(
        "index.html",
        segment=segment
    )

if __name__=="__main__":
    app.run(debug=True)