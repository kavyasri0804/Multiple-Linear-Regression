import pickle

import flask
from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import sklearn
app=Flask(__name__)
with open("MLR_MODEL.pkl","rb") as f:
    model = pickle.load(f)
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict",methods=["GET","POST"])
def predict():
    R_DSpend=request.form.get("R&D")
    Administration=request.form.get("Administration")
    Marketing=request.form.get("Marketing")
    State=request.form.get("State")
    sol=model.predict(R_DSpend,Administration,Marketing,State).reshape(-1,1)
    return render_template("index.html",prediction_text=sol)
if __name__ == "__main__":
    app.run(debug=True)