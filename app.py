from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import csv

app = Flask(__name__)

weights = np.genfromtxt("WEIGHTS_FILE.csv", dtype=np.float64, delimiter=",")

def predict(X):
    X = np.insert(X, 0, 1, axis=1)
    prediction = X.dot(weights)
    return prediction

@app.route("/")
def home_page():
    return render_template("index.html")

@app.route("/submit", methods=['POST', 'GET'])
def submit():
    ts_on_site = float(request.form['time-spent-on-site'])
    membership_duration = float(request.form['membership-duration'])
    ts_on_app = float(request.form['time-spent-on-app'])
    session_duration = float(request.form['session-duration'])

    X = np.array([ts_on_site, membership_duration, ts_on_app, session_duration])
    X = np.reshape(X, (1,len(X)))
    prediction = predict(X)[0]
    prediction = round(prediction, 2)

    return render_template("index.html", prediction=f'The predicted annual amount spent by the customer is Rs {prediction}')

if __name__ == "__main__":
    app.run(debug=True)