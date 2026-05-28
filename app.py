from flask import Flask, render_template, request
from src.model import load_pipeline
import os

app=Flask(__name__)
pipeline=load_pipeline(os.path.join(os.path.dirname(__file__),"outputs","model.pkl"))

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html", predicted_genre=None)

@app.route("/predict", methods=["POST"])
def predict():
    description= request.form['movie_description']
    predicted_genre=pipeline.predict([description])[0]
    return render_template("index.html", predicted_genre=predicted_genre)


if __name__ == "__main__":
    app.run(debug=True)