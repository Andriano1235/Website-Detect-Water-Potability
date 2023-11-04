from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open("model_mlp.pkl", 'rb'))


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    feature = [np.array(float_features)]
    prediction = model.predict(feature)
    ans = prediction[0]
    if ans == 1:
        ans = "Layak untuk diminum"
    else:
        ans = "Tidak layak untuk diminum"
    return render_template('index.html', prediction_text="{}".format(ans))


if __name__ == "__main__":
    app.run()
