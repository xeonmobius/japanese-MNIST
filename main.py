from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import io, base64, os 
from base64 import decodestring
from PIL import Image
import numpy as np
import pandas as pd

app = Flask(__name__)

model = tf.keras.models.load_model("model.h5")
class_map = pd.read_csv('k49_classmap.csv')

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/get_prediction', methods=['POST', 'GET'])
def get_prediction():
    if request.method == "POST":
        data = request.get_data(as_text=True)
        buf = io.BytesIO(base64.b64decode(data.split(",")[-1]))
        img = Image.open(buf)
        img = img.resize([28,28])
        
        corrected_img = Image.new("RGBA", (28, 28), "white")
        corrected_img.paste(img, (0,0), img)
        corrected_img = np.asarray(corrected_img)

        corrected_img = corrected_img[:, :, 0]
        corrected_img = np.invert(corrected_img)
        corrected_img = corrected_img / 255.0
        corrected_img = corrected_img.reshape(1, 28, 28, 1)

        pred = model.predict(corrected_img, verbose=0)
        char_pred = np.argmax(pred)
        confidence = str(np.max(pred)*100)[0:2]
        
        character = class_map.loc[char_pred, "char"]

    return {"prediction": f"Predicting: {character}",
            "confidence": f"Confidence: {confidence}%"}

if __name__ == "__main__":
    app.run(debug=False)