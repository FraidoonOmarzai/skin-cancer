from flask import Flask, render_template, request
from keras.models import load_model
import cv2
from PIL import Image
import numpy as np

app = Flask(__name__)


def predict_label(img_path):
	model = load_model("model/skin_model.h5")

	img = cv2.imread(img_path)
	img = Image.fromarray(img)
	img = img.resize((224, 224))
	img = np.array(img)
	img = np.expand_dims(img, axis=0)

	pred = model.predict(img)
	return pred[0]


@app.route("/")
def main():
	return render_template("skin.html")


@app.route("/predictSkinC", methods = ['GET', 'POST'])
def get_output():
	dic ={ 0:"Benign", 1:"Malignant!"}

	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename
		img.save(img_path)

		p = predict_label(img_path)[0]
		print(np.round(p))

	return render_template("skin.html", prediction = dic[np.round(p)], img_path = img_path)


if __name__ =='__main__':
	app.run(debug = True)
