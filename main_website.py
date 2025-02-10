import tensorflow_hub as hub
import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import imghdr
from PIL import Image 
import os 
import pillow_heif
import base64 
# import cv2 as cv
import io
from imageio import imread
pillow_heif.register_heif_opener()
app = Flask(__name__)
IMAGE_SHAPE = (224,224)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
pets_labels_dict = {
   0    :   'Abyssinian'                 ,  
   1    :   'Bengal'                     ,
   2    :   'Birman'                     ,
   3    :   'Bombay'                     ,
   4    :   'British_Shorthair'          ,
   5    :   'Egyptian_Mau'               ,
   6    :   'Maine_Coon'                 ,
   7    :   'Persian'                    ,
   8    :   'Ragdoll'                    ,
   9    :   'Russian_Blue'               ,
   10    :   'Siamese'                    ,
   11    :   'Sphynx'                     ,
}

model = load_model('Model_Penelitian_Final.keras',
				   custom_objects={'KerasLayer': hub.KerasLayer})
model.make_predict_function()

def predict_label(img_path):
	img_data=img_path.decode('utf-8')
	img = imread(io.BytesIO(base64.b64decode(img_data)))
	gambar = image.img_to_array(img)/255.0
	p = model.predict(gambar[np.newaxis, ...])
	print(p)
	predicted_label = np.argmax(p)
	return pets_labels_dict[predicted_label]


@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


@app.route("/about", methods=['GET', 'POST'])
def about_page():
	return render_template("about.html")


@app.route("/predict", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		type_image = imghdr.what(img)
		if type_image != 'jpg'	:
			im = Image.open(img)
			rgb_im = im.convert("RGB")
			rgb_im=rgb_im.resize(IMAGE_SHAPE)
			data = io.BytesIO()
			rgb_im.save(data, "JPEG")
			encoded_img_data = base64.b64encode(data.getvalue())
			p = predict_label(encoded_img_data) 
	return render_template("hasilpredik.html", img_data=encoded_img_data.decode('utf-8'), 
						prediction= p)




@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'),500

if __name__ =='__main__':
	#app.debug = True
	app.run(debug = False,host='0.0.0.0')


