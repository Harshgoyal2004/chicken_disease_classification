from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from src.cnnClassifier.pipeline.stage_05_predict import PredictionPipeline
from src.cnnClassifier.utils.common import decodeImage

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class clientApp:
    def __init__(self):
        self.filename = "input_image.jpg"
        self.classifier = PredictionPipeline(filename=self.filename)

@app.route("/", methods=['GET'])
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    try:
        if request.method == 'POST':
            os.system('python3 main.py')
            return "Training successful!!"
        else:
            return render_template("index.html")
    except Exception as e:
        return str(e)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        data = request.get_json(force=True)
        image_b64 = data.get('image') if data else None
        if not image_b64:
            return jsonify({"error": "no image provided"}), 400

        # decode and save image, then run prediction
        decodeImage(image_b64, client_app.filename)
        result = client_app.classifier.predict()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    client_app = clientApp()
    app.run(host='0.0.0.0', port=8080, debug=True)
