
import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from predict import predict_image, predict_audio, predict_fusion

APP_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
UPLOAD_FOLDER = os.path.join(ROOT_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_IMAGE = {"png","jpg","jpeg"}
ALLOWED_AUDIO = {"wav"}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  

def allowed_file(filename, allowed_set):
    return "." in filename and filename.rsplit(".",1)[1].lower() in allowed_set

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    image_file = request.files.get("image")
    audio_file = request.files.get("audio")

    image_path = None
    audio_path = None

    if image_file and image_file.filename != "" and allowed_file(image_file.filename, ALLOWED_IMAGE):
        fname = secure_filename("uploaded_image_" + image_file.filename)
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
        image_file.save(image_path)

    if audio_file and audio_file.filename != "" and allowed_file(audio_file.filename, ALLOWED_AUDIO):
        fname = secure_filename("uploaded_audio_" + audio_file.filename)
        audio_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
        audio_file.save(audio_path)

    
    if image_path and audio_path:
        res = predict_fusion(image_path, audio_path)
    elif image_path:
        res = {"image": predict_image(image_path)}
    elif audio_path:
        res = {"audio": predict_audio(audio_path)}
    else:
        return redirect(url_for("index"))

    return render_template("result.html", result=res)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)


