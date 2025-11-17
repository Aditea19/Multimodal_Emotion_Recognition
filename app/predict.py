
import os
import numpy as np
import traceback
import cv2
import librosa
import tensorflow as tf
import tensorflow_hub as hub

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad"]


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMAGE_MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_image_model.h5")
AUDIO_MODEL_PATH = os.path.join(ROOT_DIR, "models", "audio_model.h5")


image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH)
audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)


yamnet_path = os.path.join(BASE_DIR, "../models/yamnet")
yamnet_model = tf.saved_model.load(yamnet_path)


def preprocess_image(path, size=(128, 128)):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Image not readable: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size[1], size[0]))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, 0)


def wav_to_yamnet_embedding(wav_path):
    
    wav, sr = librosa.load(wav_path, sr=16000, mono=True)
    wav = wav.astype('float32')
    scores, embeddings, spectrogram = yamnet_model(wav)
    emb = np.mean(embeddings.numpy(), axis=0)   
    return np.expand_dims(emb, 0).astype('float32')


def predict_image(path):
    try:
        x = preprocess_image(path)
        preds = image_model.predict(x, verbose=0)[0]
        idx = int(np.argmax(preds))
        return {"label": CLASS_NAMES[idx], "confidence": float(np.max(preds)), "probs": preds.tolist()}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

def predict_audio(path):
    try:
        emb = wav_to_yamnet_embedding(path)
        preds = audio_model.predict(emb, verbose=0)[0]
        idx = int(np.argmax(preds))
        return {"label": CLASS_NAMES[idx], "confidence": float(np.max(preds)), "probs": preds.tolist()}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

def predict_fusion(image_path, audio_path, w_img=0.8, w_aud=0.2):
    
    img_res = predict_image(image_path) if image_path else None
    aud_res = predict_audio(audio_path) if audio_path else None

    if (img_res and "probs" in img_res) and (aud_res and "probs" in aud_res):
        p_img = np.array(img_res["probs"], dtype=np.float32)
        p_aud = np.array(aud_res["probs"], dtype=np.float32)
        fused = w_img * p_img + w_aud * p_aud
        idx = int(np.argmax(fused))
        return {
            "fusion": {"label": CLASS_NAMES[idx], "confidence": float(np.max(fused)), "probs": fused.tolist()},
            "image": img_res,
            "audio": aud_res
        }
    else:
        
        return {"image": img_res, "audio": aud_res}

