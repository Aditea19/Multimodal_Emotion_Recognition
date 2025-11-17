# Multimodal Emotion Recognition System

This project implements a multimodal emotion recognition system capable of identifying human emotions using three different inputs:

- Facial images  
- Speech audio (.wav files)  
- A fusion of both modalities  

The system integrates deep learning models with a Flask-based web interface, making it possible to evaluate each modality independently as well as in combination.

---

## Scope of the Project

The aim of this project is to explore whether combining multiple modalities can improve emotion recognition performance compared to using a single source of input. Facial expressions and speech signals both contain emotional cues, but each has limitations when used alone. This system demonstrates how multimodal learning can strengthen prediction reliability, especially in challenging real-world scenarios where one modality may be noisy or unavailable.

The project covers the complete pipeline:

- Dataset preparation  
- Model training (image, audio, and fusion)  
- Feature extraction using YAMNet  
- Model evaluation  
- Web deployment using Flask  
- User-facing interface for testing  
- Deployment-ready structure for platforms like Render or GitHub  

---

## System Capabilities

- Emotion prediction from **images** using a CNN-based classifier  
- Emotion prediction from **audio** using YAMNet embeddings + a dense classifier  
- **Fusion model** combining image and audio predictions through a weighted score  
- Web-based interface for uploading files  
- Clear output with predicted label and confidence  
- Supports image-only, audio-only, or combined fusion prediction  

---

## Technical Overview

**Image Model**  
A convolutional neural network trained on facial emotion datasets. The model outputs a probability vector across six emotion classes.

**Audio Model**  
Audio features are extracted using YAMNet (TensorFlow Hub). The embeddings are passed to a classifier trained on speech-based emotion data.

**Fusion Mechanism**  
Fusion is implemented using a weighted approach, where softmax outputs from the image and audio models are combined:


Weights can be adjusted depending on modality importance.

**Web Application**  
The Flask backend handles file uploads, runs model inference, and returns structured predictions to the frontend. The interface is simple, responsive, and suitable for deployment.

---

## Model Performance (Summary)

You can update this section with your final results. For example:

- **Image Model Accuracy (test set):** 81%  
- **Audio Model Accuracy (test set):** 55%  
- **Fusion Model Accuracy (test set):** 87%  

---

## Project Structure


---

## Technologies Used

- Python  
- TensorFlow / Keras  
- TensorFlow Hub (YAMNet)  
- NumPy  
- Librosa  
- OpenCV  
- Flask  
- HTML, CSS, JavaScript  

---

## Deployment

The project is structured to run locally or be deployed to platforms like Render.  
A `Procfile` and `requirements.txt` are included to support server-side execution with Gunicorn.

---

## Author

Aditi Arya  
