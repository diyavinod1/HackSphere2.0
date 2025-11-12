# NeuroWave<br><h3>AI-based Early Detection of Parkinson’s from Voice</h3>

<i>“Your Voice Can Speak for Your Health.”</i><br>
A web-based tool that uses Machine Learning to analyze subtle voice variations and predict early signs of Parkinson’s disease.

## 🩺 Overview

NeuroWave is a machine learning–powered web application that detects early signs of Parkinson’s disease using voice analysis.<br>
The app allows users to record their voice or upload an audio file, which is then analyzed using AI models to predict whether the person might be at risk.

## 🧩 Problem Statement

Parkinson’s disease is often diagnosed only after physical symptoms appear, by which time significant neurological damage has occurred.<br>
However, early indicators such as changes in speech pattern, jitter, shimmer, and pitch can be detected using advanced audio analysis.

## 💡 Motivation

- Current diagnosis methods are costly, invasive, and slow.<br>
- Voice is an easily accessible, non-invasive biomarker.<br>
- AI can capture minute changes in voice frequency and amplitude that humans often miss.

## 🚀 Features

- Voice Recording: Capture audio directly from the browser.<br>
- Audio Upload: Option to upload .wav files.<br>
- AI Prediction: ML model predicts “Healthy” or “At Risk.”<br>
- Result Visualization: Graphs or confidence scores.<br>
- Privacy Preserving: Runs locally, no cloud storage.

## 🧠 Tech Stack
Frontend:	HTML, CSS, JavaScript<br>
Backend:	Flask (Python)<br>
ML Libraries:	scikit-learn, librosa, numpy, pandas, matplotlib<br>
Model:	Random Forest Classifier<br>
Dataset:	UCI Parkinson’s Dataset<br>

## ⚙️ System Workflow

Step-by-Step Flow:<br>
1️⃣ User records or uploads voice<br>
2️⃣ Flask backend extracts features using librosa<br>
3️⃣ Trained Random Forest model predicts health status<br>
4️⃣ Result is displayed in the web UI<br>

## 🤖 Model Details

Algorithm: Random Forest Classifier<br>
Accuracy: ~90% on UCI dataset<br>
Key Features Used: Jitter, Shimmer, Pitch variation

Goal: Binary classification — Healthy vs Parkinson’s Risk

## 🧩 Installation & Usage
Prerequisites

Make sure you have the following installed:

- Python 3.x<br>
- pip<br>
- Virtual environment (optional but recommended)

Steps to Run:

1️⃣ Clone the repository
   git clone https://github.com/diyavinod1/HackSphere2.0.git <br>
   cd HackSphere2.0

2️⃣ Install dependencies
   pip install flask librosa soundfile scikit-learn numpy pandas matplotlib requests  <br>
   For better audio support, also install:<br>
   pip install ffmpeg-python

3️⃣ Run the Flask app<br>
   python3 app.py

4️⃣ Open in browser
   http://localhost:5000/

## 📈 Results

Achieved ~90% accuracy on the UCI dataset.<br>
Real-time prediction through Flask-based web interface.<br>
Successful detection of Parkinson’s voice traits through ML.

Sample Output:<br>
Result: At Risk of Parkinson’s<br>
Confidence: 87%

## 🔭 Future Scope

- Incorporate deep learning models (CNN/LSTM) for feature learning.<br>
- Create mobile app version for live screening.<br>
- Integrate with healthcare APIs for clinical collaboration.<br>
- Collect real-world voice samples for model retraining.

## 👩‍💻 Contributors

Team NeuroWave

Diya Vinod — ML Model & Backend<br>
Dharanipriya K — Frontend UI<br>
Haripriya K — Dataset & Research


<i>“NeuroWave — Turning Voice into a Diagnostic Signal.”</i>
