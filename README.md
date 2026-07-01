# InterviewIQ — AI-Powered Interview Performance Coach

InterviewIQ is an AI-driven web application that helps users practice and improve their interview performance. It analyzes posture, facial expression, and speech delivery using computer vision and speech-to-text transcription, then generates a personalized performance score with actionable coaching tips.

## Features
- Practice Questions — Randomized interview questions across Data Science and Soft Skills categories
- Webcam Snapshot Analysis — Capture a photo while answering to score posture and facial expression
- Video Upload Analysis — Upload a short practice video for frame-by-frame posture and emotion scoring
- Speech Analysis — Record audio in-browser or upload a file to get speaking pace, filler-word detection, and transcript
- Performance Scoring — Combined score based on posture, filler words, and speaking pace with coaching tips
- Session History — Save and review past practice sessions

## Tech Stack
- Frontend: Streamlit
- Emotion Detection: Custom CNN (PyTorch) — 67.19% validation accuracy
- Face and Pose Detection: MediaPipe FaceLandmarker and PoseLandmarker (Tasks API)
- Speech to Text: OpenAI Whisper
- Audio Processing: ffmpeg, librosa, soundfile
- Image Processing: OpenCV
- Deployment: Streamlit Community Cloud via GitHub

## Model Details
- Custom CNN trained on FER2013 and CK+ datasets
- 7 emotion classes: Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral
- 3 versions trained (V1, V2, V3) — V1 selected with 67.19% validation accuracy
- V2 rejected due to overfitting, V3 rejected due to underfitting

## Project Structure
- app.py — Main Streamlit application
- scorer.py — Core scoring logic (vision, speech, overall score)
- session_manager.py — Save and load session history
- question_bank.py — Interview question bank
- requirements.txt — Python dependencies
- packages.txt — System level dependencies
- models/ — Trained model files

## Running Locally
pip install -r requirements.txt
streamlit run app.py

Note: Whisper and MediaPipe task models download automatically on first run.

## Live Demo
https://sridharshini819-ai-interview-analysis.streamlit.app

## Future Enhancements
- Improve emotion model accuracy
- Real-time live video feedback
- Multi-language speech support
- User accounts and cross-device history
- Mobile app version

## Author
Sridharshini
