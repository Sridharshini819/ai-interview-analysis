import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
for k in list(sys.modules.keys()):
    if 'tensorflow' in k:
        del sys.modules[k]

import re
from collections import Counter
import json
import pickle
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import whisper

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.25)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.25)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5   = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6   = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(0.25)
        self.fc1   = nn.Linear(256 * 6 * 6, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.drop4 = nn.Dropout(0.5)
        self.fc2   = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop1(self.pool1(x))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.drop2(self.pool2(x))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.drop3(self.pool3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.drop4(x)
        x = self.fc2(x)
        return x


emotion_model = EmotionCNN(num_classes=7).to(device)
_ckpt = torch.load(os.path.join(MODELS_DIR, 'emotion_final.pt'), map_location=device)
emotion_model.load_state_dict(_ckpt['model_state'])
emotion_model.eval()

with open(os.path.join(MODELS_DIR, 'eye_contact_config.pkl'), 'rb') as f:
    eye_config = pickle.load(f)
LEFT_IRIS  = eye_config['LEFT_IRIS']
RIGHT_IRIS = eye_config['RIGHT_IRIS']
LEFT_EYE   = eye_config['LEFT_EYE']
RIGHT_EYE  = eye_config['RIGHT_EYE']

with open(os.path.join(MODELS_DIR, 'posture_config.pkl'), 'rb') as f:
    posture_config = pickle.load(f)
NOSE           = posture_config['NOSE']
LEFT_SHOULDER  = posture_config['LEFT_SHOULDER']
RIGHT_SHOULDER = posture_config['RIGHT_SHOULDER']
LEFT_EAR       = posture_config['LEFT_EAR']
RIGHT_EAR      = posture_config['RIGHT_EAR']

with open(os.path.join(MODELS_DIR, 'filler_words.json'), 'r') as f:
    FILLER_WORDS = json.load(f)

WHISPER_DIR   = os.path.join(MODELS_DIR, 'whisper')
whisper_model = whisper.load_model('base', download_root=WHISPER_DIR)

EMOTION_LABELS = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear',
    3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'
}

import urllib.request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

FACE_MODEL_PATH = os.path.join(MODELS_DIR, 'face_landmarker.task')
POSE_MODEL_PATH = os.path.join(MODELS_DIR, 'pose_landmarker.task')

if not os.path.exists(FACE_MODEL_PATH):
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        FACE_MODEL_PATH
    )
if not os.path.exists(POSE_MODEL_PATH):
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
        POSE_MODEL_PATH
    )

_face_landmarker_options = mp_vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
    running_mode=mp_vision.RunningMode.IMAGE,
    num_faces=1,
)
_pose_landmarker_options = mp_vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
    running_mode=mp_vision.RunningMode.IMAGE,
    num_poses=1,
)
_face_mesh = mp_vision.FaceLandmarker.create_from_options(_face_landmarker_options)
_pose = mp_vision.PoseLandmarker.create_from_options(_pose_landmarker_options)


def get_eye_contact_score(face_landmarks, frame_w, frame_h):
    lm = face_landmarks

    def get_center(indices):
        xs = [lm[i].x * frame_w for i in indices]
        ys = [lm[i].y * frame_h for i in indices]
        return np.array([np.mean(xs), np.mean(ys)])

    def get_eye_width(indices):
        pts = np.array([[lm[i].x * frame_w, lm[i].y * frame_h] for i in indices])
        return np.linalg.norm(pts[0] - pts[3])

    left_iris_center  = get_center(LEFT_IRIS)
    right_iris_center = get_center(RIGHT_IRIS)
    left_eye_center   = get_center(LEFT_EYE)
    right_eye_center  = get_center(RIGHT_EYE)
    left_eye_width    = get_eye_width(LEFT_EYE)
    right_eye_width   = get_eye_width(RIGHT_EYE)

    left_offset  = np.linalg.norm(left_iris_center - left_eye_center) / (left_eye_width + 1e-6)
    right_offset = np.linalg.norm(right_iris_center - right_eye_center) / (right_eye_width + 1e-6)
    avg_offset = (left_offset + right_offset) / 2.0
    return round(max(0.0, 1.0 - avg_offset * 1.5), 3)


def get_posture_score(pose_landmarks, frame_w, frame_h):
    lm = pose_landmarks

    def pt(idx):
        return np.array([lm[idx].x * frame_w, lm[idx].y * frame_h])

    nose       = pt(NOSE)
    l_shoulder = pt(LEFT_SHOULDER)
    r_shoulder = pt(RIGHT_SHOULDER)
    l_ear      = pt(LEFT_EAR)
    r_ear      = pt(RIGHT_EAR)

    scores = []
    shoulder_diff = abs(l_shoulder[1] - r_shoulder[1])
    shoulder_dist = np.linalg.norm(l_shoulder - r_shoulder) + 1e-6
    scores.append(max(0.0, 1.0 - (shoulder_diff / shoulder_dist) * 3.0))

    ear_diff = abs(l_ear[1] - r_ear[1])
    ear_dist = np.linalg.norm(l_ear - r_ear) + 1e-6
    scores.append(max(0.0, 1.0 - (ear_diff / ear_dist) * 3.0))

    mid_shoulder_x = (l_shoulder[0] + r_shoulder[0]) / 2.0
    head_offset = abs(nose[0] - mid_shoulder_x) / shoulder_dist
    scores.append(max(0.0, 1.0 - head_offset * 2.0))

    return round(float(np.mean(scores)), 3)


def get_emotion_score(frame_gray, face_landmarks, frame_w, frame_h):
    lm = face_landmarks
    xs = [lm[i].x * frame_w for i in range(468)]
    ys = [lm[i].y * frame_h for i in range(468)]
    x1, x2 = int(max(0, min(xs))), int(min(frame_w, max(xs)))
    y1, y2 = int(max(0, min(ys))), int(min(frame_h, max(ys)))

    if x2 - x1 < 10 or y2 - y1 < 10:
        return 'Neutral', 0.0

    face = frame_gray[y1:y2, x1:x2]
    face = cv2.resize(face, (48, 48))
    face = face.astype(np.float32) / 255.0
    face = torch.tensor(face).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        out = emotion_model(face)
        probs = torch.softmax(out, dim=1)
        conf, pred = probs.max(dim=1)

    label = EMOTION_LABELS[pred.item()]
    return label, round(conf.item(), 3)


def process_frame(frame_bgr):
    h, w = frame_bgr.shape[:2]
    frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    result = {
        'eye_contact':    0.0,
        'posture':        0.0,
        'emotion':        'Neutral',
        'emotion_conf':   0.0,
        'face_detected':  False,
        'pose_detected':  False,
    }

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    face_results = _face_mesh.detect(mp_image)
    if face_results.face_landmarks:
        face_lm = face_results.face_landmarks[0]
        result['face_detected'] = True
        result['eye_contact']   = get_eye_contact_score(face_lm, w, h)
        emotion_label, emotion_conf = get_emotion_score(frame_gray, face_lm, w, h)
        result['emotion']      = emotion_label
        result['emotion_conf'] = emotion_conf

    pose_results = _pose.detect(mp_image)
    if pose_results.pose_landmarks:
        result['pose_detected'] = True
        result['posture'] = get_posture_score(pose_results.pose_landmarks[0], w, h)

    return result


def analyze_speech(audio_path):
    import subprocess
    converted = audio_path.rsplit('.', 1)[0] + '_conv.wav'
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', audio_path,
            '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', converted
        ], capture_output=True, timeout=60)
        use_path = converted if os.path.exists(converted) else audio_path
    except Exception:
        use_path = audio_path

    try:
        result = whisper_model.transcribe(use_path, language='en', fp16=False, without_timestamps=True)
    except Exception as e:
        return {
            'transcript': f'Audio error: {str(e)}',
            'filler_counts': {}, 'total_fillers': 0,
            'word_count': 0, 'duration_sec': 0,
            'wpm': 0, 'wpm_score': 0.0, 'filler_score': 0.0
        }

    transcript = result['text'].lower().strip()
    segments   = result.get('segments', [])
    duration   = segments[-1]['end'] if segments else 1.0

    filler_counts, total_fillers = {}, 0
    for filler in FILLER_WORDS:
        pattern = r'\b' + re.escape(filler) + r'\b'
        count = len(re.findall(pattern, transcript))
        if count > 0:
            filler_counts[filler] = count
            total_fillers += count

    word_count = len(transcript.split())
    wpm = round((word_count / max(duration, 1)) * 60, 1)

    if 120 <= wpm <= 160:
        wpm_score = 1.0
    elif 100 <= wpm < 120 or 160 < wpm <= 180:
        wpm_score = 0.75
    elif 80 <= wpm < 100 or 180 < wpm <= 200:
        wpm_score = 0.5
    else:
        wpm_score = 0.25

    filler_rate  = (total_fillers / max(word_count, 1)) * 100
    filler_score = max(0.0, 1.0 - filler_rate * 0.15)

    return {
        'transcript':    transcript,
        'filler_counts': filler_counts,
        'total_fillers': total_fillers,
        'word_count':    word_count,
        'duration_sec':  round(duration, 1),
        'wpm':           wpm,
        'wpm_score':     round(wpm_score, 3),
        'filler_score':  round(filler_score, 3),
    }


def get_overall_score(eye_score, posture_score, filler_score, wpm_score):
    score = (
        posture_score * 0.35 +
        filler_score  * 0.35 +
        wpm_score      * 0.30
    ) * 100
    return round(score, 1)


def get_tips(eye_score, posture_score, emotion_label, filler_result):
    tips = []

    if posture_score < 0.4:
        tips.append(('Posture', 'warning', 'Sit up straight. Keep your shoulders level and head upright.'))
    elif posture_score < 0.7:
        tips.append(('Posture', 'info', 'Posture looks okay. Try to keep your head centered.'))
    else:
        tips.append(('Posture', 'success', 'Great posture. You look confident and professional.'))

    if emotion_label in ['Angry', 'Disgust', 'Fear']:
        tips.append(('Expression', 'warning', f'You look {emotion_label.lower()}. Take a breath and try to relax your face.'))
    elif emotion_label == 'Sad':
        tips.append(('Expression', 'info', 'Try to show more energy. A slight smile goes a long way.'))
    elif emotion_label in ['Happy', 'Neutral']:
        tips.append(('Expression', 'success', 'Your expression looks great. Natural and approachable.'))

    total_fillers = filler_result.get('total_fillers', 0)
    if total_fillers > 5:
        top_fillers = sorted(filler_result.get('filler_counts', {}).items(), key=lambda x: x[1], reverse=True)[:2]
        filler_str = ', '.join([f'"{w}"' for w, _ in top_fillers])
        tips.append(('Speech', 'warning', f'Too many filler words ({filler_str}). Pause silently instead.'))
    elif total_fillers > 2:
        tips.append(('Speech', 'info', 'A few filler words detected. Try replacing them with a pause.'))
    else:
        tips.append(('Speech', 'success', 'Very clean speech. Minimal filler words detected.'))

    wpm = filler_result.get('wpm', 0)
    if wpm < 100:
        tips.append(('Pace', 'warning', f'Speaking too slowly ({wpm} WPM). Pick up the pace slightly.'))
    elif wpm > 180:
        tips.append(('Pace', 'warning', f'Speaking too fast ({wpm} WPM). Slow down so the interviewer can follow.'))
    else:
        tips.append(('Pace', 'success', f'Good speaking pace ({wpm} WPM). Clear and easy to follow.'))

    return tips


def analyze_video(video_path, max_duration_sec=120, sample_interval_sec=1.0):
    import subprocess
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            'error': 'Could not open video file.',
            'eye_avg': 0.0, 'posture_avg': 0.0, 'dominant_emotion': 'Neutral',
            'frames_analyzed': 0, 'faces_detected': 0, 'poses_detected': 0,
            'speech': None
        }

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    if duration_sec > max_duration_sec:
        cap.release()
        return {
            'error': f'Video is {duration_sec:.0f}s, longer than the {max_duration_sec}s limit. Please trim it and try again.',
            'eye_avg': 0.0, 'posture_avg': 0.0, 'dominant_emotion': 'Neutral',
            'frames_analyzed': 0, 'faces_detected': 0, 'poses_detected': 0,
            'speech': None
        }

    frame_interval = max(1, int(fps * sample_interval_sec))
    eye_scores, posture_scores, emotions = [], [], []
    faces_detected, poses_detected, frames_analyzed = 0, 0, 0

    video_face_options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
    )
    video_pose_options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
    )
    video_face_mesh = mp_vision.FaceLandmarker.create_from_options(video_face_options)
    video_pose = mp_vision.PoseLandmarker.create_from_options(video_pose_options)

    def process_frame_isolated(frame_bgr):
        h, w = frame_bgr.shape[:2]
        frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        result = {
            'eye_contact': 0.0, 'posture': 0.0, 'emotion': 'Neutral',
            'emotion_conf': 0.0, 'face_detected': False, 'pose_detected': False,
        }
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        face_results = video_face_mesh.detect(mp_image)
        if face_results.face_landmarks:
            face_lm = face_results.face_landmarks[0]
            result['face_detected'] = True
            result['eye_contact'] = get_eye_contact_score(face_lm, w, h)
            emotion_label, emotion_conf = get_emotion_score(frame_gray, face_lm, w, h)
            result['emotion'] = emotion_label
            result['emotion_conf'] = emotion_conf
        pose_results = video_pose.detect(mp_image)
        if pose_results.pose_landmarks:
            result['pose_detected'] = True
            result['posture'] = get_posture_score(pose_results.pose_landmarks[0], w, h)
        return result

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frames_analyzed += 1
            try:
                res = process_frame_isolated(frame)
                if res['face_detected']:
                    faces_detected += 1
                    eye_scores.append(res['eye_contact'])
                    emotions.append(res['emotion'])
                if res['pose_detected']:
                    poses_detected += 1
                    posture_scores.append(res['posture'])
            except Exception:
                pass
        frame_idx += 1
    cap.release()
    video_face_mesh.close()
    video_pose.close()

    eye_avg = round(float(np.mean(eye_scores)), 3) if eye_scores else 0.0
    posture_avg = round(float(np.mean(posture_scores)), 3) if posture_scores else 0.0
    dominant_emotion = Counter(emotions).most_common(1)[0][0] if emotions else 'Neutral'

    audio_path = video_path.rsplit('.', 1)[0] + '_audio.wav'
    speech_result = None
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', audio_path
        ], capture_output=True, timeout=120)
        if os.path.exists(audio_path):
            speech_result = analyze_speech(audio_path)
    except Exception:
        speech_result = None

    return {
        'error': None,
        'eye_avg': eye_avg,
        'posture_avg': posture_avg,
        'dominant_emotion': dominant_emotion,
        'frames_analyzed': frames_analyzed,
        'faces_detected': faces_detected,
        'poses_detected': poses_detected,
        'duration_sec': round(duration_sec, 1),
        'speech': speech_result
    }
