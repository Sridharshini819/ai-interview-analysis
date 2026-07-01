import os
import json
import uuid
from datetime import datetime

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'sessions')
os.makedirs(SESSIONS_DIR, exist_ok=True)


def save_session(overall, eye_avg, pos_avg, speech_result, dom_emo, tips):
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:6]
    data = {
        'session_id':    session_id,
        'timestamp':      datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'overall_score':  overall,
        'eye_contact':    round(eye_avg * 100, 1),
        'posture':        round(pos_avg * 100, 1),
        'emotion':        dom_emo,
        'wpm':            speech_result.get('wpm', 0),
        'total_fillers':  speech_result.get('total_fillers', 0),
        'transcript':     speech_result.get('transcript', ''),
        'tips': [
            {'category': c, 'level': l, 'tip': t} for c, l, t in tips
        ],
    }
    path = os.path.join(SESSIONS_DIR, f'{session_id}.json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    return session_id, path


def load_all_sessions():
    sessions = []
    if not os.path.exists(SESSIONS_DIR):
        return sessions
    for fname in sorted(os.listdir(SESSIONS_DIR), reverse=True):
        if fname.endswith('.json'):
            try:
                with open(os.path.join(SESSIONS_DIR, fname), 'r') as f:
                    sessions.append(json.load(f))
            except Exception:
                continue
    return sessions


def delete_session(session_id):
    path = os.path.join(SESSIONS_DIR, f'{session_id}.json')
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def get_session_stats():
    sessions = load_all_sessions()
    if not sessions:
        return None
    scores = [s['overall_score'] for s in sessions]
    total_sessions = len(sessions)
    avg_score  = round(sum(scores) / total_sessions, 1)
    best_score = round(max(scores), 1)
    if total_sessions >= 2:
        improvement = round(scores[0] - scores[-1], 1)
    else:
        improvement = 0.0
    return {
        'total_sessions': total_sessions,
        'avg_score':      avg_score,
        'best_score':     best_score,
        'improvement':    improvement,
    }
