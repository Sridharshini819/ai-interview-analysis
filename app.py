import os
import sys
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import tempfile
import numpy as np
import cv2
import streamlit as st
from collections import deque, Counter
from datetime import datetime
from scorer import process_frame, analyze_speech, get_overall_score, get_tips, analyze_video
from session_manager import save_session, load_all_sessions, delete_session, get_session_stats
from question_bank import get_random_question, get_all_questions

st.set_page_config(page_title='InterviewIQ', page_icon='IQ', layout='wide')

st.markdown('''
<style>
[data-testid="stAppViewContainer"]{background:#0d1b2a;min-height:100vh}
.block-container{padding:1rem 2rem}
* {font-family:'Segoe UI','Helvetica Neue',Arial,sans-serif;}
.logo-row{display:flex;align-items:center;justify-content:center;gap:0.7rem;padding:1rem 0 0.2rem}
.logo-mark{background:#3a6ea5;color:#ffffff;font-weight:800;font-size:1.4rem;
padding:0.4rem 0.7rem;border-radius:8px;letter-spacing:1px}
.main-title{font-size:2.4rem;font-weight:700;color:#e8edf2;letter-spacing:-0.5px}
.sub-title{font-size:0.95rem;color:#9fb3c8;text-align:center;margin-bottom:1.5rem;
letter-spacing:1.5px;text-transform:uppercase}
.question-box{background:#16273b;color:#e8edf2;
padding:1.4rem 1.8rem;border-radius:10px;font-size:1.15rem;font-weight:600;
margin-bottom:1rem;border-left:4px solid #3a6ea5}
.hint-box{background:#16273b;color:#9fb3c8;padding:0.8rem 1.1rem;border-radius:8px;
font-size:0.92rem;border-left:3px solid #5b8fc7;margin-bottom:0.8rem}
.score-box{background:#16273b;border-radius:12px;
padding:1.5rem 1rem;text-align:center;border:1px solid #233954}
.score-label{font-size:0.72rem;color:#9fb3c8;font-weight:700;text-transform:uppercase;letter-spacing:0.08em}
.score-number{font-size:2.6rem;font-weight:800;line-height:1.1;color:#5b8fc7}
.big-score{font-size:5.5rem;font-weight:800;text-align:center;line-height:1;color:#5b8fc7}
.tip-success{background:rgba(61,140,97,0.18);border-left:4px solid #3d8c61;
padding:0.7rem 1rem;border-radius:8px;margin:0.4rem 0;color:#7fc99a;font-size:0.92rem}
.tip-warning{background:rgba(196,140,40,0.18);border-left:4px solid #c48c28;
padding:0.7rem 1rem;border-radius:8px;margin:0.4rem 0;color:#e0ad55;font-size:0.92rem}
.tip-info{background:rgba(58,110,165,0.18);border-left:4px solid #3a6ea5;
padding:0.7rem 1rem;border-radius:8px;margin:0.4rem 0;color:#7eb0e0;font-size:0.92rem}
h1,h2,h3,h4{color:#e8edf2;font-weight:600}
div[data-testid='stVerticalBlock'] h3{color:#ffffff !important;font-weight:700 !important}
.stMarkdown h3{color:#ffffff !important;font-weight:700 !important}
[data-testid='stMarkdownContainer'] h3{color:#ffffff !important;font-weight:700 !important}
p,label,.stMarkdown{color:#c3d2e0}
[data-testid='stMetricValue']{color:#e8edf2;font-weight:700}
[data-testid='stMetricLabel']{color:#9fb3c8}
.stTabs [data-baseweb='tab']{color:#9fb3c8;font-weight:600;font-size:1rem}
.stTabs [aria-selected='true']{color:#5b8fc7;border-bottom-color:#5b8fc7}
.stButton>button{background:#3a6ea5;color:#ffffff;
border:none;border-radius:8px;font-weight:600;padding:0.5rem 1.5rem}
.stButton>button:hover{background:#4a7fb8}
.stProgress>div>div{background:#5b8fc7}
div[data-baseweb='select']>div{background:#16273b;border-color:#3a6ea5;color:#e8edf2;
min-width:220px}
div[data-baseweb='select'] *{color:#e8edf2 !important}
div[data-baseweb='popover']{min-width:220px}
[data-testid='stFileUploaderDropzone']{background:#16273b;border:1.5px dashed #3a6ea5;border-radius:8px}
[data-testid='stFileUploaderDropzoneInstructions'] span{color:#e8edf2;font-weight:600;font-size:1rem}
hr{border-color:#233954}
</style>
''', unsafe_allow_html=True)

defaults = {
    'eye_scores':        deque(maxlen=60),
    'posture_scores':    deque(maxlen=60),
    'emotions':          deque(maxlen=60),
    'current':           {'eye':0.0,'posture':0.0,'emotion':'Neutral','face':False,'pose':False},
    'session_active':    False,
    'speech_result':     None,
    'last_saved_id':     None,
    'current_question':  get_random_question(),
    'question_category': 'All Questions',
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def render_tip(category, level, tip):
    st.markdown(f'<div class="tip-{level}"><b>{category}:</b> {tip}</div>', unsafe_allow_html=True)

st.markdown('''
<div class="logo-row">
  <div class="logo-mark">IQ</div>
  <div class="main-title">InterviewIQ</div>
</div>
''', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Interview Performance Coach</div>',unsafe_allow_html=True)

tab1,tab2,tab3,tab4 = st.tabs(['Practice Session','Live Scores','Session Report','History'])

with tab1:
    st.subheader('Interview Question')
    qc1,qc2,qc3 = st.columns([3,1.3,1])
    with qc2:
        category = st.selectbox('Category',['All Questions','Data Science','Soft Skills'],key='qcat')
        st.session_state.question_category = category
    with qc3:
        st.markdown('<br>',unsafe_allow_html=True)
        if st.button('New Question',type='primary',use_container_width=True):
            st.session_state.current_question = get_random_question(category)
        if st.button('Skip',use_container_width=True):
            st.session_state.current_question = get_random_question(category)
    with qc1:
        st.markdown(f'<div class="question-box">{st.session_state.current_question}</div>',unsafe_allow_html=True)
    st.markdown('---')
    col_cam,col_tips = st.columns([2,1])
    with col_cam:
        st.subheader('Webcam Snapshot')
        st.markdown('<div class="hint-box">Click the camera button below, then Take Photo to capture a snapshot for scoring.</div>', unsafe_allow_html=True)
        camera_photo = st.camera_input('Take a photo while answering', key='camera_snap')
        if camera_photo is not None:
            bytes_data = camera_photo.getvalue()
            nparr = np.frombuffer(bytes_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            try:
                res = process_frame(img)
                st.session_state.current = {
                    'eye':     res['eye_contact'],
                    'posture': res['posture'],
                    'emotion': res['emotion'],
                    'face':    res['face_detected'],
                    'pose':    res['pose_detected'],
                }
                if res['face_detected']:
                    st.session_state.eye_scores.append(res['eye_contact'])
                    st.session_state.emotions.append(res['emotion'])
                if res['pose_detected']:
                    st.session_state.posture_scores.append(res['posture'])
                st.session_state.session_active = True
                st.success('Photo captured and scored.')

                env_tips = []
                brightness = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
                if brightness < 80:
                    env_tips.append('Lighting looks dim. Sit facing a light source for a clearer image.')
                elif brightness > 200:
                    env_tips.append('Lighting looks washed out. Try reducing strong backlight behind you.')
                if not res['face_detected']:
                    env_tips.append('No face detected. Make sure your face is centered and well-lit.')
                if not res['pose_detected']:
                    env_tips.append('Pose not detected. Sit a little further back so your shoulders are visible.')
                if res['posture'] < 0.6:
                    env_tips.append('Try sitting up a bit straighter with your shoulders level.')
                env_tips.append('Dress as you would for a real interview.')
                env_tips.append('A plain, tidy background helps the interviewer focus on you.')
                if env_tips:
                    st.markdown('**Presentation tips**')
                    for t in env_tips[:3]:
                        st.info(t)
            except Exception as e:
                st.error(f'Scoring error: {e}')
        st.markdown('---')
        st.subheader('Upload a Practice Video')
        st.markdown('<div class="hint-box">Upload a short video (under 2 minutes) of yourself answering a question. We will analyze posture, expression, and speech together.</div>', unsafe_allow_html=True)
        video_file = st.file_uploader('Upload your video (.mp4, .mov, .avi)', type=['mp4','mov','avi'], key='video_upload_tab1')
        if video_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(video_file.read())
                video_tmp_path = tmp.name
            with st.spinner('Analyzing video -- this may take a minute...'):
                video_result = analyze_video(video_tmp_path)
            if video_result.get('error'):
                st.error(video_result['error'])
            else:
                st.session_state.posture_scores.append(video_result['posture_avg'])
                st.session_state.eye_scores.append(video_result['eye_avg'])
                st.session_state.emotions.append(video_result['dominant_emotion'])
                if video_result.get('speech'):
                    st.session_state.speech_result = video_result['speech']
                st.success(f"Video analyzed -- {video_result['frames_analyzed']} frames checked, "
                           f"{video_result['faces_detected']} with a face detected.")
                vc1, vc2, vc3 = st.columns(3)
                vc1.metric('Posture', f"{video_result['posture_avg']*100:.0f}%")
                vc2.metric('Expression', video_result['dominant_emotion'])
                vc3.metric('Duration', f"{video_result['duration_sec']}s")
                if video_result.get('speech'):
                    sp_v = video_result['speech']
                    st.markdown(f"**Speech:** {sp_v['wpm']} WPM, {sp_v['total_fillers']} filler words")
                    if sp_v.get('transcript'):
                        st.text_area('Video Transcript', value=sp_v['transcript'], height=80, key='ta_video')
        st.markdown('---')
        st.subheader('Record Your Answer')
        st.markdown('<div class="hint-box">Click the microphone below to record your answer directly in the browser.</div>', unsafe_allow_html=True)
        recorded_audio = st.audio_input('Record your answer', key='audio_recorder_live')
        if recorded_audio is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(recorded_audio.getvalue())
                tmp_path = tmp.name
            with st.spinner('Analyzing your recording...'):
                speech = analyze_speech(tmp_path)
            if speech:
                st.session_state.speech_result = speech
                st.success(f"Done. WPM: {speech['wpm']} | Fillers: {speech['total_fillers']}")
                if speech['filler_counts']:
                    st.warning(f"Fillers detected: {speech['filler_counts']}")
        st.markdown('---')
        st.caption('Or, if recording in-browser does not work well on your device, upload a file instead:')
        st.markdown('**Upload your answer (.wav or .mp3)**')
        audio_file = st.file_uploader('Upload audio file', type=['wav','mp3'], key='audio_tab1', label_visibility='collapsed')
        if audio_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            with st.spinner('Analyzing speech...'):
                speech = analyze_speech(tmp_path)
            if speech:
                st.session_state.speech_result = speech
                st.success(f"Done. WPM: {speech['wpm']} | Fillers: {speech['total_fillers']}")
                if speech['filler_counts']:
                    st.warning(f"Fillers detected: {speech['filler_counts']}")
    with col_tips:
        st.subheader('Live Feedback')
        cur = st.session_state.current
        st.markdown('**Detection Status**')
        if cur['face']:
            st.markdown('<div class="tip-success">Face detected</div>',unsafe_allow_html=True)
        else:
            st.markdown('<div class="tip-warning">No face -- move closer</div>',unsafe_allow_html=True)
        if cur['pose']:
            st.markdown('<div class="tip-success">Pose detected</div>',unsafe_allow_html=True)
        else:
            st.markdown('<div class="tip-warning">Pose not detected</div>',unsafe_allow_html=True)
        st.markdown('---')
        st.markdown('**Current Scores**')
        st.markdown(f"Posture: **{cur['posture']*100:.0f}%**")
        st.progress(float(cur['posture']))
        st.markdown(f"Expression: **{cur['emotion']}**")
        st.markdown('---')
        st.markdown('**Browse Questions**')
        all_q = get_all_questions(st.session_state.get('question_category','All Questions'))
        st.caption(f'{len(all_q)} questions available')
        with st.expander('Show question list'):
            for i,q in enumerate(all_q[:15],1):
                st.markdown(f'{i}. {q}')
            if len(all_q)>15:
                st.caption(f'... and {len(all_q)-15} more')

with tab2:
    st.subheader('Real-Time Performance Dashboard')
    pos_avg  = float(np.mean(list(st.session_state.posture_scores))) if st.session_state.posture_scores else 0.0
    emo_list = list(st.session_state.emotions)
    dom_emo  = Counter(emo_list).most_common(1)[0][0] if emo_list else 'Neutral'
    sp = st.session_state.speech_result or {
        'wpm':0,'wpm_score':0.0,'filler_score':0.0,
        'total_fillers':0,'filler_counts':{},'transcript':'','word_count':0,'duration_sec':0
    }
    overall  = get_overall_score(0.0, pos_avg, sp['filler_score'], sp['wpm_score'])
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="score-box"><div class="score-label">Posture</div><div class="score-number">{pos_avg*100:.0f}%</div></div>',unsafe_allow_html=True)
        st.progress(pos_avg)
    with c2:
        st.markdown(f'<div class="score-box"><div class="score-label">Expression</div><div class="score-number" style="font-size:1.6rem">{dom_emo}</div></div>',unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="score-box"><div class="score-label">Overall</div><div class="score-number">{overall}</div></div>',unsafe_allow_html=True)
    st.markdown('---')
    if st.session_state.speech_result:
        sp2 = st.session_state.speech_result
        st.subheader('Speech Results')
        ca,cb,cc,cd = st.columns(4)
        ca.metric('WPM',sp2['wpm']); cb.metric('Fillers',sp2['total_fillers'])
        cc.metric('Words',sp2['word_count']); cd.metric('Duration',f"{sp2['duration_sec']}s")
        if sp2['transcript']:
            st.text_area('Transcript', value=sp2['transcript'], height=100, key='ta_tab2')
    else:
        st.info('Record or upload your answer in the Practice Session tab to see speech scores here.')

with tab3:
    st.subheader('Session Report')
    pos_avg  = float(np.mean(list(st.session_state.posture_scores))) if st.session_state.posture_scores else 0.0
    emo_list = list(st.session_state.emotions)
    dom_emo  = Counter(emo_list).most_common(1)[0][0] if emo_list else 'Neutral'
    sp = st.session_state.speech_result or {
        'wpm':0,'wpm_score':0.0,'filler_score':0.0,
        'total_fillers':0,'filler_counts':{},'transcript':'No audio recorded yet.',
        'word_count':0,'duration_sec':0
    }
    overall  = get_overall_score(0.0, pos_avg, sp['filler_score'], sp['wpm_score'])
    tips     = get_tips(0.0, pos_avg, dom_emo, sp)
    question = st.session_state.get('current_question','')
    st.markdown(f'<div class="big-score">{overall}<span style="font-size:2rem">/100</span></div>',unsafe_allow_html=True)
    label = 'Strong performance' if overall>=70 else 'Good effort, keep practicing' if overall>=50 else 'Keep practicing, you will improve'
    st.markdown(f"<h3 style='text-align:center;color:#9fb3c8'>{label}</h3>",unsafe_allow_html=True)
    if question:
        st.markdown(f'<div class="question-box">{question}</div>',unsafe_allow_html=True)
    st.markdown('---')
    cl,cr = st.columns(2)
    with cl:
        st.markdown('### Score Breakdown')
        for name,score,weight in [
            ('Posture',        pos_avg,            0.35),
            ('Speech Fillers', sp['filler_score'], 0.35),
            ('Speaking Pace',  sp['wpm_score'],    0.30),
        ]:
            st.markdown(f'**{name}** -- {score*100:.0f}% *(weight {int(weight*100)}%)*')
            st.progress(float(score))
    with cr:
        st.markdown('### Coaching Tips')
        for cat,lvl,tip in tips:
            render_tip(cat,lvl,tip)
    st.markdown('---')
    st.markdown('### Speech Details')
    cs1,cs2,cs3,cs4 = st.columns(4)
    cs1.metric('Duration', f"{sp['duration_sec']}s"); cs2.metric('Words', sp['word_count'])
    cs3.metric('WPM', sp['wpm']); cs4.metric('Fillers', sp['total_fillers'])
    if sp['transcript'] and sp['transcript'] != 'No audio recorded yet.':
        st.text_area('Transcript', value=sp['transcript'], height=100, key='ta_report')
    st.markdown('---')
    st.markdown('### Save Session')
    if st.button('Save to History', type='primary'):
        sid,path = save_session(overall, 0.0, pos_avg, sp, dom_emo, tips)
        st.session_state.last_saved_id = sid
        st.success('Session saved.')

with tab4:
    st.subheader('Practice History')
    stats = get_session_stats()
    if stats:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric('Total Sessions', stats['total_sessions']); c2.metric('Average Score', stats['avg_score'])
        c3.metric('Best Score', stats['best_score']); c4.metric('Improvement', f"{stats['improvement']:+.1f}",delta=stats['improvement'])
        st.markdown('---')
    sessions = load_all_sessions()
    if not sessions:
        st.info('No sessions yet. Complete a session and click Save in the Report tab.')
    else:
        st.markdown(f'**{len(sessions)} sessions saved**')
        for s in sessions:
            with st.expander(f"{s['timestamp']} -- Score: {s['overall_score']}/100"):
                c1,c2,c3 = st.columns(3)
                c1.metric('Overall', f"{s['overall_score']}/100")
                c2.metric('Posture', f"{s['posture']}%")
                c3.metric('WPM', s['wpm'])
                if s.get('transcript'):
                    st.text_area('Transcript', value=s['transcript'], height=60, key=f"hist_{s['session_id']}")
                if s.get('tips'):
                    st.markdown('**Coaching Tips:**')
                    for t in s['tips']:
                        render_tip(t['category'],t['level'],t['tip'])
                if st.button('Delete this session', key=f"del_{s['session_id']}"):
                    delete_session(s['session_id'])
                    st.rerun()
