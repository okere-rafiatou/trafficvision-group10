import csv
import io
import json
import os
import time
import threading
import zipfile
from pathlib import Path

import cv2 as cv
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from utils.tracker import TRAFFIC_CLASSES, TrafficTracker

app = Flask(__name__)

UPLOAD_FOLDER = Path('uploads')
LOG_FOLDER    = Path('logs')
OUTPUT_FOLDER = Path('outputs')
for d in (UPLOAD_FOLDER, LOG_FOLDER, OUTPUT_FOLDER):
    d.mkdir(exist_ok=True)

# ── Shared state 
_lock = threading.Lock()
_state = {
    'frame':         None,
    'counts':        {},
    'detected':      True,
    'processing':    False,
    'total_frames':  0,
    'current_frame': 0,
    'error':         None,
    'status_msg':    '',
}


# ── Processing thread (file / youtube) ──────────────────────────────────────
def _processing_thread(video_path, selected_classes, log_path, output_path,
                       scene_name='', video_name=''):
    from utils.tracker import CSV_HEADER
    cap = None
    writer = None
    try:
        tracker = TrafficTracker('models/yolo11n.pt', selected_classes)
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open video: {video_path}')

        fps   = cap.get(cv.CAP_PROP_FPS) or 30
        w     = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        if not scene_name:
            scene_name = Path(video_path).stem
        if not video_name:
            video_name = Path(video_path).name

        writer = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        track_ids_seen = set()
        counts = {c: 0 for c in selected_classes}
        frame_idx = 0
        tracker.reset_tracks()

        with _lock:
            _state['total_frames'] = total
            _state['counts']       = counts.copy()
            _state['error']        = None

        with open(log_path, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(CSV_HEADER)
            while cap.isOpened():
                with _lock:
                    if not _state['processing']:
                        break
                success, frame = cap.read()
                if not success:
                    break
                annotated, counts, detected = tracker.process_frame(
                    frame, frame_idx, fps, csv_writer, track_ids_seen, counts,
                    video_name=video_name, scene_name=scene_name,
                    frame_width=w, frame_height=h)
                writer.write(annotated)

                _, buf = cv.imencode('.jpg', annotated, [cv.IMWRITE_JPEG_QUALITY, 80])
                with _lock:
                    _state['frame']         = buf.tobytes()
                    _state['counts']        = counts.copy()
                    _state['detected']      = detected
                    _state['current_frame'] = frame_idx

                frame_idx += 1

    except Exception as e:
        with _lock:
            _state['error'] = str(e)
        print(f'[ERROR] Processing thread: {e}')
    finally:
        if cap:    cap.release()
        if writer: writer.release()
        with _lock:
            _state['processing'] = False


# ── Webcam thread ────────────────────────────────────────────────────────────
def _webcam_thread(selected_classes, log_path, scene_name='webcam_live'):
    from utils.tracker import CSV_HEADER
    cap = None
    try:
        tracker = TrafficTracker('models/yolo11n.pt', selected_classes)
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError('Cannot open webcam (index 0). Check connection.')

        fps = cap.get(cv.CAP_PROP_FPS) or 30
        w   = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        track_ids_seen = set()
        counts    = {c: 0 for c in selected_classes}
        frame_idx = 0
        tracker.reset_tracks()

        with _lock:
            _state['total_frames']  = 0
            _state['current_frame'] = 0
            _state['counts']        = counts.copy()
            _state['error']         = None

        with open(log_path, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(CSV_HEADER)
            while True:
                with _lock:
                    if not _state['processing']:
                        break
                success, frame = cap.read()
                if not success:
                    break
                annotated, counts, detected = tracker.process_frame(
                    frame, frame_idx, fps, csv_writer, track_ids_seen, counts,
                    video_name='webcam', scene_name=scene_name,
                    frame_width=w, frame_height=h)

                _, buf = cv.imencode('.jpg', annotated, [cv.IMWRITE_JPEG_QUALITY, 80])
                with _lock:
                    _state['frame']         = buf.tobytes()
                    _state['counts']        = counts.copy()
                    _state['detected']      = detected
                    _state['current_frame'] = frame_idx

                frame_idx += 1

    except Exception as e:
        with _lock:
            _state['error'] = str(e)
        print(f'[ERROR] Webcam thread: {e}')
    finally:
        if cap: cap.release()
        with _lock:
            _state['processing'] = False


# ── YouTube stream helper ────────────────────────────────────────────────────
def _youtube_thread(url, selected_classes, log_path, output_path):
    try:
        import yt_dlp
        ydl_opts = {
            'format':  'best[ext=mp4][height<=480]/best[height<=480]/best',
            'quiet':   True,
            'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
        }
        with _lock:
            _state['status_msg'] = 'Extracting YouTube stream URL…'
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            stream_url = info['url']
            video_name = info.get('title', 'youtube_video')[:60]
            scene_name = video_name
        with _lock:
            _state['status_msg'] = 'Stream ready — starting detection…'
    except ImportError:
        with _lock:
            _state['error'] = 'yt-dlp not installed. Run: pip install yt-dlp'
            _state['processing'] = False
        return
    except Exception as e:
        with _lock:
            _state['error'] = f'YouTube error: {e}'
            _state['processing'] = False
        return

    _processing_thread(stream_url, selected_classes, log_path, output_path,
                       scene_name=scene_name, video_name=video_name)


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/app')
def app_page():
    return render_template('index.html', classes=TRAFFIC_CLASSES)


@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory('data', filename)


@app.route('/process', methods=['POST'])
def process():
    if 'video' not in request.files or request.files['video'].filename == '':
        return jsonify({'error': 'No video file provided'}), 400

    with _lock:
        if _state['processing']:
            return jsonify({'error': 'Already processing. Click Stop or Reset first.'}), 409

    video = request.files['video']
    selected_classes = request.form.getlist('classes') or TRAFFIC_CLASSES
    video_path  = str(UPLOAD_FOLDER / video.filename)
    video.save(video_path)

    stem        = Path(video.filename).stem
    ts          = int(time.time())
    log_path    = str(LOG_FOLDER    / f'{stem}_{ts}.csv')
    output_path = str(OUTPUT_FOLDER / f'{stem}_{ts}_out.mp4')

    with _lock:
        _state['processing']    = True
        _state['frame']         = None
        _state['counts']        = {c: 0 for c in selected_classes}
        _state['current_frame'] = 0
        _state['error']         = None

    scene_name = request.form.get('scene_name', '').strip() or Path(video_path).stem
    threading.Thread(target=_processing_thread,
                     args=(video_path, selected_classes, log_path, output_path),
                     kwargs={'scene_name': scene_name, 'video_name': Path(video_path).name},
                     daemon=True).start()
    return jsonify({'status': 'started'})


@app.route('/process_youtube', methods=['POST'])
def process_youtube():
    url = request.form.get('url', '').strip()
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    with _lock:
        if _state['processing']:
            return jsonify({'error': 'Already processing. Click Stop or Reset first.'}), 409

    selected_classes = request.form.getlist('classes') or TRAFFIC_CLASSES
    ts          = int(time.time())
    log_path    = str(LOG_FOLDER    / f'yt_{ts}.csv')
    output_path = str(OUTPUT_FOLDER / f'yt_{ts}_out.mp4')

    with _lock:
        _state['processing']    = True
        _state['frame']         = None
        _state['counts']        = {c: 0 for c in selected_classes}
        _state['current_frame'] = 0
        _state['total_frames']  = 0
        _state['error']         = None

    threading.Thread(target=_youtube_thread,
                     args=(url, selected_classes, log_path, output_path),
                     daemon=True).start()
    return jsonify({'status': 'started'})


@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    with _lock:
        if _state['processing']:
            return jsonify({'error': 'Already processing. Click Stop or Reset first.'}), 409

    selected_classes = request.form.getlist('classes') or TRAFFIC_CLASSES
    ts       = int(time.time())
    log_path = str(LOG_FOLDER / f'webcam_{ts}.csv')

    with _lock:
        _state['processing']    = True
        _state['frame']         = None
        _state['counts']        = {c: 0 for c in selected_classes}
        _state['current_frame'] = 0
        _state['total_frames']  = 0
        _state['error']         = None

    threading.Thread(target=_webcam_thread,
                     args=(selected_classes, log_path),
                     daemon=True).start()
    return jsonify({'status': 'started'})


@app.route('/stop', methods=['POST'])
def stop():
    with _lock:
        _state['processing'] = False
    return jsonify({'status': 'stopped'})


@app.route('/reset', methods=['POST'])
def reset():
    with _lock:
        _state['processing']    = False
        _state['frame']         = None
        _state['counts']        = {}
        _state['current_frame'] = 0
        _state['total_frames']  = 0
        _state['error']         = None
        _state['status_msg']    = ''
    return jsonify({'status': 'reset'})


@app.route('/stream')
def stream():
    def generate():
        while True:
            with _lock:
                frame = _state['frame']
            if frame:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/frame')
def frame():
    with _lock:
        f = _state['frame']
    if f is None:
        return '', 204
    return Response(f, mimetype='image/jpeg')


@app.route('/status')
def status():
    with _lock:
        return jsonify({
            'counts':        _state['counts'],
            'detected':      _state['detected'],
            'processing':    _state['processing'],
            'current_frame': _state['current_frame'],
            'total_frames':  _state['total_frames'],
            'error':         _state['error'],
            'status_msg':    _state['status_msg'],
        })


@app.route('/dashboard')
def dashboard():
    scenes = []
    class_totals = {}
    for log_file in sorted(LOG_FOLDER.glob('*.csv')):
        try:
            scene = _parse_log(log_file)
            scenes.append(scene)
            for cls, cnt in scene['counts'].items():
                class_totals[cls] = class_totals.get(cls, 0) + cnt
        except Exception:
            pass
    total_objects = sum(class_totals.values())
    summary = {
        'total_scenes':  len(scenes),
        'total_objects': total_objects,
        'top_class':     max(class_totals, key=class_totals.get) if class_totals else '—',
        'class_totals':  class_totals,
    }
    return render_template('dashboard.html',
                           scenes=json.dumps(scenes),
                           summary=json.dumps(summary))


@app.route('/download_dataset')
def download_dataset():
    log_files = list(LOG_FOLDER.glob('*.csv'))
    if not log_files:
        return jsonify({'error': 'No logs available yet'}), 404
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for log_file in log_files:
            zf.write(log_file, log_file.name)
    buf.seek(0)
    return Response(
        buf.getvalue(),
        mimetype='application/zip',
        headers={'Content-Disposition': 'attachment; filename=traffic_dataset.zip'}
    )


def _parse_log(log_path):
    counts   = {}
    timeline = {}
    seen_ids = set()
    with open(log_path, newline='') as f:
        for row in csv.DictReader(f):
            uid    = f"{row['track_id']}_{row['class_name']}"
            cls    = row['class_name']
            bucket = str(int(float(row['timestamp_sec']) // 10) * 10)
            if uid not in seen_ids:
                seen_ids.add(uid)
                counts[cls] = counts.get(cls, 0) + 1
            timeline.setdefault(bucket, {})
            timeline[bucket][cls] = timeline[bucket].get(cls, 0) + 1
    return {'name': Path(log_path).stem, 'counts': counts, 'timeline': timeline}


if __name__ == '__main__':
    app.run(debug=False, threaded=True, host='0.0.0.0', port=7860)
