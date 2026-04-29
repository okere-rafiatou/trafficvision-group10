import csv
import math
import cv2 as cv
from ultralytics import YOLO

TRAFFIC_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
GROUP_ID = 'group_10'

CSV_HEADER = [
    'frame', 'timestamp_sec', 'scene_name', 'group_id', 'video_name',
    'track_id', 'class_name', 'confidence',
    'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
    'cx', 'cy', 'frame_width', 'frame_height',
    'crossed_line', 'direction', 'speed_px_s',
]


class TrafficTracker:
    def __init__(self, model_path='models/yolo11n.pt', selected_classes=None):
        self.model = YOLO(model_path)
        self.selected_classes = selected_classes or TRAFFIC_CLASSES
        self._class_ids = None
        self._prev = {}  # track_id -> (cx, cy, frame_idx)

    @property
    def class_ids(self):
        if self._class_ids is None:
            self._class_ids = [k for k, v in self.model.names.items()
                               if v in self.selected_classes]
        return self._class_ids

    def reset_tracks(self):
        self._prev = {}

    def process_frame(self, frame, frame_idx, fps, csv_writer,
                      track_ids_seen, counts,
                      video_name='', scene_name='',
                      frame_width=0, frame_height=0, line_y=None):

        if line_y is None:
            line_y = frame_height // 2 if frame_height else 0

        timestamp = frame_idx / fps if fps > 0 else frame_idx
        results = self.model.track(frame, persist=True,
                                   classes=self.class_ids, verbose=False)

        detected = False

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                track_id  = int(boxes.id[i])
                cls_id    = int(boxes.cls[i])
                cls_name  = self.model.names[cls_id]
                conf      = float(boxes.conf[i])
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                direction  = ''
                speed      = 0.0
                crossed    = False

                if track_id in self._prev:
                    px, py, pf = self._prev[track_id]
                    dt = (frame_idx - pf) / fps if fps > 0 and frame_idx != pf else 0
                    if dt > 0:
                        dist  = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                        speed = dist / dt
                        direction = 'up' if cy < py else ('down' if cy > py else '')
                    # crossed_line: centre crosses the reference line between frames
                    if line_y and ((py < line_y <= cy) or (py > line_y >= cy)):
                        crossed = True

                self._prev[track_id] = (cx, cy, frame_idx)

                csv_writer.writerow([
                    frame_idx,
                    f'{timestamp:.3f}',
                    scene_name,
                    GROUP_ID,
                    video_name,
                    track_id,
                    cls_name,
                    f'{conf:.3f}',
                    x1, y1, x2, y2,
                    cx, cy,
                    frame_width, frame_height,
                    'true' if crossed else 'false',
                    direction,
                    f'{speed:.2f}',
                ])

                if track_id not in track_ids_seen:
                    track_ids_seen.add(track_id)
                    if cls_name in counts:
                        counts[cls_name] += 1

                detected = True

        annotated = results[0].plot()

        # Draw the counting line
        if line_y and frame_width:
            cv.line(annotated, (0, line_y), (frame_width, line_y),
                    (0, 200, 255), 2)
            cv.putText(annotated, 'counting line', (8, line_y - 6),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        if not detected:
            cv.putText(annotated, 'No objects detected', (10, 35),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        y_off = 65 if not detected else 30
        for cls_name, count in counts.items():
            if count > 0:
                cv.putText(annotated, f'{cls_name}: {count}', (10, y_off),
                           cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                y_off += 28

        return annotated, counts, detected

