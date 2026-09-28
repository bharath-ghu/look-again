import cv2
import numpy as np
import csv
import os
import time
from datetime import datetime
from collections import defaultdict

from qai_hub_models.models.yolov7.model import YoloV7
from qai_hub_models.models.yolov7.app import YoloV7DetectionApp

print("Loading YOLOv7 model... (this may take a moment)")
model = YoloV7.from_pretrained()
app = YoloV7DetectionApp(model)
print("Model loaded.")

STATIC_FRAMES_THRESHOLD = 15
POSITION_TOLERANCE = 30
UNCERTAIN_LOW = 0.5
UNCERTAIN_HIGH = 0.75

tracked_objects = {}
next_id = 0

LOG_PATH = "look_again_log.csv"
log_file_exists = os.path.isfile(LOG_PATH)
log_file = open(LOG_PATH, "a", newline="")
log_writer = csv.writer(log_file)
if not log_file_exists:
    log_writer.writerow(["timestamp", "before_conf", "after_conf", "outcome"])

TIMING_LOG_PATH = "timing_log.csv"
timing_log_exists = os.path.isfile(TIMING_LOG_PATH)
timing_log_file = open(TIMING_LOG_PATH, "a", newline="")
timing_writer = csv.writer(timing_log_file)
if not timing_log_exists:
    timing_writer.writerow(["timestamp", "stage", "duration_ms"])


def log_timing(stage, duration_seconds):
    ts = datetime.now().isoformat(timespec="seconds")
    timing_writer.writerow([ts, stage, round(duration_seconds * 1000, 2)])
    timing_log_file.flush()


def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def match_or_create(center):
    global next_id
    for obj_id, obj in tracked_objects.items():
        ox, oy = obj["center"]
        if abs(ox - center[0]) < POSITION_TOLERANCE and abs(oy - center[1]) < POSITION_TOLERANCE:
            return obj_id
    obj_id = next_id
    next_id += 1
    tracked_objects[obj_id] = {"center": center, "count": 0, "flagged": False}
    return obj_id


def look_again(rgb_frame_640, box):
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(640, int(x2))
    y2 = min(640, int(y2))

    box_w = x2 - x1
    box_h = y2 - y1

    min_size = 120
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    half_w = max(box_w / 2 + 25, min_size / 2)
    half_h = max(box_h / 2 + 25, min_size / 2)

    x1p = max(0, int(cx - half_w))
    y1p = max(0, int(cy - half_h))
    x2p = min(640, int(cx + half_w))
    y2p = min(640, int(cy + half_h))

    crop = rgb_frame_640[y1p:y2p, x1p:x2p]
    if crop.shape[0] < 5 or crop.shape[1] < 5:
        return None

    crop_resized = cv2.resize(crop, (640, 640))

    crop_boxes, crop_scores, crop_class_idx = app.predict_boxes_from_image(crop_resized, raw_output=True)

    boxes2 = crop_boxes[0]
    scores2 = crop_scores[0]

    if len(scores2) == 0:
        return None

    best_idx = int(np.argmax(scores2.numpy())) if hasattr(scores2, "numpy") else int(np.argmax(scores2))
    best_conf = scores2[best_idx].item()
    return best_conf


def resolve_confidence(before_conf, after_conf):
    timestamp = datetime.now().isoformat(timespec="seconds")

    if after_conf is None:
        log_writer.writerow([timestamp, round(before_conf, 3), "", "no_detection_fallback"])
        log_file.flush()
        return before_conf, "no_detection_fallback"

    if after_conf > before_conf:
        log_writer.writerow([timestamp, round(before_conf, 3), round(after_conf, 3), "improved"])
        log_file.flush()
        return after_conf, "improved"

    log_writer.writerow([timestamp, round(before_conf, 3), round(after_conf, 3), "kept_original"])
    log_file.flush()
    return before_conf, "kept_original"


OUTCOME_LABELS = {
    "improved": "RE-SENSED: IMPROVED",
    "kept_original": "RE-SENSED: CONFIRMED",
    "no_detection_fallback": "RE-SENSED: FALLBACK",
}

OUTCOME_COLORS = {
    "improved": (0, 255, 255),
    "kept_original": (255, 200, 0),
    "no_detection_fallback": (128, 128, 128),
}


def motion_monitor(camera_index=0, diff_threshold=25, min_motion_area=500):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read initial frame")
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        frame_delta = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(frame_delta, diff_threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = any(cv2.contourArea(c) > min_motion_area for c in contours)

        yield frame, motion_detected

        prev_gray = gray

    cap.release()


if __name__ == "__main__":
    for frame, motion_detected in motion_monitor():
        display_frame = frame.copy()

        if motion_detected:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame_640 = cv2.resize(rgb_frame, (640, 640))

            t0 = time.perf_counter()
            pred_boxes, pred_scores, pred_class_idx = app.predict_boxes_from_image(rgb_frame_640, raw_output=True)
            log_timing("base_detection", time.perf_counter() - t0)

            boxes = pred_boxes[0]
            scores = pred_scores[0]

            seen_ids = set()
            for i, box in enumerate(boxes):
                conf = scores[i].item()

                final_conf = conf
                outcome = None
                if UNCERTAIN_LOW <= conf < UNCERTAIN_HIGH:
                    box_coords = box.tolist()
                    t1 = time.perf_counter()
                    new_conf = look_again(rgb_frame_640, box_coords)
                    log_timing("look_again", time.perf_counter() - t1)
                    final_conf, outcome = resolve_confidence(conf, new_conf)

                x1, y1, x2, y2 = box.tolist()
                center = get_center((x1, y1, x2, y2))
                obj_id = match_or_create(center)
                seen_ids.add(obj_id)

                tracked_objects[obj_id]["center"] = center
                tracked_objects[obj_id]["count"] += 1

                frame_h = frame.shape[0]
                frame_w = frame.shape[1]
                sx1 = int(x1 * frame_w / 640)
                sy1 = int(y1 * frame_h / 640)
                sx2 = int(x2 * frame_w / 640)
                sy2 = int(y2 * frame_h / 640)

                class_idx = int(pred_class_idx[0][i].item())
                is_person = (class_idx == 0)
                ITEM_CLASSES = {24, 25, 26, 27, 28, 39, 63, 64, 67, 73}
                is_valid_item = (class_idx in ITEM_CLASSES)

                if is_person or not is_valid_item:
                    tracked_objects[obj_id]["count"] = 0
                    tracked_objects[obj_id]["flagged"] = False
                elif tracked_objects[obj_id]["count"] >= STATIC_FRAMES_THRESHOLD:
                    tracked_objects[obj_id]["flagged"] = True

                is_flagged = tracked_objects[obj_id]["flagged"]
                count = tracked_objects[obj_id]["count"]

                if is_flagged:
                    color = (0, 0, 255)
                    label = "LEFT BEHIND"
                elif outcome is not None:
                    color = OUTCOME_COLORS[outcome]
                    label = OUTCOME_LABELS[outcome] + " conf:" + str(round(final_conf, 2))
                else:
                    color = (0, 255, 0)
                    label = "tracking " + str(count) + " conf:" + str(round(final_conf, 2))

                cv2.rectangle(display_frame, (sx1, sy1), (sx2, sy2), color, 2)
                cv2.putText(display_frame, label, (sx1, sy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            for obj_id in list(tracked_objects.keys()):
                if obj_id not in seen_ids:
                    del tracked_objects[obj_id]

        cv2.imshow("Event Classification Pipeline", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    log_file.close()
    timing_log_file.close()