import cv2
import mediapipe as mp
import time

from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks import python as mp_tasks

CAP_WIDTH  = 640
CAP_HEIGHT = 480

GESTURE_COOLDOWN = 0.35
MOVE_COOLDOWN    = 0.08
MODEL_PATH       = "hand_landmarker.task"

_running = False


def dist_sq(a, b):
    return (a.x - b.x)**2 + (a.y - b.y)**2


def detect_gesture(lm):
    if dist_sq(lm[4], lm[8]) < 0.004:
        return "PINCH"

    index_up  = lm[8].y  < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up   = lm[16].y < lm[14].y
    pinky_up  = lm[20].y < lm[18].y

    if index_up and not middle_up and not ring_up and not pinky_up:
        return "POINT_LEFT" if lm[8].x < lm[5].x else "POINT_RIGHT"

    if index_up and middle_up and not ring_up and not pinky_up:
        return "V_SIGN"

    if not (index_up or middle_up or ring_up or pinky_up):
        return "FIST"

    if index_up and middle_up and ring_up and pinky_up:
        return "OPEN_HAND"

    return "NONE"


def stop_stream():
    global _running
    _running = False


def gesture_stream():
    global _running
    _running = True

    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options      = HandLandmarkerOptions(
        base_options                  = base_options,
        running_mode                  = RunningMode.VIDEO,
        num_hands                     = 2,
        min_hand_detection_confidence = 0.7,
        min_hand_presence_confidence  = 0.7,
        min_tracking_confidence       = 0.7
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

    last_trigger_time = 0
    last_move_time    = 0
    prev_palm_x       = None
    prev_palm_y       = None
    prev_pinch_dist   = None

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            while _running:
                success, frame = cap.read()
                if not success:
                    break

                frame        = cv2.flip(frame, 1)
                imageRGB     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                current_time = time.time()
                timestamp_ms = int(current_time * 1000)

                mp_image = mp.Image(
                    image_format = mp.ImageFormat.SRGB,
                    data         = imageRGB
                )

                result    = landmarker.detect_for_video(mp_image, timestamp_ms)
                all_hands = result.hand_landmarks or []

                can_trigger = current_time - last_trigger_time > GESTURE_COOLDOWN
                can_move    = current_time - last_move_time    > MOVE_COOLDOWN

                if all_hands:
                    gestures = [detect_gesture(lm) for lm in all_hands]

                    for i, lm in enumerate(all_hands):
                        gesture = gestures[i]
                        palm_x  = lm[9].x
                        palm_y  = lm[9].y

                        if gesture == "FIST":
                            if prev_palm_x is not None and can_move:
                                dx = palm_x - prev_palm_x
                                dy = palm_y - prev_palm_y

                                if abs(dx) > 0.025 or abs(dy) > 0.025:
                                    direction      = ("r" if dx > 0 else "l") if abs(dx) > abs(dy) else ("d" if dy > 0 else "u")
                                    last_move_time = current_time

                                    yield {
                                        "gesture"    : "FIST",
                                        "hand_count" : len(all_hands),
                                        "confidence" : 0.0,
                                        "palm"       : {"x": palm_x, "y": palm_y},
                                        "timestamp"  : current_time,
                                        "meta"       : {"direction": direction}
                                    }

                            prev_palm_x = palm_x
                            prev_palm_y = palm_y

                        elif gesture == "POINT_LEFT" and can_trigger:
                            last_trigger_time = current_time
                            yield {
                                "gesture"    : "POINT_LEFT",
                                "hand_count" : len(all_hands),
                                "confidence" : 0.0,
                                "palm"       : {"x": palm_x, "y": palm_y},
                                "timestamp"  : current_time,
                                "meta"       : {}
                            }

                        elif gesture == "POINT_RIGHT" and can_trigger:
                            last_trigger_time = current_time
                            yield {
                                "gesture"    : "POINT_RIGHT",
                                "hand_count" : len(all_hands),
                                "confidence" : 0.0,
                                "palm"       : {"x": palm_x, "y": palm_y},
                                "timestamp"  : current_time,
                                "meta"       : {}
                            }

                        elif gesture == "PINCH" and can_trigger:
                            pinch_dist = dist_sq(lm[4], lm[8])

                            if prev_pinch_dist is not None:
                                delta = pinch_dist - prev_pinch_dist
                                if abs(delta) > 0.0015:
                                    resize_amount     = int(delta * 800)
                                    last_trigger_time = current_time

                                    yield {
                                        "gesture"    : "PINCH",
                                        "hand_count" : len(all_hands),
                                        "confidence" : 0.0,
                                        "palm"       : {"x": palm_x, "y": palm_y},
                                        "timestamp"  : current_time,
                                        "meta"       : {"resize_amount": resize_amount}
                                    }

                            prev_pinch_dist = pinch_dist

                        elif gesture == "V_SIGN" and can_trigger:
                            last_trigger_time = current_time
                            yield {
                                "gesture"    : "V_SIGN",
                                "hand_count" : len(all_hands),
                                "confidence" : 0.0,
                                "palm"       : {"x": palm_x, "y": palm_y},
                                "timestamp"  : current_time,
                                "meta"       : {}
                            }

                        elif gesture == "OPEN_HAND" and can_trigger:
                            last_trigger_time = current_time
                            yield {
                                "gesture"    : "OPEN_HAND",
                                "hand_count" : len(all_hands),
                                "confidence" : 0.0,
                                "palm"       : {"x": palm_x, "y": palm_y},
                                "timestamp"  : current_time,
                                "meta"       : {}
                            }

                        if gesture not in ("FIST", "PINCH"):
                            prev_palm_x     = None
                            prev_palm_y     = None
                            prev_pinch_dist = None

                else:
                    prev_palm_x     = None
                    prev_palm_y     = None
                    prev_pinch_dist = None

    finally:
        cap.release()
