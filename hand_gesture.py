import cv2
import mediapipe as mp
import time

from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks import python as mp_tasks

CAP_WIDTH  = 640
CAP_HEIGHT = 480

GESTURE_COOLDOWN = 1.5
MOVE_COOLDOWN    = 0.016
MODEL_PATH       = "hand_landmarker.task"
PHOTO_DIR        = "/home/prasiddha/Pictures/GesturesImages"

_running = False


def dist_sq(a, b):
    return (a.x - b.x)**2 + (a.y - b.y)**2


def detect_gesture(lm) -> str:
    index_up  = lm[8].y  < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up   = lm[16].y < lm[14].y
    pinky_up  = lm[20].y < lm[18].y

    if index_up and not any([middle_up, ring_up, pinky_up]):
        return "POINT_LEFT" if lm[8].x < lm[5].x else "POINT_RIGHT"

    if index_up and middle_up and not any([ring_up, pinky_up]):
        return "V_SIGN"

    if all([index_up, middle_up, ring_up, pinky_up]):
        return "OPEN_HAND"

    if not any([index_up, middle_up, ring_up, pinky_up]):
        return "FIST"

    return "NONE"


def stop_stream() -> None:
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
        min_hand_detection_confidence = 0.75,
        min_hand_presence_confidence  = 0.75,
        min_tracking_confidence       = 0.75
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

    last_trigger_time = 0.0
    last_move_time    = 0.0
    prev_palm         = None
    prev_gesture      = "NONE"

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
                    for lm in all_hands:
                        gesture = detect_gesture(lm)
                        palm    = (lm[9].x, lm[9].y)

                        if gesture == "FIST":
                            if prev_palm is not None and can_move:
                                dx = palm[0] - prev_palm[0]
                                dy = palm[1] - prev_palm[1]

                                if abs(dx) > 0.025 or abs(dy) > 0.025:
                                    direction      = ("r" if dx > 0 else "l") if abs(dx) > abs(dy) else ("d" if dy > 0 else "u")
                                    last_move_time = current_time

                                    yield {
                                        "gesture"    : "FIST",
                                        "hand_count" : len(all_hands),
                                        "confidence" : 0.0,
                                        "palm"       : {"x": palm[0], "y": palm[1]},
                                        "timestamp"  : current_time,
                                        "meta"       : {"direction": direction}
                                    }

                            prev_palm = palm

                        elif gesture == "POINT_LEFT" and can_trigger:
                            last_trigger_time = current_time
                            yield {
                                "gesture"    : "POINT_LEFT",
                                "hand_count" : len(all_hands),
                                "confidence" : 0.0,
                                "palm"       : {"x": palm[0], "y": palm[1]},
                                "timestamp"  : current_time,
                                "meta"       : {}
                            }

                        elif gesture == "POINT_RIGHT" and can_trigger:
                            last_trigger_time = current_time
                            yield {
                                "gesture"    : "POINT_RIGHT",
                                "hand_count" : len(all_hands),
                                "confidence" : 0.0,
                                "palm"       : {"x": palm[0], "y": palm[1]},
                                "timestamp"  : current_time,
                                "meta"       : {}
                            }

                        elif gesture == "V_SIGN" and can_trigger and prev_gesture != "V_SIGN":
                            photo_path        = f"{PHOTO_DIR}/gesture_photo_{int(current_time)}.jpg"
                            last_trigger_time = current_time
                            yield {
                                "gesture"    : "V_SIGN",
                                "hand_count" : len(all_hands),
                                "confidence" : 0.0,
                                "palm"       : {"x": palm[0], "y": palm[1]},
                                "timestamp"  : current_time,
                                "meta"       : {"photo_path": photo_path}
                            }

                        elif gesture == "OPEN_HAND" and can_trigger and prev_gesture != "OPEN_HAND":
                            last_trigger_time = current_time
                            yield {
                                "gesture"    : "OPEN_HAND",
                                "hand_count" : len(all_hands),
                                "confidence" : 0.0,
                                "palm"       : {"x": palm[0], "y": palm[1]},
                                "timestamp"  : current_time,
                                "meta"       : {"app": "app.zen_browser.zen"}
                            }

                        if gesture != "FIST":
                            prev_palm = None

                        prev_gesture = gesture

                else:
                    prev_palm    = None
                    prev_gesture = "NONE"

    finally:
        cap.release()
