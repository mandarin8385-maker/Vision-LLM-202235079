# 물 마시기 프로젝트 1: 카메라 → YOLO → 컵 횟수 판정 → JSON 저장
# 수업 04의 실시간 탐지, 06-A-7의 JSON 전달 코드를 바탕으로 작성했습니다.
# 카메라 종류는 아래 pipeline 내용만 교체하여 사용합니다.
# 배치 예: 프로젝트/src 와 프로젝트/Project02/두 .py 파일
# Project02 폴더에서 실행하며, 모델·JSON 경로는 파일 위치에서 한 단계 위를 기준으로 합니다.
# 실행: python water_camera.py

import cv2
import json
import os
import time
from pathlib import Path
from statistics import median
from ultralytics import YOLO


# -------------------- 설정 --------------------
# .py가 있는 폴더에서 한 단계 위로 이동합니다. 그곳에 src 폴더가 있습니다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLO_MODEL_PATH = str(PROJECT_ROOT / "src/models/YOLO/yolo11n_int8.engine")
JSON_PATH = str(PROJECT_ROOT / "src/json/water_vision_data.json")
TEMP_JSON_PATH = str(PROJECT_ROOT / "src/json/water_vision_data_tmp.json")

CUP_CLASS_ID = 41          # 수업 COCO 모델의 cup 클래스
CONFIDENCE = 0.25
MIN_AREA_RATIO = 0.80      # 기준 면적의 80%
MAX_AREA_RATIO = 1.20      # 기준 면적의 120%
CALIBRATION_SECONDS = 2.0  # 시작 시 컵을 고정할 시간
MIN_SAMPLES = 5
CALIBRATION_TOLERANCE = 0.15
ABSENCE_SECONDS = 0.8      # 연속 미검출 확인 시간
RETURN_SECONDS = 0.8       # 정상 위치·크기로 연속 복귀 확인 시간
CENTER_TOLERANCE = 0.75    # 기준 컵 너비·높이 대비 중심 이동 허용. None이면 위치 조건 해제
MAX_FRAME_GAP = 2.0       # 관측이 오래 끊기면 연속 확인 시간을 다시 측정
JSON_INTERVAL = 0.2      # 기본 저장 간격. 상태가 달라지면 이 시간 전에도 저장


# -------------------- 함수 정의 --------------------
def cup_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def same_position(box, reference):
    if CENTER_TOLERANCE is None:
        return True
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = reference
    return (
        abs((x1 + x2) / 2 - (rx1 + rx2) / 2) <= (rx2 - rx1) * CENTER_TOLERANCE
        and abs((y1 + y2) / 2 - (ry1 + ry2) / 2) <= (ry2 - ry1) * CENTER_TOLERANCE
    )


# -------------------- 여기부터 바로 실행 --------------------
# 목표 횟수는 키보드로 입력합니다.
goal = int(input("물을 마실 목표 횟수를 입력하세요. 예: 3 > "))

# JSON을 저장할 폴더를 만듭니다. 이미 있으면 그대로 사용합니다.
Path(JSON_PATH).parent.mkdir(parents=True, exist_ok=True)
yolo = YOLO(YOLO_MODEL_PATH)

# [수업 06-A-7] Jetson CSI 카메라 파이프라인
pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)
# 카메라를 바꾸려면 위 pipeline의 내용만 교체합니다.
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

# 현재 횟수와 컵 상태
count = 0
state = "calibrating"

# 처음 컵의 기준을 정하는 데 사용하는 값
baseline_area = None         # 기준 컵 면적
baseline_box = None          # 기준 컵 위치
samples = []                 # 기준을 정하기 위해 모으는 컵 검출 좌표

# 기준 설정 / 사라짐 / 복귀가 시작된 시각
calibration_started = None
absence_started = None
return_started = None
last_frame_time = None       # 관측이 오래 끊겼는지 확인하는 이전 프레임 시각

# assistant에 전달할 정보
voice_request_id = 0          # v 키를 누른 횟수
session_id = str(time.time_ns())  # 이번 실행을 구분하는 번호
vision_data = None           # JSON으로 보낼 데이터

# JSON을 마지막으로 저장한 시각과 그때의 상태
last_saved = 0
last_state_key = None

print("컵받침 위에 컵 하나를 놓고 약 2초 동안 고정하세요.")
print("q: 종료 / r: 기준 재측정(횟수 유지) / v: 음성 질문")
print("Gemma·음성 안내를 사용하려면 water_assistant.py도 실행하세요.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라 프레임을 읽지 못했습니다.")
        break

    # [수업 04 / 06] YOLO 탐지 결과 추출. 컵 클래스만 선택했습니다.
    result = yolo.predict(
        source=frame, conf=CONFIDENCE, iou=0.5,
        classes=[CUP_CLASS_ID], verbose=False,
    )[0]

    objects = []
    boxes = []
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((x1, y1, x2, y2))
        objects.append({
            "class": result.names[class_id],
            "confidence": round(confidence, 3),
            "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
        })

    selected_box = max(boxes, key=cup_area) if boxes else None
    now = time.monotonic()
    if last_frame_time is not None and now - last_frame_time > MAX_FRAME_GAP:
        samples = []
        calibration_started = None
        absence_started = None
        return_started = None
    last_frame_time = now
    area_ratio = None
    event = None

    # [추가 1] 가장 큰 컵이 안정적으로 보이면 기준 면적·위치를 고정합니다.
    if baseline_area is None:
        if selected_box is None:
            samples = []
            calibration_started = None
        else:
            if samples:
                stable_size = abs(cup_area(selected_box) / cup_area(samples[0]) - 1) <= CALIBRATION_TOLERANCE
                if not stable_size or not same_position(selected_box, samples[0]):
                    samples = []
                    calibration_started = None
            if calibration_started is None:
                calibration_started = now
            samples.append(selected_box)
            if now - calibration_started >= CALIBRATION_SECONDS and len(samples) >= MIN_SAMPLES:
                baseline_area = median(cup_area(box) for box in samples)
                baseline_box = tuple(median(box[i] for box in samples) for i in range(4))
                samples = []
                state = "present"
                event = "calibrated"

    # [추가 2] 사라짐 확인 → 기준 크기·위치로 복귀 확인 → 1회 증가
    elif count < goal:
        area_ratio = cup_area(selected_box) / baseline_area if selected_box else None
        at_home = (
            selected_box is not None
            and MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO
            and same_position(selected_box, baseline_box)
        )

        if state != "absent":
            if selected_box is None:
                if absence_started is None:
                    absence_started = now
                state = "moving"
                if now - absence_started >= ABSENCE_SECONDS:
                    state = "absent"
                    return_started = None
                    event = "absent"
            else:
                # 컵이 보이면 크기가 범위 밖이어도 '사라짐'으로 세지 않습니다.
                absence_started = None
                state = "present" if at_home else "moving"
        else:
            if not at_home:
                return_started = None
            else:
                if return_started is None:
                    return_started = now
                if now - return_started >= RETURN_SECONDS:
                    count += 1
                    absence_started = None
                    return_started = None
                    state = "complete" if count >= goal else "present"
                    event = "complete" if count >= goal else "counted"

    if event:
        print(f"[컵 이벤트] {event} | {count}/{goal}회")

    # [수업 04 / 06] 탐지 결과와 프로젝트 상태를 화면에 표시합니다.
    output_frame = result.plot()
    if baseline_box is not None:
        x1, y1, x2, y2 = map(int, baseline_box)
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        area_ratio = cup_area(selected_box) / baseline_area if selected_box else None

    ratio_text = "--" if area_ratio is None else f"{area_ratio * 100:.0f}%"
    lines = [
        f"Water quest: {count}/{goal}",
        f"State: {state} | Cup area: {ratio_text}",
        "q: quit   r: recalibrate   v: voice question",
    ]
    if state == "calibrating":
        lines.append("Keep ONE cup still on its coaster.")
    elif state == "complete":
        lines.append("QUEST COMPLETE!")
    for index, line in enumerate(lines):
        cv2.putText(output_frame, line, (20, 35 + index * 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Water Drinking Project", output_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("r") and count < goal:
        baseline_area = None
        baseline_box = None
        area_ratio = None
        samples = []
        calibration_started = None
        absence_started = None
        return_started = None
        state = "calibrating"
        print("횟수를 유지하고 컵 기준을 다시 측정합니다.")
    elif key == ord("v"):
        voice_request_id += 1
        print("음성 안내 프로그램의 '말씀을 시작해 주세요' 안내 후 말하세요.")

    # [수업 06-A-7] 아래 Dictionary를 JSON 파일로 저장하여 assistant에 전달합니다.
    # objects: 탐지 결과 / quest: 목표·횟수·상태
    # session_id: 이번 실행 구분 / voice_request_id: v 키 요청 번호
    # running: 실행 중 여부 / updated_at: 마지막 갱신 시각
    vision_data = {
        "image_width": frame.shape[1],
        "image_height": frame.shape[0],
        "objects": objects,
        "quest": {
            "goal": goal, "count": count, "remaining": max(0, goal - count),
            "completed": count >= goal, "state": state,
            "baseline_area": baseline_area, "area_ratio": area_ratio,
        },
        "session_id": session_id,
        "voice_request_id": voice_request_id,
        "running": True,
        "updated_at": time.time(),
    }
    # 0.2초가 지났거나 컵 상태·횟수·질문 요청이 달라지면 저장합니다.
    state_key = (state, count, voice_request_id)
    if now - last_saved >= JSON_INTERVAL or state_key != last_state_key:
        # [수업 06-A-7] Dictionary → 임시 JSON 저장 → 실제 JSON으로 교체
        with open(TEMP_JSON_PATH, "w", encoding="utf-8") as file:
            json.dump(vision_data, file, ensure_ascii=False, indent=4)
        os.replace(TEMP_JSON_PATH, JSON_PATH)
        last_saved = now
        last_state_key = state_key
    if key == ord("q"):
        break


# [수업 06-A-7] 반복문이 끝나면 카메라와 창을 닫습니다. 종료는 q 키를 사용합니다.
cap.release()
cv2.destroyAllWindows()

# 음성 안내 프로그램에도 카메라가 종료되었음을 알려 줍니다.
if vision_data is not None:
    vision_data["running"] = False
    vision_data["updated_at"] = time.time()
    with open(TEMP_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(vision_data, file, ensure_ascii=False, indent=4)
    os.replace(TEMP_JSON_PATH, JSON_PATH)
