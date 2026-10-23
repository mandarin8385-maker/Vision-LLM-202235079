# <center>< Section Project ></center>

# 본 섹션에서 배운 내용을 토대로 프로젝트를 진행합니다.<br>
# 모든 기술 (이미지 조정, 색공간 변환, Image Enhancement, 필터링, Color Segmentation, Contour Detection, 객체 추적, 신체 인식)을 활용하여, 자유롭게 객체를 인식하고 추적하며 신체와의 관계성을 통해 결과를 텍스트 혹은 이미지 overlay로 표현합시다 (결과 표현은 창의적으로 다른 방식으로도 허용).

# TODO: Section 2 "Computer Vision" Project

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from datetime import datetime


# 중간 점 기준으로 각도 계산하는 함수
def calculate_angle(p1, p2, p3):
    vector1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    vector2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])

    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    cosine = np.dot(vector1, vector2) / (magnitude1 * magnitude2)
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))

# AI의 도움을 받아 작성한 함수, 파일을 저장하는 역할을 함.
def save_frame(frame):
    # 이 Python 파일이 있는 폴더 안에 저장 폴더 지정
    save_folder = Path(__file__).resolve().parent / "saved_images"
    # 폴더 생성. 이미 있으면 그대로 사용
    save_folder.mkdir(exist_ok=True)

    # 저장할 때마다 다른 이름이 되도록 현재 시간 사용
    file_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".png"
    save_path = save_folder / file_name

    # 글씨와 색상까지 그려진 프레임 저장
    success = cv2.imwrite(str(save_path), frame)

    if success == True:
        print("저장 완료:", save_path)
    else:
        print("이미지 저장 실패")

base_option = python.BaseOptions(model_asset_path="src/models/MediaPipe/hand_landmarker.task")  # 모델 경로 지정하는 옵션
options = vision.HandLandmarkerOptions(base_options=base_option, num_hands=2)                   # 모델 경로와 최대 손 개수 지정
hand_detector = vision.HandLandmarker.create_from_options(options)                              # 해당 옵션으로 손 검출하는 객체 생성

angle_threshold = 160             # 손가락 펼쳐짐 여부 판단 각도 임계값

# 각 손가락의 각도를 계산할 landmark index
finger_angle_points = (
    (1, 2, 3),      # 엄지: 2번 중심
    (5, 6, 7),      # 검지: 6번 중심
    (9, 10, 11),    # 중지: 10번 중심
    (13, 14, 15),   # 약지: 14번 중심
    (17, 18, 19),   # 소지: 18번 중심
)

#####################################################
# macOS 카메라 연결. Jetson에서는 카메라 연결 코드 변경 필요
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")
#####################################################

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 프레임 높이와 너비
    h, w = frame.shape[:2]

    # 이미지 좌우 반전 및 RGB로 색공간 변환 (전처리)
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 프레임 내 손 탐지
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = hand_detector.detect(mp_image)
    # 손목의 x좌표를 기준으로 화면 왼쪽부터 정렬
    # 감지된 손들을 손목(0번 관절)의 x좌표가 작은 순서,
    # 즉 화면 왼쪽부터 오른쪽 순서로 정렬한다.
    hands = sorted(result.hand_landmarks, key=lambda hand: hand[0].x)
    
    # 손마다 펼친 손가락 개수를 저장할 리스트
    finger_counts = []

    # 02_Computer-Vision.ipynb 실시간 신체 인식 부분
    # Rule-based (규칙 기반) 시스템으로 펼쳐진 손가락 개수 계산 참고
    for hand in hands:
        finger_count = 0

        for point1_idx, point2_idx, point3_idx in finger_angle_points:
            angle = calculate_angle(
                hand[point1_idx],
                hand[point2_idx],
                hand[point3_idx],
            )

            if angle >= angle_threshold:
                finger_count += 1

        finger_counts.append(finger_count)

    # 손가락 개수에 따라 손 모양 결정
    hand_shapes = []

    for count in finger_counts:
        # 엄지를 검지 옆에 붙여 주먹을 쥐면,
        # 엄지 관절 각도 때문에 손가락이 1개 펴진 것으로 계산될 수 있다.
        # 자연스럽게 쥔 주먹도 인식하도록 0개와 1개를 모두 바위로 처리한다.
        if count == 0 or count == 1:
            shape = "Rock"
        # 엄지를 검지 옆에 붙여 가위를 만들면,
        # 엄지 관절 각도 때문에 엄지가 추가로 계산되어 총 3개로 인식될 수 있다
        # 자연스럽게 만든 가위도 인식하도록 2개와 3개를 모두 가위로 처리한다.
        elif count == 2 or count == 3:
            shape = "Scissors"
        elif count == 5:
            shape = "Paper"
        else:
            shape = "Unknown"

        hand_shapes.append(shape)

    # 감지된 손 개수를 플레이어 수로 표시
    player_count = len(hands)

    if player_count == 2:
        status = "Ready"
        status_color = (0,255,0)      # 초록색
    else:
        status = "Waiting for 2 players"
        status_color = (0,255,255)    # 노란색

    cv2.putText(frame, f"Players: {player_count}/2", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(frame, status, (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    # 기본 색: 노란색
    player_colors = [(0,255,255), (0,255,255)]

    if len(hands) == 2:
        left_shape = hand_shapes[0]
        right_shape = hand_shapes[1]

        # 두 손 모두 판정 가능하고, 서로 다른 모양일 때 승패 결정
        if "Unknown" not in hand_shapes and left_shape != right_shape:
            if (
                (left_shape == "Rock" and right_shape == "Scissors")
                or (left_shape == "Scissors" and right_shape == "Paper")
                or (left_shape == "Paper" and right_shape == "Rock")
            ):
                # 왼쪽 승리, 오른쪽 패배
                player_colors[0] = (0,255,0)
                player_colors[1] = (0,0,255)
            else:
                # 왼쪽 패배, 오른쪽 승리
                player_colors[0] = (0,0,255)
                player_colors[1] = (0,255,0)

    # 탐지 결과의 각 손마다 가위바위보 표시하기
    for index, hand in enumerate(hands):
        h, w = frame.shape[:2]  # 프레임 높이와 너비
        points = [(int(p.x * w), int(p.y * h)) for p in hand]  # 프레임 높이와 너비 길이 기준 각 landmark 좌표
        # 손의 가장 왼쪽 위치에서 글씨 시작
        text_x = min([point[0] for point in points])
        # 손의 가장 위쪽보다 20픽셀 위에 표시
        text_y = max(70, min([point[1] for point in points]) - 20)
        # 현재 손에 해당하는 가위바위보 모양
        shape = hand_shapes[index]
        cv2.putText(frame, shape, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, player_colors[index], 3)

    # 상단에 표시할 기본 문구와 색상
    result_text = "Waiting for 2 players"
    result_color = (0,255,255)

    if len(hands) == 2:
        # 왼쪽 글씨가 초록색이면 왼쪽 승리
        if player_colors[0] == (0,255,0):
            result_text = "Left wins!"
            result_color = (0,255,0)

        # 오른쪽 글씨가 초록색이면 오른쪽 승리
        elif player_colors[1] == (0,255,0):
            result_text = "Right wins!"
            result_color = (0,255,0)

        elif left_shape == "Unknown" or right_shape == "Unknown":
            result_text = "Unknown gesture"
        
        # 두 손이 같은 가위바위보 모양이면 무승부
        elif left_shape == right_shape:
            result_text = "Draw"
        
        else:
            result_text = "Unable to judge"

    # getTextSize()는 ((글씨 너비, 높이), 기준선 관련 값)을 반환한다.
    # 첫 번째 [0]으로 (너비, 높이)를 꺼내 text_size에 저장한다.
    # text_size[0]으로 글씨 너비를 꺼내 중앙 시작 위치를 계산한다.
    text_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
    result_x = (w - text_size[0]) // 2

    cv2.putText(frame, result_text, (result_x,150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, result_color, 2) 

    # 모든 손의 글씨를 그린 뒤 키 입력 확인
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("s"):
        save_frame(frame)
    
    cv2.imshow("Rock Paper Scissors", frame)

hand_detector.close()
cap.release()
cv2.destroyAllWindows()