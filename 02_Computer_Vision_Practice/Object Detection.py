import cv2
import numpy as np
import matplotlib.pyplot as plt

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

while True:
    ret, frame = cap.read()

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # TODO: 배운 기술(이미지 전처리, 특징 추출)을 활용하여 실시간으로 객체를 검출

    # TODO: 각 프레임 이미지를 전처리

    ball_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
    # 초록 LAB lower/upper range
    green_lower = np.array([30, 60, 90], dtype=np.uint8)
    green_upper = np.array([230, 115, 180], dtype=np.uint8)

    # 각 채널 분리
    l, a, b = cv2.split(ball_lab)

    # CLAHE 생성 (clipLimit은 2.0, tileGridSize는 8x8로 설정)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # L 채널에 CLAHE 적용 후 강화된 이미지 생성
    l_clahe_enhanced = clahe.apply(l)
    ball_enhanced_lab = cv2.merge((l_clahe_enhanced, a, b))
    
    # TODO: Color Segmentation으로 Binary Mask 생성

    # 강화된 이미지로 binary mask 생성
    mask_enhanced = cv2.inRange(ball_enhanced_lab, green_lower, green_upper)

    # TODO: Morphological Operation을 적용하여 객체 영역 추출
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    ball_mask_opened2 = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel, iterations=2)

    # 최종 binary mask
    final_ball_mask = ball_mask_opened2


    # TODO: 객체의 중심과 반지름을 구해 프레임 위에 overlay

    contours, _ = cv2.findContours(
        final_ball_mask,        
        cv2.RETR_EXTERNAL,       
        cv2.CHAIN_APPROX_SIMPLE  
    )

    if len(contours) > 0:
        # 가장 큰 contour
        largest_contour = max(contours, key=cv2.contourArea)

        # 가장 큰 contour만 채운 binary mask 생성
        largest_mask = np.zeros_like(final_ball_mask)
        cv2.drawContours(
            largest_mask,
            [largest_contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

        # 가장 큰 contour만 채운 binary mask를 반투명 빨간색으로 overlay
        mask_color = np.zeros_like(frame)
        mask_color[:, :, 0] = largest_mask
        alpha = 0.15
        frame = cv2.addWeighted(frame, 1.0, mask_color, alpha, 0)

        # 최소 외접원 중심 및 반지름
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center_circle = (int(x), int(y))
        radius = int(radius)

        # Red Circle
        cv2.circle(frame, center_circle, radius, (0,0,255), 3)
        cv2.circle(frame, center_circle, 5, (0,0,255), -1)

    cv2.imshow("VideoCapture with GStreamer", frame)

cap.release()
cv2.destroyAllWindows()