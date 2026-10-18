import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.data

basketball = cv2.imread('../src/images/basketball_crop.jpg')
basketball_rgb = cv2.cvtColor(basketball, cv2.COLOR_BGR2RGB)

# 이미지를 HSV 색공간으로 변환
basketball_hsv = cv2.cvtColor(basketball, cv2.COLOR_BGR2HSV)

# lower_orange_hsv, upper_orange_hsv 배열 생성 (농구공 Hue는 대략 5~25, S와 V 범위는 시도)
lower_orange_hsv = np.array([5, 30, 30])
upper_orange_hsv = np.array([25, 235, 235])

# HSV mask 생성
mask_hsv = cv2.inRange(basketball_hsv, lower_orange_hsv, upper_orange_hsv)

lab = cv2.cvtColor(basketball, cv2.COLOR_BGR2LAB)

lower_orange_lab = np.array([ 70, 130,  70])
upper_orange_lab = np.array([180, 230, 185])

mask_lab = cv2.inRange(lab, lower_orange_lab, upper_orange_lab)

result_hsv = cv2.bitwise_and(basketball_rgb, basketball_rgb, mask=mask_hsv)
result_lab = cv2.bitwise_and(basketball_rgb, basketball_rgb, mask=mask_lab)

lower_basketball_lab = np.array([70, 130, 70])
upper_basketball_lab = np.array([180, 230, 185])

# LAB 색공간으로 변환
basketball_lab = cv2.cvtColor(basketball, cv2.COLOR_BGR2LAB)

# 각 채널 분리
l, a, b = cv2.split(basketball_lab)

# CLAHE 생성 (clipLimit은 3.0, tileGridSize는 8x8로 설정)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# L 채널에 CLAHE 적용 후 강화된 이미지 생성
l_clahe_enhanced = clahe.apply(l)
basketball_enhanced_lab = cv2.merge((l_clahe_enhanced, a, b))

# 강화된 이미지로 binary mask 생성
mask_enhanced = cv2.inRange(basketball_enhanced_lab, lower_basketball_lab, upper_basketball_lab)

# 강화된 이미지를 RGB로 변환 후 마스크를 통해 이미지 내 관심 영역만 추출
basketball_enhanced_rgb = cv2.cvtColor(basketball_enhanced_lab, cv2.COLOR_LAB2RGB)
final_result = cv2.bitwise_and(basketball_enhanced_rgb, basketball_enhanced_rgb, mask=mask_enhanced)

# TODO: Morphological operation을 적용하여 최종 binary mask 생성
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# 2번이 좋은 binary mask라고 판단
ball_mask_opened1 = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel)
ball_mask_opened2 = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel, iterations=2)
ball_mask_opened3 = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel, iterations=3)

# closing을 1~3번 해봤지만 그닥 차이를 못느낌. opening 2회를 적용하여 최종 binary mask 생성.
ball_mask_closed1 = cv2.morphologyEx(ball_mask_opened2, cv2.MORPH_CLOSE, kernel)
ball_mask_closed2 = cv2.morphologyEx(ball_mask_opened2, cv2.MORPH_CLOSE, kernel, iterations=2)
ball_mask_closed3 = cv2.morphologyEx(ball_mask_opened2, cv2.MORPH_CLOSE, kernel, iterations=3)

# 최종 binary mask
final_basketball_mask = ball_mask_opened2

# TODO: Binary mask로 외곽선 검출
# TODO: 최종 농구공 영역 및 외곽선 선택 

# Contour를 그릴 원본 이미지 copy
final_img = basketball.copy()

# 컨투어 추출 
contours, _ = cv2.findContours(
    final_basketball_mask,        
    cv2.RETR_EXTERNAL,       
    cv2.CHAIN_APPROX_SIMPLE  
)

if len(contours) > 0:
    # 가장 큰 contour
    largest_contour = max(contours, key=cv2.contourArea)

    # 가장 큰 contour만 채운 binary mask 생성
    final_mask = np.zeros_like(final_basketball_mask)
    cv2.drawContours(
        final_mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # 가장 큰 contour만 채운 binary mask를 반투명 초록색으로 overlay
    mask_color = np.zeros_like(final_img)
    mask_color[:, :, 1] = final_mask
    alpha = 0.35
    final_img = cv2.addWeighted(final_img, 1.0, mask_color, alpha, 0)

    # Contour 그리기 (초록색 외곽선)
    cv2.drawContours(final_img, [largest_contour], -1, (0,255,0), 2)

plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
