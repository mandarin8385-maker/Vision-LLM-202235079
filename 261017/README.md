# Assignment 2 - Basketball Segmentation

## 파일 구조

- assigment02.py: 농구공 영역 추출 및 이미지 처리 코드
- README.md: 코드 설명 및 실행 방법
- basketball_crop.jpg: 이미지 처리에 사용되는 원본 이미지 (../src/images/ 경로에서 불러옴)

## 실행 방식
필요한 라이브러리: OpenCV, NumPy, Matplotlib, scikit-image
1. Python IDLE에서 assigment2.py 파일을 엽니다.
2. Run Module(F5)을 통해 코드를 실행합니다.
3. Matplotlib 창에서 최종 농구공 영역 및 외곽선 검출 결과를 확인합니다.
* 이미지 파일은 실행 위치를 기준으로 ../src/images/basketball_crop.jpg 경로에 있어야 합니다.

## 기타 참고 사항

### 작업 중 경험한 문제점

Python 환경에서 OpenCV와 Matplotlib 라이브러리가 설치되어 있지 않아 ModuleNotFoundError가 발생하였습니다.

필요한 라이브러리를 설치하여 문제를 해결하였으며, Matplotlib을 이용한 결과 출력 과정에서 plt.show()가 누락되어 이미지가 표시되지 않는 문제도 수정하였습니다.

### AI 활용

ChatGPT를 활용하여 Python 실행 환경에서 발생한 라이브러리 설치 오류와 이미지 출력 문제를 해결하는 데 도움을 받았습니다.

### 추가 설명

Morphological Operation을 여러 번 적용한다고 반드시 좋은 결과가 나오는 것은 아니라고 판단하였습니다.

따라서 Opening과 Closing의 적용 횟수를 변경하며 결과를 비교하였고, Closing을 추가로 적용해도 뚜렷한 차이가 없어 Opening 2회를 최종적으로 선택하였습니다.
