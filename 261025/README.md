# Assignment 03 - GPU 가속 및 딥러닝 실습

## 파일 구조

- **assignment03_1.py**: GPU 가속 잔상 효과
- **assignment03_2.py**: GPU 가속 Convolution2D
- **assignment03_3.py**: MNIST 데이터셋 `main_gpu()` 학습
- **assignment03_4.py**: CIFAR-10 모델 학습 및 저장
- **assignment03_5.py**: IoU 구현
- **assignment03_6.py**: NMS 구현
- **CIFAR10_CNN.pth**: 제출용 CIFAR-10 학습 모델
- **../src/images/**: 입력 이미지 `seagull.jpg`, `two_apples1.png`~`two_apples3.png`
- **../src/datasets/**: MNIST·CIFAR-10 데이터셋 저장 폴더

## 실행 방식

필요한 라이브러리: NumPy, OpenCV, Matplotlib, CuPy, PyTorch, Torchvision, scikit-image

1~4번은 NVIDIA CUDA GPU 환경에서, 5~6번은 CPU 환경에서도 실행할 수 있습니다.

`src`와 `261025`는 같은 상위 폴더 아래에 있어야 합니다. 아래는 터미널 위치가 `261025`일 때의 실행 예시입니다.

```bash
python assignment03_1.py
```

2~6번도 파일명을 바꿔 각각 실행합니다. 1·2·5·6번은 최종 이미지를 표시하고, 3·4번은 에포크별 Loss·Accuracy와 소요 시간을 출력합니다.

## 기타 참고 사항

- 경로는 각 Python 파일의 위치를 기준으로 찾습니다.
- 데이터셋이 없으면 처음 실행할 때 다운로드합니다.
- 4번은 모델을 `261025/CIFAR10_CNN.pth`와 `src/models/CIFAR10/CIFAR10_CNN.pth`에 각각 저장합니다. `src/models/CIFAR10/` 폴더는 미리 준비해야 합니다.
- 6번은 박스와 신뢰도를 무작위로 생성하므로 결과가 달라질 수 있습니다.
- **AI 활용**: ChatGPT로 TODO 구현 예시, Python 문법·오류 설명, README 정리에 도움을 받았으며, 코드는 직접 파일에 반영했습니다.
