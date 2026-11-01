# Assignment 03 - GPU 가속 및 딥러닝 실습

## 파일 구조

- **assignment03_1.py**: GPU 가속 잔상 효과
- **assignment03_2.py**: GPU 가속 Convolution2D
- **assignment03_3.py**: MNIST 데이터셋 `main_gpu()` 학습
- **assignment03_4.py**: CIFAR-10 모델 학습 및 저장
- **assignment03_5.py**: IoU 구현
- **assignment03_6.py**: NMS 구현
- **CIFAR10_CNN.pth**: 4번 학습 완료 후 `261025` 폴더에 생성되는 모델 파일
- **../src/images/**: 입력 이미지 `seagull.jpg`, `two_apples1.png`~`two_apples3.png`
- **../src/datasets/**: MNIST·CIFAR-10 데이터셋
- **../src/models/CIFAR10/CIFAR10_CNN.pth**: 4번 학습 완료 후 생성되는 모델 파일

## 실행 방식

필요한 라이브러리: NumPy, OpenCV, Matplotlib, CuPy, PyTorch, Torchvision, scikit-image

필요한 라이브러리가 설치된 Python 환경을 선택하고, 터미널의 현재 위치를 `261025` 폴더로 맞춥니다. `src`와 `261025`는 같은 상위 폴더 아래에 있어야 합니다.

```bash
python assignment03_1.py
```

2~6번도 파일명을 바꿔 각각 실행합니다. 1·2·5·6번은 최종 이미지를 표시하고, 3·4번은 에포크별 Loss·Accuracy와 소요 시간을 출력합니다.

## 기타 참고 사항

- macOS의 VS Code에서 작성했습니다. GPU 전체 실행 확인과 4번 모델 파일 생성은 학교 CUDA 환경에서 진행할 예정입니다.
- 데이터셋이 없으면 처음 실행할 때 다운로드합니다. 4번은 학습한 모델을 `../src/models/CIFAR10/`와 현재 실행 폴더에 각각 저장합니다. 저장 폴더를 미리 준비하고, `261025` 폴더에서 실행해야 합니다.
- 6번은 박스와 신뢰도를 무작위로 생성하므로 결과가 달라질 수 있습니다.
- **AI 활용**: ChatGPT로 TODO 구현 예시, Python 문법·오류 설명, README 정리에 도움을 받았으며, 코드는 직접 파일에 반영했습니다.
