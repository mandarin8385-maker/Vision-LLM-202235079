# 물 마시기 프로젝트 2: JSON 읽기 → Gemma 설명 → 음성 출력 / 음성 질문
# 먼저 이 프로그램을 실행하고 'JSON 입력 대기 중'이 뜨면 카메라를 실행
# MIC_DEVICE와 모델 경로 수정 필요

import json
import os
import subprocess
import time
from pathlib import Path
from llama_cpp import Llama


# camera와 같은 하위 폴더에 두며, 한 단계 위의 src를 사용
PROJECT_ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = str(PROJECT_ROOT / "src/json/water_vision_data.json")
GEMMA_MODEL_PATH = str(PROJECT_ROOT / "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf")
CONTEXT_WINDOW = 2048
MAX_TOKENS = 150

MIC_DEVICE = "plughw:2,0"  # arecord -l로 확인한 마이크 card, device 번호
AUDIO_FILE = str(PROJECT_ROOT / "src/audio/input.wav")
WHISPER_PATH = str(PROJECT_ROOT / "whisper.cpp/build-cpu/bin/whisper-cli")
WHISPER_MODEL = str(PROJECT_ROOT / "whisper.cpp/models/ggml-base.bin")
RECORD_SECONDS = 5
PIPER_PYTHON = str(PROJECT_ROOT / ".piper_venv/bin/python")
PIPER_MODEL = str(PROJECT_ROOT / "src/models/Piper/ko_KR-kss-medium.onnx")
OUTPUT_FILE = str(PROJECT_ROOT / "src/audio/response.wav")


# STT 함수
def speech_to_text():
    subprocess.run(
        [
            "pasuspender", "--",
            "arecord",
            "-D", MIC_DEVICE,
            "-f", "S16_LE",
            "-r", "16000",
            "-c", "1",
            "-d", str(RECORD_SECONDS),
            AUDIO_FILE,
        ],
        check=True,
    )

    result = subprocess.run(
        [
            WHISPER_PATH,
            "-m", WHISPER_MODEL,
            "-f", AUDIO_FILE,
            "-l", "ko",
            "--no-timestamps",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    return result.stdout.strip()


# TTS 함수
def text_to_speech(text):
    subprocess.run(
        [
            PIPER_PYTHON,
            "-m",
            "piper",

            "-m",
            PIPER_MODEL,

            "-f",
            OUTPUT_FILE,

            "--",
            text,
        ],
        check=True,
    )

    subprocess.run(
        [
            "aplay",
            OUTPUT_FILE,
        ],
        check=True,
    )


def detections_to_text(vision_data):
    # 파일 읽기는 반복문에서 처리하고 탐지 정보를 문장으로 변환
    objects = vision_data["objects"]
    if len(objects) == 0:
        return "현재 탐지된 객체가 없습니다."
    sentences = []
    for index, obj in enumerate(objects, start=1):
        sentence = f"{index}번 객체는 {obj['class']}이며, confidence는 {obj['confidence']:.2f}입니다."
        sentences.append(sentence)
    return "\n".join(sentences)


def say(text):
    print(f"\n[안내] {text}", flush=True)
    text_to_speech(text)


def explain(vision_data, question):
    quest = vision_data["quest"]
    vision_text = detections_to_text(vision_data)

    # Gemma에 프로젝트 상태와 질문 전달
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": """
                    Instruction:
                    당신은 사용자의 물 마시는 습관을 응원하는 친근한 도우미입니다.
                    이 프로젝트에서는 Context의 확정 횟수를 사용자가 물을 마신 횟수로 간주합니다.
                    물을 마신 사용자에게 자연스럽게 칭찬하고, 목표까지 몇 번 남았는지 알려 주세요.
                    사용자가 질문하면 현재 진행 상황을 바탕으로 답해 주세요.

                    Constraint:
                    횟수와 남은 횟수, 목표 완료 여부는 Context에 제공된 값을 정확히 따르세요.
                    '컵 탐지', '컵 복귀', '컵 반환', '객체', '검출' 같은 기술적인 표현은 사용하지 마세요.
                    물을 마신 양이나 건강 상태는 추측하지 마세요.
                    목표를 완료했다면 추가로 마시라고 권하지 말고 목표 달성을 축하해 주세요.
                    같은 칭찬을 반복하기보다 표현을 자연스럽게 바꿔 주세요.

                    Output Format:
                    친근한 한국어 존댓말로 두 문장 이내.
                    진행 중에는 물 마신 행동에 대한 칭찬과 남은 횟수를 포함하세요.
                    목표 완료 시에는 달성을 축하하는 문장으로 답하세요.
                """,
            },
            {
                "role": "user",
                "content": f"""
                    Context:
                    {vision_text}
                    목표 횟수: {quest['goal']}
                    확정 횟수: {quest['count']}
                    남은 횟수: {quest['remaining']}
                    목표 완료: {quest['completed']}
                    컵 상태: {quest['state']}

                    질문 또는 안내 요청:
                    {question}
                """,
            },
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.7,
    )
    answer = response["choices"][0]["message"]["content"].strip()
    print(f"[Gemma] {answer}", flush=True)

    # Gemma가 만든 칭찬과 진행 상황 안내를 음성으로 출력
    say(answer)


Path(AUDIO_FILE).parent.mkdir(parents=True, exist_ok=True)
llm = Llama(
    model_path=GEMMA_MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=CONTEXT_WINDOW,
    n_batch=32,
    n_ubatch=32,
    verbose=False,
)

current_session = None
last_count = 0
last_voice_request = 0
last_modified = 0

print("컵 카메라 JSON 입력 대기 중... 종료: Ctrl+C", flush=True)
print("카메라 창에서 v 키 → 안내가 끝나면 5초 동안 질문하세요.", flush=True)

while True:
    # JSON 파일 존재 여부와 수정 시각을 확인하는 반복문
    if os.path.exists(JSON_PATH):
        current_modified = os.path.getmtime(JSON_PATH)

        if current_modified != last_modified:
            last_modified = current_modified
            # 변경된 JSON 파일을 바로 읽기
            with open(JSON_PATH, "r", encoding="utf-8") as file:
                vision_data = json.load(file)

            session_id = vision_data["session_id"]
            quest = vision_data["quest"]

            if not vision_data["running"]:
                # 카메라 종료 직전에 완료된 경우도 한 번 안내
                if session_id == current_session and quest["completed"] and quest["count"] > last_count:
                    last_count = quest["count"]
                    explain(vision_data, "목표 달성을 한 문장으로 축하해 주세요.")
                time.sleep(0.1)
                continue

            # 카메라를 다시 실행하면 새 목표와 진행 상황을 사용
            if session_id != current_session:
                current_session = session_id
                last_count = 0
                last_voice_request = 0
                if not quest["completed"]:
                    say(f"물 마시기 프로젝트를 시작합니다. 목표는 {quest['goal']}회입니다.")

            # 횟수 증가 / 목표 완료 때만 Gemma를 실행
            if quest["count"] > last_count:
                last_count = quest["count"]
                if quest["completed"]:
                    explain(vision_data, "목표 달성을 한 문장으로 축하해 주세요.")
                else:
                    explain(vision_data, "사용자가 물을 한 번 마셨습니다. 자연스럽게 칭찬하고 목표까지 남은 횟수를 알려 주세요.")

            # 카메라 창의 v 키 요청 → STT → Gemma → TTS
            if vision_data["voice_request_id"] > last_voice_request:
                last_voice_request = vision_data["voice_request_id"]
                say("말씀을 시작해 주세요. 5초 동안 녹음합니다.")
                question = speech_to_text()
                print(f"[STT] {question}", flush=True)

                # 녹음 중에도 카메라는 계속 실행되므로 최신 횟수를 읽음
                with open(JSON_PATH, "r", encoding="utf-8") as file:
                    latest = json.load(file)
                if latest["running"] and latest["session_id"] == current_session:
                    explain(latest, question)
                else:
                    print("현재 카메라 상태를 확인할 수 없습니다. 질문을 다시 요청해 주세요.")

    time.sleep(0.1)
