# 물 마시기 프로젝트 2: JSON 읽기 → Gemma 설명 → 음성 출력 / 음성 질문
# 수업 06-A-7의 분리 실행과 06-C의 STT·TTS 함수를 사용합니다.
# water_camera.py와 같은 폴더에 두고 다른 터미널에서 실행하세요.
# 실행: python water_assistant.py
# 먼저 이 프로그램을 실행하고 'JSON 입력 대기 중'이 뜨면 카메라를 실행합니다.
# MIC_DEVICE와 모델 경로를 자신의 수업 환경에 맞게 수정하세요.

import json
import os
import subprocess
import time
from pathlib import Path
from llama_cpp import Llama


# -------------------- 설정 --------------------
# camera와 같은 하위 폴더에 두며, 한 단계 위의 src를 사용합니다.
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


# -------------------- 함수 정의 --------------------
# [수업 06-C] STT 함수 본문 그대로
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


# [수업 06-C] TTS 함수 본문 그대로
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
    # [수업 06-A] 파일 읽기는 반복문에서 하고 객체 설명 문장은 그대로 사용
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
    if quest["completed"]:
        facts = f"퀘스트 완료! 목표 {quest['goal']}회 중 {quest['count']}회를 기록했습니다."
    else:
        facts = f"목표 {quest['goal']}회 중 {quest['count']}회, 남은 횟수는 {quest['remaining']}회입니다."

    # [수업 06-A / 05] 기존 messages 구조에 프로젝트 문맥과 질문을 넣습니다.
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": """
                    Instruction:
                    당신은 '물 마시기 프로젝트'의 안내 도우미입니다.
                    제공된 컵 탐지 정보와 확정 횟수에 따라 질문에 답하거나 격려하시오.

                    Constraint:
                    횟수와 완료 여부는 Python이 계산한 Context를 그대로 따르시오.
                    기록은 컵이 사라졌다 복귀한 횟수이며 실제 마신 양은 알 수 없습니다.
                    탐지 결과에 없는 사실을 추측하지 마시오.

                    Output Format:
                    한국어 두 문장 이내.
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
        temperature=0.3,
    )
    answer = response["choices"][0]["message"]["content"].strip()
    print(f"[Gemma] {answer}", flush=True)

    # 확정된 숫자는 Python이 직접 읽어 주고 Gemma의 설명을 이어 붙입니다.
    say(f"{facts} {answer}")


# -------------------- 여기부터 바로 실행 --------------------
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
    # [수업 06-A-7] JSON 파일 존재 여부와 수정 시각을 확인하는 반복문
    if os.path.exists(JSON_PATH):
        current_modified = os.path.getmtime(JSON_PATH)

        if current_modified != last_modified:
            last_modified = current_modified
            # [수업 06-A-7] 변경된 JSON 파일을 바로 읽습니다.
            with open(JSON_PATH, "r", encoding="utf-8") as file:
                vision_data = json.load(file)

            session_id = vision_data["session_id"]
            quest = vision_data["quest"]

            if not vision_data["running"]:
                # 카메라 종료 직전에 완료된 경우도 한 번 안내합니다.
                if session_id == current_session and quest["completed"] and quest["count"] > last_count:
                    last_count = quest["count"]
                    explain(vision_data, "목표 달성을 한 문장으로 축하해 주세요.")
                time.sleep(0.1)
                continue

            # 카메라를 다시 실행하면 새 목표와 진행 상황을 사용합니다.
            if session_id != current_session:
                current_session = session_id
                last_count = 0
                last_voice_request = 0
                if not quest["completed"]:
                    say(f"물 마시기 프로젝트를 시작합니다. 목표는 {quest['goal']}회입니다.")

            # [추가] 횟수 증가 / 목표 완료 때만 Gemma를 실행합니다.
            if quest["count"] > last_count:
                last_count = quest["count"]
                if quest["completed"]:
                    explain(vision_data, "목표 달성을 한 문장으로 축하해 주세요.")
                else:
                    explain(vision_data, "컵 복귀가 기록되었습니다. 짧게 격려해 주세요.")

            # [추가] 카메라 창의 v 키 요청 → STT → Gemma → TTS
            if vision_data["voice_request_id"] > last_voice_request:
                last_voice_request = vision_data["voice_request_id"]
                say("말씀을 시작해 주세요. 5초 동안 녹음합니다.")
                question = speech_to_text()
                print(f"[STT] {question}", flush=True)

                # 녹음하는 동안 카메라는 계속 실행되므로 최신 횟수를 읽습니다.
                with open(JSON_PATH, "r", encoding="utf-8") as file:
                    latest = json.load(file)
                if latest["running"] and latest["session_id"] == current_session:
                    explain(latest, question)
                else:
                    print("현재 카메라 상태를 확인할 수 없습니다. 질문을 다시 요청해 주세요.")

    time.sleep(0.1)