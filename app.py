import streamlit as st
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
from google import genai
from collections import Counter
import time
import re

# 1. 페이지 설정
st.set_page_config(page_title="YOLO & Gemini 분석", layout="wide")
st.title("객체 탐지 및 AI 분석 서비스")

# 2. 모델 로드
@st.cache_resource
def load_model():
    model_path = Path("models/best.pt")
    if not model_path.exists():
        st.error("모델 파일을 찾을 수 없습니다.")
        st.stop()
    return YOLO(model_path)

model = load_model()

# 3. 설정
api_key = st.sidebar.text_input("Gemini API Key 입력", type="password")

FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]

def generate_with_fallback(client, prompt, max_retries=3):
    last_error = None
    for model_name in FALLBACK_MODELS:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        "system_instruction": (
                            "당신은 산업 검사 현장의 분석 보조원입니다. "
                            "탐지된 객체 목록만 근거로 짧고 실무적으로 답하세요. "
                            "불필요한 서론, 감탄사, 장황한 설명 없이 3~5문장으로 핵심만 말하세요."
                        ),
                        "max_output_tokens": 300,
                        "temperature": 0.3,
                    },
                )
                return response, model_name
            except Exception as e:
                error_str = str(e)
                last_error = e
                if "503" in error_str or "UNAVAILABLE" in error_str:
                    wait_time = min(2 ** attempt * 3, 20)
                    st.warning(f"{model_name} 서버 혼잡. {wait_time}초 후 재시도... ({attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                elif "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    match = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)", error_str)
                    wait_time = int(match.group(1)) + 2 if match else 15
                    st.warning(f"{model_name} 사용량 한도 초과. {wait_time}초 후 재시도... ({attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    raise
        st.warning(f"{model_name} 계속 실패, 다음 모델로 전환합니다...")
    raise last_error

# 4. 분석 실행
uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    results = model.predict(source=image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("원본")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("탐지 결과")
        st.image(results[0].plot(), use_container_width=True)

    names = model.names
    detected_objects = [names[int(box.cls.item())] for box in results[0].boxes]
    confidences = [float(box.conf.item()) for box in results[0].boxes]

    # ---- 탐지 정보 표시 ----
    st.subheader("탐지된 객체 정보")
    if detected_objects:
        counts = Counter(detected_objects)
        st.write(f"총 탐지 개수: **{len(detected_objects)}개**")

        count_cols = st.columns(len(counts))
        for col, (obj_name, cnt) in zip(count_cols, counts.items()):
            col.metric(obj_name, f"{cnt}개")

        with st.expander("상세 내역 (클래스 / 신뢰도)"):
            for obj_name, conf in zip(detected_objects, confidences):
                st.write(f"- {obj_name}: {conf:.2%}")
    else:
        st.warning("탐지된 객체가 없습니다.")

    # ---- AI 분석 ----
    if api_key and detected_objects:
        if st.button("AI 분석 실행"):
            try:
                client = genai.Client(api_key=api_key)
                counts = Counter(detected_objects)
                summary = ", ".join(f"{k} {v}개" for k, v in counts.items())
                prompt = f"탐지된 객체 목록: {summary}. 이 검사 결과를 간단히 분석해줘."

                with st.spinner("분석 중..."):
                    response, used_model = generate_with_fallback(client, prompt)
                st.caption(f"사용된 모델: {used_model}")
                st.info(response.text)

            except Exception as e:
                st.error(f"분석 오류: {e}")
                st.write("오류가 지속되면 [Google AI Studio](https://aistudio.google.com/)에서 'Create API key'를 눌러 새 키를 발급받으세요.")
    elif not api_key and detected_objects:
        st.warning("사이드바에 Gemini API Key를 입력해주세요.")
