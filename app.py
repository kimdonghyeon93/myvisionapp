import streamlit as st
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
from google import genai
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

def generate_with_retry(client, model_name, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=model_name, contents=prompt)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < max_retries - 1:
                    match = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)", error_str)
                    wait_time = int(match.group(1)) + 2 if match else 15
                    st.warning(f"무료 사용량 한도 초과. {wait_time}초 후 재시도합니다... ({attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
            elif "503" in error_str or "UNAVAILABLE" in error_str:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    st.warning(f"서버 혼잡. {wait_time}초 후 재시도합니다... ({attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
            raise
    raise Exception("여러 번 재시도했지만 실패했습니다.")

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

    if api_key and detected_objects:
        if st.button("AI 분석 실행"):
            try:
                client = genai.Client(api_key=api_key)
                prompt = f"탐지된 객체: {', '.join(detected_objects)}. 상황 분석해줘."

                with st.spinner("분석 중..."):
                    response = generate_with_retry(client, "gemini-3.5-flash", prompt)
                st.info(response.text)

            except Exception as e:
                st.error(f"분석 오류: {e}")
                st.write("오류가 지속되면 [Google AI Studio](https://aistudio.google.com/)에서 'Create API key'를 눌러 새 키를 발급받으세요.")
    elif not detected_objects:
        st.warning("탐지된 객체가 없습니다.")
    elif not api_key:
        st.warning("사이드바에 Gemini API Key를 입력해주세요.")
