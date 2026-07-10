import streamlit as st
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import google.generativeai as genai

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

# 4. 분석 실행
uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    results = model.predict(source=image)
    st.image(results[0].plot())
    
    names = model.names
    detected_objects = [names[int(box.cls.item())] for box in results[0].boxes]
    
    if api_key and detected_objects:
        if st.button("AI 분석 실행"):
            try:
                genai.configure(api_key=api_key)
                
                # 핵심 수정: 모델명을 직접 넣지 말고, 
                # 공식적으로 지원되는 'gemini-1.5-flash'를 명확하게 호출
                gemini = genai.GenerativeModel("gemini-1.5-flash")
                
                prompt = f"탐지된 객체: {', '.join(detected_objects)}. 상황 분석해줘."
                
                response = gemini.generate_content(prompt)
                st.info(response.text)
                
            except Exception as e:
                st.error(f"분석 오류: {e}")
                st.write("오류가 지속되면 [Google AI Studio](https://aistudio.google.com/)에서 'Create API key'를 눌러 새 키를 발급받으세요.")
