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
        st.error(f"모델 파일을 찾을 수 없습니다: {model_path.absolute()}")
        st.stop()
    return YOLO(model_path)

model = load_model()

# 3. 사이드바 설정
st.sidebar.header("설정")
api_key = st.sidebar.text_input("Gemini API Key 입력", type="password")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# 4. 파일 업로드
uploaded_file = st.file_uploader("이미지를 업로드하세요...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file)
    
    with col1:
        st.image(image, caption="업로드 이미지", use_container_width=True)
    
    with st.spinner("탐지 중..."):
        results = model.predict(source=image, conf=conf_threshold)
        res_plotted = results[0].plot()
    
    with col2:
        st.image(res_plotted, caption="탐지 결과", use_container_width=True)
        
    names = model.names
    detected_objects = [names[int(box.cls.item())] for box in results[0].boxes]
    obj_count = len(detected_objects)
    
    st.subheader("탐지 요약")
    st.write(f"**총 탐지된 객체 수:** {obj_count}개")
    st.write(f"**객체 목록:** {', '.join(detected_objects) if detected_objects else '없음'}")
    
    # 5. Gemini 분석 로직 (동적 모델 선택으로 404 오류 방지)
    if api_key and obj_count > 0:
        if st.button("AI 분석 실행"):
            try:
                genai.configure(api_key=api_key)
                
                # 오류 방지: 사용 가능한 모델 목록 중 첫 번째 모델을 자동으로 선택
                models = [m for m in genai.list_models() if "generateContent" in m.supported_methods]
                if not models:
                    st.error("사용 가능한 모델을 찾을 수 없습니다. API 키 권한을 확인하세요.")
                else:
                    selected_model = models[0]
                    gemini = genai.GenerativeModel(selected_model.name)
                    
                    prompt = f"탐지된 객체: {', '.join(detected_objects)}. 이미지의 상황이나 주의사항을 간단히 분석해줘."
                    
                    with st.spinner(f"{selected_model.name}으로 분석 중..."):
                        response = gemini.generate_content(prompt)
                        st.subheader("AI 분석 결과")
                        st.info(response.text)
            except Exception as e:
                st.error(f"분석 중 오류 발생: {e}")
                st.write("팁: 404 오류가 계속되면 Google AI Studio에서 새 API 키를 발급받아보세요.")
    elif obj_count > 0:
        st.warning("분석을 위해 사이드바에 API 키를 입력하세요.")
