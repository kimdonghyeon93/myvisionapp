import streamlit as st
from PIL import Image
from ultralytics import YOLO
from pathlib import Path

# 1. 페이지 설정
st.set_page_config(page_title="YOLO 자동 탐지 서비스", layout="wide")
st.title("객체 탐지 서비스 (YOLO)")

# 2. 모델 로드
@st.cache_resource
def load_model():
    model_path = Path("models/best.pt")
    if not model_path.exists():
        st.error(f"모델 파일을 찾을 수 없습니다: {model_path.absolute()}")
        st.stop()
    return YOLO(model_path)import streamlit as st
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import google.generativeai as genai

# 1. 페이지 설정 및 API 키 설정
st.set_page_config(page_title="YOLO & Gemini 분석", layout="wide")
st.title("객체 탐지 및 AI 분석 서비스")

# API 키 입력 (사이드바에서 설정)
api_key = st.sidebar.text_input("Gemini API Key 입력", type="password")

@st.cache_resource
def load_model():
    model_path = Path("models/best.pt")
    if not model_path.exists():
        st.error(f"모델 파일을 찾을 수 없습니다.")
        st.stop()
    return YOLO(model_path)

model = load_model()

# 2. 사이드바 설정
st.sidebar.header("탐지 설정")
conf_threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.25, 0.05)

uploaded_file = st.file_uploader("이미지를 업로드하세요...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file)
    
    with col1:
        st.image(image, caption="업로드 이미지", use_container_width=True)
    
    # 탐지 실행
    results = model.predict(source=image, conf=conf_threshold)
    res_plotted = results[0].plot()
    
    with col2:
        st.image(res_plotted, caption="탐지 결과", use_container_width=True)
        
    # 정보 추출
    names = model.names
    detected_objects = [names[int(box.cls.item())] for box in results[0].boxes]
    obj_count = len(detected_objects)
    
    # UI에 개수 및 목록 표시
    st.subheader("탐지 요약")
    st.write(f"**총 탐지된 객체 수:** {obj_count}개")
    st.write(f"**객체 목록:** {', '.join(detected_objects)}")
    
    # 3. Gemini API를 이용한 해석
    if api_key and obj_count > 0:
        if st.button("AI 분석 실행"):
            try:
                genai.configure(api_key=api_key)
                gemini = genai.GenerativeModel('gemini-1.5-flash')
                
                prompt = f"다음 객체들이 이미지에서 탐지되었습니다: {', '.join(detected_objects)}. 이 객체들을 바탕으로 현재 이미지의 상황이나 주의사항을 간단히 분석해줘."
                response = gemini.generate_content(prompt)
                
                st.subheader("AI 분석 결과")
                st.info(response.text)
            except Exception as e:
                st.error(f"분석 중 오류 발생: {e}")
    elif not api_key:
        st.warning("분석을 위해 사이드바에 API 키를 입력하세요.")

model = load_model()

# 3. 사이드바 설정
st.sidebar.header("탐지 설정")
conf_threshold = st.sidebar.slider("Confidence Threshold (신뢰도)", 0.0, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold (중복 박스 제거)", 0.0, 1.0, 0.7, 0.05)

# 4. 파일 업로드
uploaded_file = st.file_uploader("이미지를 업로드하세요...", type=["jpg", "jpeg", "png"])

# 5. 자동 탐지 로직
if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption="업로드된 이미지", use_container_width=True)
    
    # 별도의 버튼 없이 즉시 실행
    with st.spinner("자동 탐지 중..."):
        results = model.predict(source=image, conf=conf_threshold, iou=iou_threshold)
        
        res_plotted = results[0].plot()
        with col2:
            st.image(res_plotted, caption="탐지 결과", use_container_width=True)
        
        st.subheader("탐지 결과 상세")
        for i, box in enumerate(results[0].boxes):
            class_id = int(box.cls.item())
            class_name = model.names[class_id]
            conf = float(box.conf.item())
            st.write(f"- **{class_name}**: {conf:.2%} 정확도")
