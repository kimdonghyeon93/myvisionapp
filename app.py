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
    return YOLO(model_path)

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
