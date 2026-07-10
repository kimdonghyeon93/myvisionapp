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
                            "불필요한 서론, 감탄사, 장황한 설명 없이 핵심만 말하세요. "
                            "여러 이미지 결과가 주어지면 이미지별로 구분해서 요약하세요."
                        ),
                        "max_output_tokens": 500,
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

# 4. 다중 이미지 업로드
uploaded_files = st.file_uploader(
    "이미지 업로드 (여러 장 선택 가능)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.write(f"총 **{len(uploaded_files)}장** 업로드됨")

    all_results_summary = []  # 이미지별 요약 텍스트 모음 (AI 분석용)
    total_counts = Counter()   # 전체 이미지 통합 클래스별 개수

    for idx, uploaded_file in enumerate(uploaded_files):
        st.divider()
        st.subheader(f"이미지 {idx + 1}: {uploaded_file.name}")

        image = Image.open(uploaded_file)
        results = model.predict(source=image)

        col1, col2 = st.columns(2)
        with col1:
            st.caption("원본")
            st.image(image, use_container_width=True)
        with col2:
            st.caption("탐지 결과")
            st.image(results[0].plot(), use_container_width=True)

        names = model.names
        detected_objects = [names[int(box.cls.item())] for box in results[0].boxes]
        confidences = [float(box.conf.item()) for box in results[0].boxes]
        counts = Counter(detected_objects)
        total_counts.update(counts)

        if detected_objects:
            st.write(f"탐지 개수: **{len(detected_objects)}개**")
            count_cols = st.columns(min(len(counts), 6))
            for col, (obj_name, cnt) in zip(count_cols, counts.items()):
                col.metric(obj_name, f"{cnt}개")

            with st.expander("상세 내역 (클래스 / 신뢰도)"):
                for obj_name, conf in zip(detected_objects, confidences):
                    st.write(f"- {obj_name}: {conf:.2%}")

            summary = ", ".join(f"{k} {v}개" for k, v in counts.items())
            all_results_summary.append(f"[{uploaded_file.name}] {summary}")
        else:
            st.warning("탐지된 객체 없음")
            all_results_summary.append(f"[{uploaded_file.name}] 탐지된 객체 없음")

    # ---- 전체 통합 정보 ----
    st.divider()
    st.subheader("전체 이미지 통합 통계")
    if total_counts:
        total_cols = st.columns(min(len(total_counts), 6))
        for col, (obj_name, cnt) in zip(total_cols, total_counts.items()):
            col.metric(f"전체 {obj_name}", f"{cnt}개")

    # ---- AI 일괄 분석 (API 호출 1회로 처리) ----
    if api_key and all_results_summary:
        if st.button("전체 이미지 AI 일괄 분석 실행"):
            try:
                client = genai.Client(api_key=api_key)
                joined_summary = "\n".join(all_results_summary)
                prompt = (
                    f"총 {len(uploaded_files)}장의 이미지에 대한 검사 결과입니다.\n"
                    f"{joined_summary}\n"
                    "전체적인 검사 결과를 이미지별로 간단히 분석하고, "
                    "전체 경향(불량/이상 패턴 등)도 요약해줘."
                )

                with st.spinner("전체 분석 중..."):
                    response, used_model = generate_with_fallback(client, prompt)
                st.caption(f"사용된 모델: {used_model}")
                st.info(response.text)

            except Exception as e:
                st.error(f"분석 오류: {e}")
                st.write("오류가 지속되면 [Google AI Studio](https://aistudio.google.com/)에서 'Create API key'를 눌러 새 키를 발급받으세요.")
    elif not api_key and all_results_summary:
        st.warning("사이드바에 Gemini API Key를 입력해주세요.")
