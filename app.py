import streamlit as st
import pandas as pd
import numpy as np
import cv2
import torch
from torch import nn
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
from google import genai
from collections import Counter
import time
import re

# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(page_title="케이블 분류 · 검출 서비스", layout="wide")
st.title("케이블 분류 / 검출 서비스")

# =========================================================
# 공통: Gemini 호출 (fallback + 재시도)
# =========================================================
FALLBACK_MODELS = ["gemini-3.1-flash-lite", "gemini-2.5-flash-lite", "gemini-2.5-flash"]

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
                            "케이블 검출/분류 결과만 근거로 짧고 실무적으로 답하세요. "
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


def render_carousel_nav(session_prefix, total):
    """이전/다음 버튼 캐러셀 네비게이션 UI. 현재 인덱스를 리턴."""
    idx_key = f"{session_prefix}_idx"
    if idx_key not in st.session_state:
        st.session_state[idx_key] = 0

    nav_col1, nav_col2, nav_col3 = st.columns([1, 4, 1])
    with nav_col1:
        if st.button("◀ 이전", use_container_width=True,
                      disabled=(st.session_state[idx_key] == 0),
                      key=f"{session_prefix}_prev"):
            st.session_state[idx_key] -= 1
            st.rerun()
    with nav_col3:
        if st.button("다음 ▶", use_container_width=True,
                      disabled=(st.session_state[idx_key] == total - 1),
                      key=f"{session_prefix}_next"):
            st.session_state[idx_key] += 1
            st.rerun()
    return st.session_state[idx_key], nav_col2


def ai_analysis_block(api_key, summary_lines, total, button_key):
    """공통 AI 일괄 분석 UI + 호출"""
    if api_key and summary_lines:
        if st.button("전체 이미지 AI 일괄 분석 실행", key=button_key):
            try:
                client = genai.Client(api_key=api_key)
                joined_summary = "\n".join(summary_lines)
                prompt = (
                    f"총 {total}장의 케이블 이미지에 대한 검사 결과입니다.\n"
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
    elif not api_key:
        st.warning("사이드바에 Gemini API Key를 입력해주세요.")


# =========================================================
# 케이블 검출 (YOLO) 모델 로드
# =========================================================
@st.cache_resource
def load_yolo_model():
    model_path = Path("models/best.pt")
    if not model_path.exists():
        st.error("YOLO 모델 파일을 찾을 수 없습니다. (models/best.pt)")
        st.stop()
    return YOLO(model_path)


# =========================================================
# 케이블 분류 (CNN) 모델 정의 및 로드
# =========================================================
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(8 * 8 * 32, 32),
            nn.ReLU(),

            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)


CLS_LABEL_NAMES = ["정상", "불량"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_cls_model():
    model_path = Path("models/small_cnn_cable_normal_defect.pth")
    if not model_path.exists():
        st.error("분류 모델 파일을 찾을 수 없습니다. (models/small_cnn_cable_normal_defect.pth)")
        st.stop()
    m = SmallCNN().to(DEVICE)
    m.load_state_dict(torch.load(model_path, map_location=DEVICE))
    m.eval()
    return m


def predict_classification(cls_model, pil_image):
    """PIL(RGB) 이미지를 받아 학습 때와 동일한 BGR 순서로 변환 후 추론"""
    image_rgb = np.array(pil_image.convert("RGB"))
    image_bgr = image_rgb[:, :, ::-1]  # RGB -> BGR (학습 데이터와 채널 순서 일치)
    image_resized = cv2.resize(image_bgr, (64, 64)).astype(np.float32) / 255.0
    x = torch.tensor(image_resized.copy()).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = cls_model(x)
        probability = torch.softmax(output, dim=1)
        pred = probability.argmax(1).item()

    return {
        "label": CLS_LABEL_NAMES[pred],
        "prob_normal": probability[0, 0].item(),
        "prob_defect": probability[0, 1].item(),
    }


# =========================================================
# 사이드바: 모드 선택 + API 키
# =========================================================
st.sidebar.header("설정")
mode = st.sidebar.radio(
    "분석 방식 선택",
    ["케이블 검출 (Detection)", "케이블 분류 (Classification)"],
)
api_key = st.sidebar.text_input("Gemini API Key 입력", type="password")

st.sidebar.divider()
st.sidebar.caption(f"연산 장치: {DEVICE.upper()}")


# =========================================================
# 모드 1: 케이블 검출 (YOLO)
# =========================================================
if mode == "케이블 검출 (Detection)":
    st.header("케이블 검출 (Detection)")
    yolo_model = load_yolo_model()

    uploaded_files = st.file_uploader(
        "이미지 업로드 (여러 장 선택 가능)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="det_uploader",
    )

    if uploaded_files:
        file_key = tuple((f.name, f.size) for f in uploaded_files)
        if st.session_state.get("det_file_key") != file_key:
            processed = []
            names = yolo_model.names
            with st.spinner("이미지 추론 중..."):
                for f in uploaded_files:
                    image = Image.open(f)
                    results = yolo_model.predict(source=image)
                    detected_objects = [names[int(box.cls.item())] for box in results[0].boxes]
                    confidences = [float(box.conf.item()) for box in results[0].boxes]
                    processed.append({
                        "name": f.name,
                        "image": image,
                        "plot": results[0].plot(),
                        "detected_objects": detected_objects,
                        "confidences": confidences,
                    })
            st.session_state.det_processed = processed
            st.session_state.det_file_key = file_key
            st.session_state.det_idx = 0

        processed = st.session_state.det_processed
        total = len(processed)

        st.subheader("이미지 탐지 결과")
        idx, nav_col2 = render_carousel_nav("det", total)
        current = processed[idx]
        with nav_col2:
            st.markdown(
                f"<div style='text-align:center; font-weight:bold;'>{idx + 1} / {total} — {current['name']}</div>",
                unsafe_allow_html=True,
            )

        col1, col2 = st.columns(2)
        with col1:
            st.caption("원본")
            st.image(current["image"], use_container_width=True)
        with col2:
            st.caption("탐지 결과")
            st.image(current["plot"], use_container_width=True)

        if current["detected_objects"]:
            counts = Counter(current["detected_objects"])
            st.write(f"탐지 개수: **{len(current['detected_objects'])}개**")
            count_cols = st.columns(min(len(counts), 6))
            for col, (obj_name, cnt) in zip(count_cols, counts.items()):
                col.metric(obj_name, f"{cnt}개")
        else:
            st.warning("탐지된 객체 없음")

        st.divider()
        st.subheader("전체 이미지 결과 표")
        table_rows = []
        summary_lines = []
        for item in processed:
            counts = Counter(item["detected_objects"])
            avg_conf = (sum(item["confidences"]) / len(item["confidences"])
                        if item["confidences"] else 0)
            table_rows.append({
                "파일명": item["name"],
                "탐지 개수": len(item["detected_objects"]),
                "클래스별 개수": ", ".join(f"{k}:{v}" for k, v in counts.items()) if counts else "-",
                "평균 신뢰도": f"{avg_conf:.1%}" if item["confidences"] else "-",
            })
            summary = ", ".join(f"{k} {v}개" for k, v in counts.items()) if counts else "탐지된 객체 없음"
            summary_lines.append(f"[{item['name']}] {summary}")

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        st.divider()
        ai_analysis_block(api_key, summary_lines, total, button_key="det_ai_btn")


# =========================================================
# 모드 2: 케이블 분류 (CNN 정상/불량)
# =========================================================
else:
    st.header("케이블 분류 (Classification)")
    cls_model = load_cls_model()

    uploaded_files = st.file_uploader(
        "이미지 업로드 (여러 장 선택 가능)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="cls_uploader",
    )

    if uploaded_files:
        file_key = tuple((f.name, f.size) for f in uploaded_files)
        if st.session_state.get("cls_file_key") != file_key:
            processed = []
            with st.spinner("이미지 분류 중..."):
                for f in uploaded_files:
                    image = Image.open(f)
                    result = predict_classification(cls_model, image)
                    processed.append({
                        "name": f.name,
                        "image": image,
                        **result,
                    })
            st.session_state.cls_processed = processed
            st.session_state.cls_file_key = file_key
            st.session_state.cls_idx = 0

        processed = st.session_state.cls_processed
        total = len(processed)

        st.subheader("이미지 분류 결과")
        idx, nav_col2 = render_carousel_nav("cls", total)
        current = processed[idx]
        with nav_col2:
            st.markdown(
                f"<div style='text-align:center; font-weight:bold;'>{idx + 1} / {total} — {current['name']}</div>",
                unsafe_allow_html=True,
            )

        img_col, info_col = st.columns([1, 1])
        with img_col:
            st.image(current["image"], use_container_width=True)
        with info_col:
            if current["label"] == "불량":
                st.error(f"판정 결과: **{current['label']}**")
            else:
                st.success(f"판정 결과: **{current['label']}**")
            st.metric("정상 확률", f"{current['prob_normal']:.1%}")
            st.metric("불량 확률", f"{current['prob_defect']:.1%}")

        st.divider()
        st.subheader("전체 이미지 결과 표")
        table_rows = []
        summary_lines = []
        for item in processed:
            table_rows.append({
                "파일명": item["name"],
                "판정": item["label"],
                "정상 확률": f"{item['prob_normal']:.1%}",
                "불량 확률": f"{item['prob_defect']:.1%}",
            })
            summary_lines.append(
                f"[{item['name']}] 판정: {item['label']} (정상 {item['prob_normal']:.1%}, 불량 {item['prob_defect']:.1%})"
            )

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        st.divider()
        ai_analysis_block(api_key, summary_lines, total, button_key="cls_ai_btn")
