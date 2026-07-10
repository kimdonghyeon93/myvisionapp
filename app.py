import streamlit as st
import pandas as pd
import numpy as np
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
from google import genai
from collections import Counter
import time
import re
import zipfile
import shutil
import tempfile

# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(page_title="케이블 분류 · 검출 서비스", layout="wide")
st.title("케이블 분류 / 검출 서비스")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
YOLO_MODEL_PATH = MODELS_DIR / "best.pt"
CLS_MODEL_PATH = MODELS_DIR / "small_cnn_cable_normal_defect.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def imread(path, flags=cv2.IMREAD_COLOR):
    """한글 경로에서도 안전하게 이미지 읽기 (노트북과 동일)"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, flags)


def extract_zip_to_temp(uploaded_zip):
    tmp_dir = Path(tempfile.mkdtemp())
    zip_path = tmp_dir / "data.zip"
    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.getbuffer())
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmp_dir)
    return tmp_dir


def find_category_root(extracted_dir):
    """zip 안에서 MVTec 카테고리 구조(train/good, test 폴더)를 찾음."""
    candidates = [extracted_dir] + [p for p in extracted_dir.rglob("*") if p.is_dir()]
    for p in candidates:
        if (p / "train" / "good").exists() and (p / "test").exists():
            return p
    return None


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
    idx_key = f"{session_prefix}_idx"
    if idx_key not in st.session_state:
        st.session_state[idx_key] = 0
    nav_col1, nav_col2, nav_col3 = st.columns([1, 4, 1])
    with nav_col1:
        if st.button("◀ 이전", use_container_width=True,
                      disabled=(st.session_state[idx_key] == 0), key=f"{session_prefix}_prev"):
            st.session_state[idx_key] -= 1
            st.rerun()
    with nav_col3:
        if st.button("다음 ▶", use_container_width=True,
                      disabled=(st.session_state[idx_key] == total - 1), key=f"{session_prefix}_next"):
            st.session_state[idx_key] += 1
            st.rerun()
    return st.session_state[idx_key], nav_col2


def ai_analysis_block(api_key, summary_lines, total, button_key):
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
    if not YOLO_MODEL_PATH.exists():
        st.error(f"YOLO 모델 파일을 찾을 수 없습니다. ({YOLO_MODEL_PATH})")
        st.stop()
    return YOLO(YOLO_MODEL_PATH)


# =========================================================
# 케이블 분류 (CNN) 모델 정의 및 로드
# =========================================================
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(8 * 8 * 32, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)


CLS_LABEL_NAMES = ["정상", "불량"]


@st.cache_resource
def load_cls_model():
    if not CLS_MODEL_PATH.exists():
        st.error(f"분류 모델 파일을 찾을 수 없습니다. ({CLS_MODEL_PATH})")
        st.stop()
    m = SmallCNN().to(DEVICE)
    m.load_state_dict(torch.load(CLS_MODEL_PATH, map_location=DEVICE))
    m.eval()
    return m


def predict_classification(cls_model, pil_image):
    image_rgb = np.array(pil_image.convert("RGB"))
    image_bgr = image_rgb[:, :, ::-1]
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
# CNN 학습용 Dataset
# =========================================================
class CableImageDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = imread(path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (64, 64)).astype(np.float32) / 255.0
        return torch.tensor(image).permute(2, 0, 1), torch.tensor(label, dtype=torch.long)


# =========================================================
# YOLO 학습용: 마스크 -> 바운딩 박스 변환
# =========================================================
def bbox_from_mask(mask_path):
    mask = imread(mask_path, cv2.IMREAD_GRAYSCALE)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


def write_yolo_label(label_path, bbox, shape, class_id):
    h, w = shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
    bw, bh = (x2 - x1) / w, (y2 - y1) / h
    label_path.write_text(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")


# =========================================================
# 사이드바
# =========================================================
st.sidebar.header("설정")
mode = st.sidebar.radio(
    "메뉴 선택",
    ["케이블 검출 (Detection)", "케이블 분류 (Classification)", "모델 학습 (Training)"],
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
        "이미지 업로드 (여러 장 선택 가능)", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True, key="det_uploader",
    )

    if uploaded_files:
        file_key = tuple((f.name, f.size) for f in uploaded_files)
        if st.session_state.get("det_file_key") != file_key:
            processed = []
            names = yolo_model.names
            with st.spinner("이미지 추론 중..."):
                for f in uploaded_files:
                    image = Image.open(f)
                    results = yolo_model.predict(source=image, conf=0.05)
                    detected_objects = [names[int(box.cls.item())] for box in results[0].boxes]
                    confidences = [float(box.conf.item()) for box in results[0].boxes]
                    processed.append({
                        "name": f.name, "image": image, "plot": results[0].plot(),
                        "detected_objects": detected_objects, "confidences": confidences,
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
            st.markdown(f"<div style='text-align:center; font-weight:bold;'>{idx + 1} / {total} — {current['name']}</div>", unsafe_allow_html=True)

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
        table_rows, summary_lines = [], []
        for item in processed:
            counts = Counter(item["detected_objects"])
            avg_conf = sum(item["confidences"]) / len(item["confidences"]) if item["confidences"] else 0
            table_rows.append({
                "파일명": item["name"], "탐지 개수": len(item["detected_objects"]),
                "클래스별 개수": ", ".join(f"{k}:{v}" for k, v in counts.items()) if counts else "-",
                "평균 신뢰도": f"{avg_conf:.1%}" if item["confidences"] else "-",
            })
            summary = ", ".join(f"{k} {v}개" for k, v in counts.items()) if counts else "탐지된 객체 없음"
            summary_lines.append(f"[{item['name']}] {summary}")

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
        st.divider()
        ai_analysis_block(api_key, summary_lines, total, button_key="det_ai_btn")


# =========================================================
# 모드 2: 케이블 분류 (CNN)
# =========================================================
elif mode == "케이블 분류 (Classification)":
    st.header("케이블 분류 (Classification)")
    cls_model = load_cls_model()

    uploaded_files = st.file_uploader(
        "이미지 업로드 (여러 장 선택 가능)", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True, key="cls_uploader",
    )

    if uploaded_files:
        file_key = tuple((f.name, f.size) for f in uploaded_files)
        if st.session_state.get("cls_file_key") != file_key:
            processed = []
            with st.spinner("이미지 분류 중..."):
                for f in uploaded_files:
                    image = Image.open(f)
                    result = predict_classification(cls_model, image)
                    processed.append({"name": f.name, "image": image, **result})
            st.session_state.cls_processed = processed
            st.session_state.cls_file_key = file_key
            st.session_state.cls_idx = 0

        processed = st.session_state.cls_processed
        total = len(processed)

        st.subheader("이미지 분류 결과")
        idx, nav_col2 = render_carousel_nav("cls", total)
        current = processed[idx]
        with nav_col2:
            st.markdown(f"<div style='text-align:center; font-weight:bold;'>{idx + 1} / {total} — {current['name']}</div>", unsafe_allow_html=True)

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
        table_rows, summary_lines = [], []
        for item in processed:
            table_rows.append({
                "파일명": item["name"], "판정": item["label"],
                "정상 확률": f"{item['prob_normal']:.1%}", "불량 확률": f"{item['prob_defect']:.1%}",
            })
            summary_lines.append(f"[{item['name']}] 판정: {item['label']} (정상 {item['prob_normal']:.1%}, 불량 {item['prob_defect']:.1%})")

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
        st.divider()
        ai_analysis_block(api_key, summary_lines, total, button_key="cls_ai_btn")


# =========================================================
# 모드 3: 모델 학습 (Training)
# =========================================================
else:
    st.header("모델 학습 (Training)")
    train_target = st.radio("학습할 모델 선택", ["케이블 분류 모델 (CNN)", "케이블 검출 모델 (YOLO)"])

    # ---------------------------------------------------
    # CNN 분류 모델 학습
    # ---------------------------------------------------
    if train_target == "케이블 분류 모델 (CNN)":
        st.subheader("케이블 분류 모델 학습 (CNN)")
        st.markdown(
            "**zip 파일 구조 (MVTec `cable` 카테고리 폴더 그대로):**\n"
            "```\n"
            "cable.zip\n"
            "└── cable/\n"
            "    ├── train/good/*.png\n"
            "    └── test/\n"
            "        ├── good/*.png\n"
            "        ├── bent_wire/*.png\n"
            "        └── ... (기타 결함 폴더)\n"
            "```"
        )

        uploaded_zip = st.file_uploader("cable 카테고리 zip 업로드", type=["zip"], key="cls_train_zip")

        col1, col2, col3 = st.columns(3)
        with col1:
            epochs = st.number_input("Epochs", min_value=1, max_value=200, value=30)
        with col2:
            batch_size = st.number_input("Batch size", min_value=1, max_value=128, value=16)
        with col3:
            defect_limit = st.number_input("결함 폴더당 최대 이미지 수", min_value=5, max_value=200, value=20)

        if uploaded_zip is not None and st.button("분류 모델 학습 시작"):
            extracted_dir = extract_zip_to_temp(uploaded_zip)
            category = find_category_root(extracted_dir)

            if category is None:
                st.error("zip 안에서 train/good, test 폴더 구조를 찾지 못했습니다.")
            else:
                normal_paths = sorted((category / "train" / "good").glob("*.png"))
                normal_paths += sorted((category / "test" / "good").glob("*.png"))

                defect_paths = []
                for defect_dir in sorted((category / "test").iterdir()):
                    if defect_dir.is_dir() and defect_dir.name != "good":
                        defect_paths += sorted(defect_dir.glob("*.png"))[:defect_limit]

                if not normal_paths or not defect_paths:
                    st.error("정상 또는 불량 이미지를 찾지 못했습니다.")
                else:
                    st.write(f"정상 이미지: {len(normal_paths)}장 / 불량 이미지: {len(defect_paths)}장")

                    samples = [(p, 0) for p in normal_paths] + [(p, 1) for p in defect_paths]
                    train_samples, test_samples = train_test_split(
                        samples, test_size=0.25, random_state=42,
                        stratify=[label for _, label in samples],
                    )
                    st.write(f"train: {len(train_samples)}장 / test: {len(test_samples)}장")

                    train_loader = DataLoader(CableImageDataset(train_samples), batch_size=batch_size, shuffle=True)
                    test_loader = DataLoader(CableImageDataset(test_samples), batch_size=32)

                    cnn = SmallCNN().to(DEVICE)
                    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
                    criterion = nn.CrossEntropyLoss()

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    chart_placeholder = st.empty()
                    loss_history = []
                    
                    for epoch in range(epochs):
                        cnn.train()
                        losses = []
                        for x, y in train_loader:
                            x, y = x.to(DEVICE), y.to(DEVICE)
                            optimizer.zero_grad()
                            loss = criterion(cnn(x), y)
                            loss.backward()
                            optimizer.step()
                            losses.append(loss.item())
                    
                        avg_loss = float(np.mean(losses))
                        loss_history.append(avg_loss)
                    
                        progress_bar.progress((epoch + 1) / epochs)
                        status_text.text(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.5f}")
                        chart_placeholder.line_chart(pd.DataFrame({"loss": loss_history}))

                    # 평가
                    cnn.eval()
                    correct = total = tp = fp = fn = 0
                    with torch.no_grad():
                        for x, y in test_loader:
                            pred = cnn(x.to(DEVICE)).argmax(1).cpu()
                            correct += (pred == y).sum().item()
                            total += len(y)
                            tp += ((pred == 1) & (y == 1)).sum().item()
                            fp += ((pred == 1) & (y == 0)).sum().item()
                            fn += ((pred == 0) & (y == 1)).sum().item()

                    accuracy = correct / total
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

                    st.success("학습 완료!")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Accuracy", f"{accuracy:.1%}")
                    m2.metric("Precision", f"{precision:.1%}")
                    m3.metric("Recall", f"{recall:.1%}")
                    m4.metric("F1 Score", f"{f1:.1%}")

                    torch.save(cnn.state_dict(), CLS_MODEL_PATH)
                    st.info(f"모델이 저장되었습니다: {CLS_MODEL_PATH}")
                    load_cls_model.clear()

                    with open(CLS_MODEL_PATH, "rb") as f:
                        st.download_button("학습된 모델 다운로드 (.pth)", f, file_name=CLS_MODEL_PATH.name)

            shutil.rmtree(extracted_dir, ignore_errors=True)

    # ---------------------------------------------------
    # YOLO 검출 모델 학습
    # ---------------------------------------------------
    else:
        st.subheader("케이블 검출 모델 학습 (YOLO)")
        st.warning(
            "YOLO 학습은 CPU 환경에서 오래 걸리고, Streamlit Cloud에서는 타임아웃/메모리 부족으로 "
            "실패할 수 있습니다. 가능하면 로컬 GPU 환경에서 학습 후 best.pt만 업로드하는 것을 권장합니다."
        )
        st.markdown(
            "**zip 파일 구조 (MVTec `cable` 카테고리, ground_truth 마스크 포함):**\n"
            "```\n"
            "cable.zip\n"
            "└── cable/\n"
            "    ├── test/\n"
            "    │   ├── good/*.png\n"
            "    │   └── bent_wire/*.png ...\n"
            "    └── ground_truth/\n"
            "        └── bent_wire/*_mask.png ...\n"
            "```"
        )

        uploaded_zip = st.file_uploader("cable 카테고리 zip 업로드 (ground_truth 포함)", type=["zip"], key="det_train_zip")

        col1, col2, col3 = st.columns(3)
        with col1:
            yolo_epochs = st.number_input("Epochs", min_value=1, max_value=300, value=50)
        with col2:
            imgsz = st.selectbox("이미지 크기", [320, 480, 640, 960], index=0)
        with col3:
            per_defect_limit = st.number_input("결함 유형당 최대 이미지 수", min_value=5, max_value=200, value=30)

        if uploaded_zip is not None and st.button("검출 모델 학습 시작"):
            extracted_dir = extract_zip_to_temp(uploaded_zip)
            category = None
            for p in [extracted_dir] + [d for d in extracted_dir.rglob("*") if d.is_dir()]:
                if (p / "test").exists() and (p / "ground_truth").exists():
                    category = p
                    break

            if category is None:
                st.error("zip 안에서 test/, ground_truth/ 폴더 구조를 찾지 못했습니다.")
            else:
                defect_dirs = sorted([p for p in (category / "test").iterdir() if p.is_dir() and p.name != "good"])
                class_names = [p.name for p in defect_dirs]
                class_to_id = {name: idx for idx, name in enumerate(class_names)}

                if not class_names:
                    st.error("test/ 폴더 안에서 결함 유형 폴더를 찾지 못했습니다.")
                else:
                    st.write("탐지 클래스:", class_names)

                    work = Path(tempfile.mkdtemp()) / "yolo_dataset"
                    for split in ["train", "val"]:
                        (work / "images" / split).mkdir(parents=True, exist_ok=True)
                        (work / "labels" / split).mkdir(parents=True, exist_ok=True)

                    i = 0
                    with st.spinner("마스크 -> 바운딩 박스 변환 중..."):
                        for defect_dir in defect_dirs:
                            for image_path in sorted(defect_dir.glob("*.png"))[:per_defect_limit]:
                                mask_path = category / "ground_truth" / defect_dir.name / f"{image_path.stem}_mask.png"
                                if not mask_path.exists():
                                    continue
                                bbox = bbox_from_mask(mask_path)
                                if bbox is None:
                                    continue

                                split = "val" if i % 5 == 0 else "train"
                                target_image = work / "images" / split / f"{i:04d}.png"
                                target_label = work / "labels" / split / f"{i:04d}.txt"
                                shutil.copy2(image_path, target_image)
                                write_yolo_label(target_label, bbox, imread(image_path).shape, class_to_id[defect_dir.name])
                                i += 1

                    if i == 0:
                        st.error("변환된 이미지가 없습니다. 파일명 규칙(_mask.png)을 확인해주세요.")
                    else:
                        yaml_lines = [
                            f"path: {work.resolve().as_posix()}",
                            "train: images/train",
                            "val: images/val",
                            "names:",
                        ]
                        for class_id, class_name in enumerate(class_names):
                            yaml_lines.append(f"  {class_id}: {class_name}")

                        yaml_path = work / "data.yaml"
                        yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
                        st.write(f"총 {i}장 변환 완료. data.yaml 생성됨.")

                        st.info("학습을 시작합니다...")
                        with st.spinner("YOLO 학습 중..."):
                            train_yolo = YOLO("yolov8n.pt")
                            train_result = train_yolo.train(
                                data=str(yaml_path),
                                epochs=int(yolo_epochs),
                                imgsz=int(imgsz),
                                batch=8,
                                project=str(tempfile.mkdtemp()),
                                name="cable_yolo_train",
                            )

                        best_pt_path = Path(train_result.save_dir) / "weights" / "best.pt"

                        if best_pt_path.exists():
                            st.success("학습 완료!")
                            shutil.copy(best_pt_path, YOLO_MODEL_PATH)
                            st.info(f"모델이 저장되었습니다: {YOLO_MODEL_PATH}")
                            load_yolo_model.clear()

                            with open(best_pt_path, "rb") as f:
                                st.download_button("학습된 모델 다운로드 (.pt)", f, file_name="best.pt")
                        else:
                            st.error("학습은 완료됐지만 best.pt 파일을 찾지 못했습니다.")

            shutil.rmtree(extracted_dir, ignore_errors=True)
