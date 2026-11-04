"""阶段 6：FastAPI 预测服务。

启动（项目根目录）:
  uvicorn lessons.stage6.app:app --reload --host 127.0.0.1 --port 8000

文档页: http://127.0.0.1:8000/docs
"""

from pathlib import Path
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("artifacts/breast_cancer_pipeline.joblib")

app = FastAPI(title="Breast Cancer Predictor", version="0.1.0")


class PredictRequest(BaseModel):
    """按训练时的特征顺序传入 30 个浮点数。"""

    features: List[float] = Field(..., min_length=30, max_length=30)


class PredictResponse(BaseModel):
    label: str
    label_id: int
    probability_benign: float


def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"未找到模型 {MODEL_PATH}。请先运行: python lessons/stage5/01_full_project.py"
        )
    return joblib.load(MODEL_PATH)


@app.on_event("startup")
def startup():
    app.state.bundle = load_bundle()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.get("/features")
def features():
    """返回训练时特征名顺序，便于构造请求。"""
    return {"feature_names": app.state.bundle["feature_names"]}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    bundle = app.state.bundle
    names = bundle["feature_names"]
    if len(body.features) != len(names):
        raise HTTPException(status_code=400, detail=f"需要 {len(names)} 个特征")

    x = np.array(body.features, dtype=float).reshape(1, -1)
    pipe = bundle["pipeline"]
    label_id = int(pipe.predict(x)[0])
    proba = pipe.predict_proba(x)[0]
    # target: 0=malignant, 1=benign
    return PredictResponse(
        label=bundle["target_names"][label_id],
        label_id=label_id,
        probability_benign=float(proba[1]),
    )
