from fastapi import FastAPI, UploadFile, File
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Spaghetti Detection API")
ort_session = ort.InferenceSession("spaghetti_model.onnx")


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_np = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np - mean) / std
    img_np = img_np.transpose(2, 0, 1)  # HWC to CHW
    return np.expand_dims(img_np, axis=0).astype(np.float32)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    input_tensor = preprocess_image(contents)

    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    ort_outs = ort_session.run(None, ort_inputs)

    pred = int(np.argmax(ort_outs[0]))
    classes = ["Dobre", "Spaghetti (Blad!)"]

    return {"status": "success", "prediction": classes[pred], "class_id": pred}