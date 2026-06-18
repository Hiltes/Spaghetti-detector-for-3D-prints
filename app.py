from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Spaghetti Detection API")
ort_session = ort.InferenceSession("spaghetti_model.onnx")


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Używamy kadrowania (Center Crop), które omówiliśmy wcześniej,
    # żeby proporcje z telefonu nie psuły wyników!
    width, height = img.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    img = img.crop((left, top, right, bottom))

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
    classes = ["Dobre", "Spaghetti (Błąd!)"]

    return {"status": "success", "prediction": classes[pred], "class_id": pred}


# ==========================================
# NOWOŚĆ: Piękny interfejs graficzny
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def main_page():
    html_content = """
    <!DOCTYPE html>
    <html lang="pl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Spaghetti Detector</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            .drag-over { border-color: #3b82f6; background-color: rgba(59, 130, 246, 0.1); }
            body { background-color: #0f172a; color: #f8fafc; font-family: ui-sans-serif, system-ui, sans-serif; }
        </style>
    </head>
    <body class="min-h-screen flex flex-col items-center justify-center p-4">

        <div class="max-w-md w-full bg-slate-800 rounded-2xl shadow-2xl p-8 border border-slate-700 text-center">
            <h1 class="text-3xl font-bold mb-2 text-white">Spaghetti <span class="text-blue-500">Detector</span></h1>
            <p class="text-slate-400 mb-8 text-sm">Wykrywanie anomalii w druku 3D (AI ONNX Model)</p>

            <!-- Strefa Drag & Drop -->
            <div id="dropzone" class="border-2 border-dashed border-slate-500 rounded-xl p-8 mb-6 transition-all duration-300 cursor-pointer hover:border-blue-400">
                <svg class="mx-auto h-12 w-12 text-slate-400 mb-4" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
                    <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
                </svg>
                <p class="text-sm text-slate-300">Przeciągnij zdjęcie tutaj lub <span class="text-blue-400 font-semibold">kliknij, aby wybrać</span></p>
                <input type="file" id="fileInput" class="hidden" accept="image/jpeg, image/png, image/jpg">
            </div>

            <!-- Podgląd zdjęcia i wynik -->
            <div id="resultContainer" class="hidden flex-col items-center">
                <img id="imagePreview" class="w-48 h-48 object-cover rounded-lg shadow-md mb-4 border-2 border-slate-600">

                <div id="loading" class="hidden text-blue-400 animate-pulse font-semibold mb-4">Analizowanie obrazu...</div>

                <div id="predictionResult" class="text-2xl font-bold px-6 py-2 rounded-full hidden"></div>

                <button id="resetBtn" class="mt-6 text-sm text-slate-400 hover:text-white underline transition-colors">Sprawdź kolejne zdjęcie</button>
            </div>
        </div>

        <script>
            const dropzone = document.getElementById('dropzone');
            const fileInput = document.getElementById('fileInput');
            const resultContainer = document.getElementById('resultContainer');
            const imagePreview = document.getElementById('imagePreview');
            const loading = document.getElementById('loading');
            const predictionResult = document.getElementById('predictionResult');
            const resetBtn = document.getElementById('resetBtn');

            // Otwieranie dialogu pliku po kliknięciu
            dropzone.addEventListener('click', () => fileInput.click());

            // Efekty Drag & Drop
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropzone.addEventListener(eventName, preventDefaults, false);
            });

            function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

            ['dragenter', 'dragover'].forEach(eventName => {
                dropzone.addEventListener(eventName, () => dropzone.classList.add('drag-over'), false);
            });

            ['dragleave', 'drop'].forEach(eventName => {
                dropzone.addEventListener(eventName, () => dropzone.classList.remove('drag-over'), false);
            });

            // Obsługa upuszczenia pliku
            dropzone.addEventListener('drop', (e) => {
                const files = e.dataTransfer.files;
                if(files.length) handleFile(files[0]);
            });

            // Obsługa wybrania pliku klasycznie
            fileInput.addEventListener('change', function() {
                if(this.files.length) handleFile(this.files[0]);
            });

            function handleFile(file) {
                if (!file.type.startsWith('image/')) return;

                // Pokazanie podglądu
                const reader = new FileReader();
                reader.readAsDataURL(file);
                reader.onload = () => {
                    imagePreview.src = reader.result;
                    dropzone.classList.add('hidden');
                    resultContainer.classList.remove('hidden');
                    resultContainer.classList.add('flex');
                    loading.classList.remove('hidden');
                    predictionResult.classList.add('hidden');

                    uploadAndPredict(file);
                }
            }

            async function uploadAndPredict(file) {
                const formData = new FormData();
                formData.append('file', file);

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();

                    loading.classList.add('hidden');
                    predictionResult.classList.remove('hidden', 'bg-green-500/20', 'text-green-400', 'border-green-500', 'bg-red-500/20', 'text-red-400', 'border-red-500', 'border-2');

                    predictionResult.textContent = data.prediction;

                    // Kolorowanie w zależności od wyniku
                    if(data.class_id === 0) {
                        predictionResult.classList.add('bg-green-500/20', 'text-green-400', 'border-2', 'border-green-500');
                    } else {
                        predictionResult.classList.add('bg-red-500/20', 'text-red-400', 'border-2', 'border-red-500');
                    }

                } catch (error) {
                    loading.classList.add('hidden');
                    predictionResult.textContent = "Błąd połączenia z serwerem";
                    predictionResult.classList.remove('hidden');
                }
            }

            resetBtn.addEventListener('click', () => {
                fileInput.value = '';
                resultContainer.classList.add('hidden');
                resultContainer.classList.remove('flex');
                dropzone.classList.remove('hidden');
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)