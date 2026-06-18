import torch
import torchvision
import torch.nn as nn
import onnxruntime as ort
import numpy as np
import time

# 1. Odtworzenie struktury modelu i załadowanie wag
model = torchvision.models.mobilenet_v2()
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('spaghetti_model.pth'))
model.eval()

# 2. Eksport do formatu ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "spaghetti_model.onnx",
                  input_names=['input'], output_names=['output'])
print("Model wyeksportowany do ONNX!")

# 3. Test zgodności predykcji
with torch.no_grad():
    torch_out = model(dummy_input).numpy()

ort_session = ort.InferenceSession("spaghetti_model.onnx")
ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
ort_out = ort_session.run(None, ort_inputs)[0]

# Sprawdzenie czy wyniki są identyczne
np.testing.assert_allclose(torch_out, ort_out, rtol=1e-03, atol=1e-05)
print("Predykcje PyTorch i ONNX są zgodne!")

# 4. Pomiar czasu inferencji (CPU)
num_tests = 100

start = time.time()
for _ in range(num_tests):
    with torch.no_grad():
        _ = model(dummy_input)
pytorch_time = (time.time() - start) / num_tests

start = time.time()
for _ in range(num_tests):
    _ = ort_session.run(None, ort_inputs)
onnx_time = (time.time() - start) / num_tests

print(f"Średni czas PyTorch CPU: {pytorch_time*1000:.2f} ms")
print(f"Średni czas ONNX Runtime CPU: {onnx_time*1000:.2f} ms")