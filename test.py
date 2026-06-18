import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# KONFIGURACJA
# ==========================================
USE_ADDED_DATA = True  # Zmień na False, jeśli chcesz testować tylko na głównym datasecie
# ==========================================

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Ładowanie bazowego zbioru
base_val_dataset = datasets.ImageFolder("dataset/val", transform=val_transform)
print("Klasy:", base_val_dataset.class_to_idx)
final_val_dataset = base_val_dataset

# Opcjonalne dołączanie własnych danych do testów
if USE_ADDED_DATA:
    try:
        added_val_dataset = datasets.ImageFolder("added data/val", transform=val_transform)
        final_val_dataset = ConcatDataset([base_val_dataset, added_val_dataset])
        print("Pomyślnie dodano własne zdjęcia ('added data/val') do zbioru testowego.")
    except Exception as e:
        print(f"Ostrzeżenie: Nie udało się załadować 'added data/val'. Błąd: {e}")

val_loader = DataLoader(final_val_dataset, batch_size=32, shuffle=False)
print(f"Łączna liczba zdjęć testowych: {len(final_val_dataset)}")

model = torchvision.models.mobilenet_v2()

num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2)

model.load_state_dict(
    torch.load("spaghetti_model.pth", map_location="cpu")
)

model.eval()

y_true = []
y_pred = []

print("Trwa testowanie modelu...")
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n===== WYNIKI =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\n===== CLASSIFICATION REPORT =====")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=base_val_dataset.classes
    )
)

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_true, y_pred))