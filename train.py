import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import ConcatDataset, DataLoader

# ==========================================
# KONFIGURACJA
# ==========================================
USE_ADDED_DATA = True  # Zmień na False, jeśli chcesz trenować tylko na głównym datasecie
# ==========================================

print("Rozpoczynam przygotowanie danych...")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Ładowanie bazowego zbioru
base_dataset = datasets.ImageFolder('dataset/train', transform=train_transform)
print(f"Znaleziono klasy w bazowym zbiorze: {base_dataset.class_to_idx}")
final_dataset = base_dataset

# Opcjonalne dołączanie własnych danych
if USE_ADDED_DATA:
    try:
        added_dataset = datasets.ImageFolder('added data/train', transform=train_transform)
        # Złączamy oba zbiory w jeden
        final_dataset = ConcatDataset([base_dataset, added_dataset])
        print("Pomyślnie dodano własne zdjęcia ('added data/train') do zbioru treningowego.")
    except Exception as e:
        print(f"Ostrzeżenie: Nie udało się załadować 'added data/train'. Błąd: {e}")

train_loader = DataLoader(final_dataset, batch_size=32, shuffle=True)
print(f"Łączna liczba zdjęć treningowych: {len(final_dataset)}")

model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)

for name, param in model.named_parameters():
    if "features.18" not in name and "classifier" not in name:
        param.requires_grad = False

num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

print("Rozpoczynam trening (10 epok)...")
model.train()
for epoch in range(10):
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoka [{epoch + 1}/10], Strata: {running_loss / len(train_loader):.4f}, Dokładność: {accuracy:.2f}%")

torch.save(model.state_dict(), 'spaghetti_model.pth')
print("Trening zakończony! Model zapisany.")