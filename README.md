![Prezentacja Gui w formacie .gif](Prezentacja.gif)

# Spaghetti Detector for 3D Prints

## Opis projektu

Projekt przedstawia proces przygotowania, wytrenowania oraz wdrożenia modelu klasyfikacji obrazów służącego do wykrywania błędów typu **"spaghetti"** występujących podczas drukowania 3D.

W przypadku nieudanego wydruku głowica drukarki kontynuuje pracę pomimo braku poprawnie budowanego modelu, co prowadzi do powstawania charakterystycznych splątanych struktur przypominających makaron (*spaghetti*).

Celem projektu było stworzenie systemu, który automatycznie rozpoznaje takie sytuacje na podstawie zdjęć.

Wykorzystane dane:
https://www.kaggle.com/datasets/justin900429/3d-printer-defected-dataset/data

---

# Zakres zrealizowanych wymagań

## 1. Wykorzystanie gotowego modelu PyTorch

W projekcie wykorzystano model:

* MobileNetV2 (PyTorch)

Model został załadowany z wagami wytrenowanymi na zbiorze ImageNet, a następnie dostosowany do klasyfikacji dwóch klas:

* `good` – poprawny wydruk
* `spaghetti` – błędny wydruk

---

## 2. Transfer Learning

Zastosowano technikę transfer learning.

Większość warstw sieci została zamrożona, a ponownemu uczeniu poddano jedynie końcowe warstwy odpowiedzialne za klasyfikację.

Pozwoliło to na wykorzystanie wiedzy zdobytej podczas treningu na ImageNet oraz dostosowanie modelu do nowego problemu przy stosunkowo niewielkim zbiorze danych.

---

## 3. Przygotowanie zbioru danych

Struktura zbioru danych:

```text
dataset/
├── train/
│   ├── good/
│   └── spaghetti/
└── val/
    ├── good/
    └── spaghetti/
```

Dane zostały przygotowane w formacie zgodnym z `torchvision.datasets.ImageFolder`.

Podczas treningu wykorzystano augmentację danych:

* Resize 224×224
* RandomHorizontalFlip
* RandomRotation
* Normalizacja zgodna z ImageNet

---

## 4. Eksport modelu do ONNX

Po zakończeniu treningu model został wyeksportowany do formatu ONNX.

Format ONNX umożliwia uruchamianie modelu niezależnie od frameworka treningowego oraz wykorzystanie ONNX Runtime do szybszej inferencji.

Wygenerowane pliki:

* `spaghetti_model.onnx`
* `spaghetti_model.onnx.data`

---

## 5. Weryfikacja zgodności predykcji

W celu sprawdzenia poprawności eksportu wykonano porównanie wyników:

* PyTorch
* ONNX Runtime

Do walidacji wykorzystano:

```python
np.testing.assert_allclose(...)
```

Wyniki obu implementacji były zgodne w granicach przyjętej tolerancji numerycznej.

---

## 6. Benchmark wydajności

Zmierzono średni czas inferencji dla:

* PyTorch CPU
* ONNX Runtime CPU

| Środowisko       | Średni czas |
| ---------------- | ----------- |
| PyTorch CPU      | 13.83 ms    |
| ONNX Runtime CPU | 2.57 ms     |

ONNX Runtime okazał się ponad 5 razy szybszy od natywnego wykonania modelu w PyTorch na CPU.

---

## 7. API

Przygotowano aplikację wykorzystującą FastAPI.

Endpoint:

```http
POST /predict
```

Przyjmuje obraz i zwraca:

```json
{
  "status": "success",
  "prediction": "Dobre",
  "class_id": 0
}
```

Model wykorzystywany przez API działa w oparciu o ONNX Runtime.

---

## 8. Konteneryzacja

Aplikacja została przygotowana do uruchamiania w środowisku Docker.

Budowanie obrazu:

```bash
docker build -t spaghetti-detector .
```

Uruchomienie kontenera:

```bash
docker run -p 8000:8000 spaghetti-detector
```

Po uruchomieniu API dostępne jest pod adresem:

```text
http://localhost:8000
```

Dokumentacja Swagger:

```text
http://localhost:8000/docs
```

---

# Uruchomienie projektu

## Trening modelu

```bash
python train.py
```

## Eksport ONNX i benchmark

```bash
python export_and_benchmark.py
```

## Uruchomienie API lokalnie

```bash
uvicorn app:app --reload
```

---

# Ewaluacja i interpretacja wyników

W celu precyzyjnego zweryfikowania skuteczności modelu wdrożono skrypt `test.py`, który analizuje m.in.:

* Accuracy (Dokładność)
* Precision (Precyzja)
* Recall (Czułość)
* F1-score
* Confusion Matrix (Macierz pomyłek)

---

## 1. Test na pierwotnej bazie danych (Kaggle)

### Wyniki

| Metryka               | Wartość |
| --------------------- | ------- |
| Accuracy              | 0.88    |
| Precision (Spaghetti) | 0.75    |
| Recall (Spaghetti)    | 0.60    |

### Confusion Matrix

| Rzeczywista \ Predykcja | good | spaghetti |
| ----------------------- | ---- | --------- |
| good                    | 38   | 2         |
| spaghetti               | 4    | 6         |

### Interpretacja

* 38 poprawnych wydruków zostało sklasyfikowanych poprawnie.
* 2 poprawne wydruki zostały błędnie oznaczone jako spaghetti.
* 6 przypadków spaghetti wykryto poprawnie.
* 4 przypadki spaghetti zostały przeoczone.

W warunkach laboratoryjnych model osiągnął wysoką skuteczność klasyfikacji oraz niewielką liczbę fałszywych alarmów.

---

## 2. Identyfikacja problemu w środowisku rzeczywistym (Domain Shift)

W celu sprawdzenia działania modelu w rzeczywistych warunkach wykonano dodatkowe zdjęcia własnej drukarki 3D.

Nowe fotografie różniły się od danych treningowych:

* innym oświetleniem,
* innym tłem,
* innym kątem wykonania zdjęć,
* innym urządzeniem rejestrującym obraz.

### Wyniki po dodaniu własnych danych testowych

| Metryka               | Wartość |
| --------------------- | ------- |
| Accuracy              | 0.79    |
| Precision (Spaghetti) | 0.54    |
| Recall (Spaghetti)    | 0.54    |

### Confusion Matrix

| Rzeczywista \ Predykcja | good | spaghetti |
| ----------------------- | ---- | --------- |
| good                    | 39   | 6         |
| spaghetti               | 6    | 7         |

### Analiza problemu

Zaobserwowany spadek jakości jest przykładem zjawiska **Domain Shift** (przesunięcia domeny danych).

Model nauczył się poprawnie rozpoznawać obrazy pochodzące ze zbioru Kaggle, jednak napotkał trudności przy analizie zdjęć wykonanych w rzeczywistym środowisku pracy drukarki.

Największym problemem okazał się wzrost liczby fałszywych alarmów oraz błędnych klasyfikacji spowodowanych odmiennym otoczeniem i warunkami fotografowania.

---

# Wniosek końcowy i dalszy rozwój projektu

Projekt potwierdził jedną twierdzenie że:

> Model jest tak dobry, jak dane, na których został wytrenowany.

W celu zwiększenia skuteczności modelu zaimplementowano możliwość rozszerzania zbioru danych o własne fotografie wykonane bezpośrednio w miejscu docelowego wdrożenia.

Przygotowana architektura pozwala na szybkie dostosowanie modelu do konkretnej drukarki 3D i warunków pracy użytkownika.

---

# Autor

**Dariusz Szymanek**
