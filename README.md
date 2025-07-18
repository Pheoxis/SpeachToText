# Projekt Rozpoznawania Mowy - Speech Recognition Model

## 📝 Opis projektu

Ten projekt implementuje model głębokiego uczenia do rozpoznawania mowy w języku polskim. Model wykorzystuje architekturę opartą na transformerach z kwantyzacją wektorową (RVQ) do konwersji sygnału audio na tekst.

## 🚀 Funkcjonalności

- **Rozpoznawanie mowy**: Konwersja plików audio na tekst
- **Trenowanie modelu**: Możliwość trenowania własnego modelu na danych TIMIT
- **Ewaluacja**: Obliczanie metryk WER (Word Error Rate) i CER (Character Error Rate)
- **Augmentacja danych**: Wbudowane techniki augmentacji audio
- **Monitoring**: Integracja z TensorBoard do śledzenia procesu trenowania

## 🛠️ Technologie

- **Python 3.10+**
- **PyTorch** - framework do głębokiego uczenia
- **torchaudio** - przetwarzanie sygnałów audio
- **transformers** - architektura transformerów
- **datasets** - zarządzanie zbiorami danych
- **tokenizers** - tokenizacja tekstu
- **jiwer** - obliczanie metryk WER/CER
- **TensorBoard** - wizualizacja procesu trenowania

## 📦 Instalacja

1. **Sklonuj repozytorium:**
```bash
git clone https://github.com/twoj-uzytkownik/speech-recognition.git
cd speech-recognition
```

2. **Zainstaluj wymagane pakiety:**
```bash
pip install torch torchaudio transformers datasets tokenizers jiwer tensorboard sounddevice librosa matplotlib seaborn pandas tqdm
```

3. **Sprawdź dostępność CUDA (opcjonalnie):**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 🎯 Jak uruchomić

### Trenowanie modelu

1. **Uruchom notebook treningowy:**
```bash
jupyter notebook traiin.ipynb
```

2. **Lub użyj skryptu Python:**
```bash
python train.py
```

### Ewaluacja modelu

```bash
python evaluate.py --model_id test21 --num_examples 100
```

### Transkrypcja pojedynczego pliku audio

```bash
python evaluate.py --audio_file sciezka/do/pliku.wav --model_path models/test21/model_final.pth
```

## 📁 Struktura projektu

```
speech-recognition/
├── dataset.py              # Zarządzanie danymi i tokenizacja
├── transcribe_model.py     # Główna architektura modelu
├── downsampling.py         # Sieć downsamplingu audio
├── rvq.py                  # Residual Vector Quantization
├── self_attention.py       # Warstwy transformera
├── augmentation.py         # Augmentacja danych audio
├── train.py               # Skrypt trenowania
├── traiin.ipynb           # Notebook treningowy
├── evaluate.py            # Ewaluacja i transkrypcja
├── models/                # Zapisane modele
└── runs/                  # Logi TensorBoard
```

## ⚙️ Konfiguracja modelu

Model można skonfigurować poprzez następujące parametry:

```python
model = TranscribeModel(
    num_codebooks=4,           # Liczba kodebooków RVQ
    codebook_size=64,          # Rozmiar każdego codebooka
    embedding_dim=256,         # Wymiar embeddingów
    num_transformer_layers=6,  # Liczba warstw transformera
    vocab_size=len(tokenizer.get_vocab()),
    strides=[8, 8, 4],        # Kroki downsamplingu
    initial_mean_pooling_kernel_size=2,
    max_seq_length=400,       # Maksymalna długość sekwencji
)
```

## 📊 Metryki

Model jest ewaluowany przy użyciu:
- **WER (Word Error Rate)** - błąd na poziomie słów
- **CER (Character Error Rate)** - błąd na poziomie znaków

## 🔧 Rozwiązywanie problemów

### Problem z pamięcią GPU
```python
# Zmniejsz batch_size w konfiguracji
BATCH_SIZE = 8  # zamiast 16 lub 32
```

### Problem z tokenizerem
```python
# Sprawdź dostępne tokeny
tokenizer = get_tokenizer()
print(list(tokenizer.get_vocab().keys())[:20])
```

### Kernel crash podczas trenowania
- Ustaw `num_workers=0` w DataLoader
- Zmniejsz rozmiar modelu
- Użyj gradient accumulation

## 📈 Monitoring trenowania

Uruchom TensorBoard aby monitorować proces trenowania:

```bash
tensorboard --logdir runs/speech2text_training/
```

## 🤝 Wkład w projekt

1. Fork repozytorium
2. Stwórz branch dla nowej funkcjonalności (`git checkout -b feature/nowa-funkcjonalnosc`)
3. Commituj zmiany (`git commit -am 'Dodaj nową funkcjonalność'`)
4. Push do brancha (`git push origin feature/nowa-funkcjonalnosc`)
5. Stwórz Pull Request

## 📄 Licencja

Ten projekt jest dostępny na licencji MIT. Zobacz plik `LICENSE` dla szczegółów.

## 👨‍💻 Autor

**Kamil Czyżewski**

## 🙏 Podziękowania

- Dataset: [m-aliabbas/idrak_timit_subsample1](https://huggingface.co/datasets/m-aliabbas/idrak_timit_subsample1)
- Inspiracja: Projekty rozpoznawania mowy z użyciem PyTorch
- Społeczność PyTorch za doskonałą dokumentację

---

**Uwaga**: Ten projekt jest w fazie rozwoju. Niektóre funkcjonalności mogą wymagać dalszego dopracowania.
