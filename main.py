import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dtw import dtw
from scipy.spatial.distance import euclidean
import os
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Audio, display, HTML
import base64

# --- Parámetros Globales ---
TARGET_SR = 16000
SPECTROGRAM_HEIGHT = 32
MAX_LEN = 200
F_MIN_FILTER = 85
CLASES = ["yes", "no", "up", "down", "left"]
CLASS_MAP = {label: idx for idx, label in enumerate(CLASES)}

# --- Modelos ---
class CNNModel(nn.Module):
    """CNN base para la extracción de características de espectrogramas."""
    def __init__(self, height, width):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2)
        )
        self.dropout = nn.Dropout(0.3)

        dummy = torch.randn(1, 1, height, width)
        out = self.conv3(self.conv2(self.conv1(dummy)))
        self.flat_dim = out.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flat_dim, 256)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = nn.GELU()(self.fc1(x))
        return x

class SiameseNetwork(nn.Module):
    """Red Siamesa para comparar dos espectrogramas."""
    def __init__(self, base_cnn):
        super(SiameseNetwork, self).__init__()
        self.base_cnn = base_cnn

    def forward_once(self, x):
        return self.base_cnn(x)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# --- Funciones de Preprocesamiento de Audio ---
def preprocess_audio(waveform, sr_original, target_sr, n_mels, f_min, max_len):
    """Remuestrea, genera y procesa un espectrograma de un audio."""
    if sr_original != target_sr:
        resampler = T.Resample(orig_freq=sr_original, new_freq=target_sr)
        waveform = resampler(waveform)

    spectrogram_transform = T.MelSpectrogram(
        sample_rate=target_sr, n_fft=1024, hop_length=512, n_mels=n_mels, f_min=f_min
    )
    spec = spectrogram_transform(waveform).squeeze(0)

    if spec.shape[1] < max_len:
        spec = F.pad(spec, (0, max_len - spec.shape[1]), "constant", 0)
    else:
        spec = spec[:, :max_len]

    return spec

# --- Funciones de Comparación ---
def calculate_similarity(model, spec1, spec2):
    """Calcula la similitud coseno entre dos espectrogramas usando el modelo Siamesa."""
    model.eval()
    with torch.no_grad():
        vec1 = model.forward_once(spec1)
        vec2 = model.forward_once(spec2)
    similarity = F.cosine_similarity(vec1, vec2, dim=1).item()
    return similarity * 100

def align_and_compare(model, user_spec, ref_spec):
    """Alinea espectrogramas con DTW y calcula la similitud."""
    user_np = user_spec.numpy().T
    ref_np = ref_spec.numpy().T

    dtw_result = dtw(user_np, ref_np, dist=euclidean)
    path_x, path_y = dtw_result[3]

    user_aligned = torch.index_select(user_spec, dim=1, index=torch.tensor(path_x, dtype=torch.long))
    ref_aligned = torch.index_select(ref_spec, dim=1, index=torch.tensor(path_y, dtype=torch.long))

    return calculate_similarity(model, user_aligned, ref_aligned), user_aligned, ref_aligned

# --- Funciones de Visualización ---
def plot_spectrograms(user_spec, ref_spec, word, title_prefix=""):
    """Visualiza los espectrogramas del usuario y de referencia."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    vmin = min(user_spec.min(), ref_spec.min())
    vmax = max(user_spec.max(), ref_spec.max())

    sns.heatmap(user_spec.squeeze().numpy(), cmap="viridis", cbar=True, ax=axes[0], vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{title_prefix}Espectrograma de tu Grabación")
    axes[0].set_xlabel("Tiempo (Frames)")
    axes[0].set_ylabel("Frecuencia (Mel Bin)")
    axes[0].invert_yaxis()

    sns.heatmap(ref_spec.squeeze().numpy(), cmap="viridis", cbar=True, ax=axes[1], vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{title_prefix}Referencia ('{word}')")
    axes[1].set_xlabel("Tiempo (Frames)")
    axes[1].set_ylabel("Frecuencia (Mel Bin)")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.show()

# --- Funciones de Grabación ---
RECORD_JS = """
<script>
// ... (código Javascript de grabación) ...
</script>
"""

def record_audio_js():
    """Muestra el botón de grabación en el notebook."""
    display(HTML(RECORD_JS))
    print("\nHaz clic en el botón para grabar tu pronunciación (5s).")

def save_audio_callback(data):
    """Guarda el audio grabado en un archivo."""
    audio = base64.b64decode(data)
    with open("user.wav", "wb") as f:
        f.write(audio)
    print("✔️ Audio guardado como 'user.wav'.")

# --- Flujo Principal ---
def main():
    # Esta función está diseñada para un entorno de notebook.
    # Para una ejecución de script estándar, se necesitaría un enfoque diferente para la entrada de audio.
    print("El script está listo para ser ejecutado en un entorno de notebook.")
    print("Para probar la funcionalidad, necesitarás ejecutar las funciones individualmente.")

if __name__ == '__main__':
    # Esta sección es para la ejecución en un entorno de notebook.
    # Para ejecutar como un script, se necesitarían ajustes adicionales.
    pass
