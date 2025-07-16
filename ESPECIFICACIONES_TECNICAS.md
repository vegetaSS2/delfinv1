## Especificaciones Técnicas y Recursos

### 1. Modelo de IA: Red Siamesa con Base CNN
La parte central del sistema es una **Red Siamesa** que utiliza una **Red Neuronal Convolucional (CNN)** como modelo base compartido.

**Arquitectura de la CNN Base:**
*   **Capas Convolucionales (Conv2d):**
    *   `Conv1`: Entrada de 1 canal, 32 filtros, kernel 3x3.
    *   `Conv2`: 32 canales de entrada, 64 filtros, kernel 3x3.
    *   `Conv3`: 64 canales de entrada, 128 filtros, kernel 3x3.
*   **Batch Normalization (BatchNorm2d):** Después de cada capa convolucional.
*   **Funciones de Activación (GELU):** Después de cada Batch Normalization.
*   **Capas de Max Pooling (MaxPool2d):** Después de cada capa convolucional.
*   **Capa Lineal (fc1):** Proyecta las características aplanadas a un vector de 256.

### 2. Algoritmo de Alineación: Dynamic Time Warping (DTW)
*   **Implementación:** Se utiliza la librería `dtw-python`.
*   **Métrica de Distancia:** Distancia Euclidiana.

### 3. Preprocesamiento de Audio
*   **Frecuencia de Muestreo Objetivo:** 16000Hz.
*   **Espectrogramas:** Mel-scale con `n_mels=32` y `f_min=85Hz`.
*   **Normalización de Longitud:** `max_len=200` (padding o truncamiento).

### 4. Recursos y Librerías
*   **PyTorch:** Framework principal para la construcción y uso de la red neuronal.
*   **Torchaudio:** Para el preprocesamiento de audio y carga del dataset.
*   **dtw-python:** Para la implementación del algoritmo DTW.
*   **NumPy y SciPy:** Para operaciones numéricas.
*   **Matplotlib y Seaborn:** Para la visualización de espectrogramas.
*   **Dataset:** SPEECHCOMMANDS.
