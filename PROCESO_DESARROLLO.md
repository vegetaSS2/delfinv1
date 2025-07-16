## Proceso de Desarrollo del Algoritmo de Comparación de Pronunciación

Este documento detalla el proceso iterativo de desarrollo de un algoritmo para comparar la pronunciación del usuario con una referencia.

### 1. Etapas Iniciales y Preprocesamiento
El proceso comenzó con la carga y el preprocesamiento básico de datos de audio del dataset SPEECHCOMMANDS, centrándose en un subconjunto de 5 palabras clave. El objetivo inicial era preparar los datos de audio para la extracción de características.

### 2. Extracción de Características
Para representar el audio de una manera que fuera útil para la comparación, se optó por extraer características de los espectrogramas Mel-scale.

### 3. Dificultades Encontradas y Soluciones Iniciales
*   **Dificultad:** Problemas con la detección de silencio inicial.
    *   **Solución:** En lugar de una detección de silencio explícita, el enfoque evolucionó hacia un preprocesamiento robusto que se centra en una ventana de tiempo fija (`max_len`).
*   **Dificultad:** Comparar directamente secuencias de audio de diferentes longitudes.

### 4. Introducción de Dynamic Time Warping (DTW)
*   **Implementación de DTW:** Se integró DTW para calcular la distancia de alineación temporal entre el espectrograma del usuario y el de referencia.

### 5. Integración con Red Siamesa (CNN)
*   **Definición de la CNN Base:** Se definió una arquitectura CNN para extraer características jerárquicas de los espectrogramas.
*   **Cálculo de Similitud Inicial (Pre-DTW):** La comparación directa de características de espectrogramas de longitud fija sin alineación temporal podía llevar a puntuaciones de similitud bajas.

### 6. Combinación de DTW y Red Siamesa para Comparación Alineada
*   **Alineación de Espectrogramas con DTW Path:** Se alinearon los espectrogramas del usuario y la referencia.
*   **Cálculo de Similitud Post-DTW:** Se compararon los vectores de características extraídos de los espectrogramas temporalmente alineados.
    *   **Impacto:** Este enfoque demostró ser significativamente más efectivo, con un aumento drástico en la puntuación de similitud.

### 7. Conclusiones
El desarrollo del algoritmo fue un proceso iterativo que abordó las dificultades de comparar secuencias de audio variables. La combinación de DTW para la alineación temporal y una Red Siamesa con una base CNN para la extracción y comparación de características de alto nivel demostró ser un enfoque efectivo.
