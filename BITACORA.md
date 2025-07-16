## Documentación del Proceso de Desarrollo del Algoritmo de Comparación de Pronunciación

Este documento detalla el proceso iterativo de desarrollo de un algoritmo para comparar la pronunciación del usuario con una referencia, utilizando técnicas de procesamiento de audio, aprendizaje profundo (Redes Siamesas con CNNs) y alineación temporal (DTW). Se basa en la revisión del historial de desarrollo y las decisiones tomadas durante el proceso.

### 1. Etapas Iniciales y Preprocesamiento

El proceso comenzó con la carga y el preprocesamiento básico de datos de audio del dataset SPEECHCOMMANDS, centrándose en un subconjunto de 5 palabras clave ("yes", "no", "up", "down", "left"). El objetivo inicial era preparar los datos de audio para la extracción de características.

*   **Carga y Filtrado del Dataset:** Se cargó el dataset SPEECHCOMMANDS y se filtraron las muestras correspondientes a las 5 palabras objetivo.
*   **Preprocesamiento Básico:** Las primeras aproximaciones involucraron el remuestreo de los audios a una frecuencia objetivo consistente (16000Hz) y la generación de espectrogramas.

### 2. Extracción de Características

Para representar el audio de una manera que fuera útil para la comparación, se optó por extraer características de los espectrogramas.

*   **Generación de Espectrogramas Mel-scale:** Se seleccionó la transformada Mel-scale para generar espectrogramas, que son una representación visual de la energía del audio a través del tiempo en diferentes bandas de frecuencia, percibidas de manera similar por el oído humano.
*   **Ajuste de Parámetros:** Se definieron parámetros consistentes para la generación de espectrogramas, incluyendo `n_mels` (altura del espectrograma) y `max_len` (longitud temporal máxima), aplicando padding o truncamiento para estandarizar las dimensiones. Se introdujo un filtro de frecuencia mínima (`f_min`) (85Hz) para mitigar el ruido de baja frecuencia.

### 3. Dificultades Encontradas y Soluciones Iniciales

Durante las etapas tempranas, surgieron desafíos significativos, particularmente en la comparación de pronunciaciones que varían en longitud y velocidad.

*   **Dificultad:** Problemas con la detección de silencio inicial. Los métodos basados en umbrales simples a menudo resultaban frágiles y poco fiables para segmentar con precisión la palabra hablada.
    *   **Solución:** En lugar de una detección de silencio explícita, el enfoque evolucionó hacia un preprocesamiento robusto que se centra en una ventana de tiempo fija (`max_len`) y confía en la capacidad de los modelos posteriores (CNN) para extraer características relevantes de esa ventana, mitigando el impacto del silencio al principio o final o ruido de baja frecuencia (con `f_min`).
    *   **Impacto:** Simplificó el pipeline de preprocesamiento pero trasladó la responsabilidad de manejar la variabilidad temporal dentro de la ventana al modelo de aprendizaje profundo.

*   **Dificultad:** Comparar directamente secuencias de audio (o sus espectrogramas) de diferentes longitudes resultaba problemático para métodos de distancia euclidiana simple o modelos que requieren entradas de tamaño fijo.

### 4. Introducción de Dynamic Time Warping (DTW)

Para abordar la dificultad de comparar secuencias de diferente longitud y velocidad, se introdujo el algoritmo DTW.

*   **Implementación de DTW:** Se integró DTW para calcular la distancia de alineación temporal entre el espectrograma del usuario y el espectrograma de referencia. Se utilizó la distancia euclidiana como métrica de coste local.
*   **Obtención de la Ruta de Alineación:** DTW proporcionó una ruta de alineación óptima, mapeando los frames temporales de un espectrograma a los del otro.
    *   **Solución:** DTW permite encontrar la correspondencia no lineal entre las secuencias, compensando las diferencias en la velocidad del habla.
    *   **Impacto:** Este fue un hito clave. Permitió una comparación de secuencias mucho más significativa y robusta a las variaciones de timing. Sin embargo, la distancia DTW por sí sola no siempre se correlaciona directamente con la similitud perceptual o fonética de alto nivel.

### 5. Integración con Red Siamesa (CNN)

Para obtener una medida de similitud más sofisticada y que aprovechara representaciones de alto nivel, se decidió integrar una Red Siamesa con una CNN base.

*   **Definición de la CNN Base:** Se definió una arquitectura CNN con capas convolucionales, Batch Normalization, funciones de activación GELU y Max Pooling para extraer características jerárquicas de los espectrogramas.
*   **Creación de la Red Siamesa:** Se construyó una Red Siamesa utilizando la CNN definida como modelo base compartido, con el propósito de comparar dos entradas (espectrogramas) y determinar su similitud basada en la distancia (o similitud coseno) de sus vectores de características extraídos por la CNN.
*   **Cálculo de Similitud Inicial (Pre-DTW):** Se calculó la similitud coseno entre los vectores de características de los espectrogramas procesados (padding/truncamiento a `max_len`) usando la CNN base de la Siamesa.
    *   **Dificultad:** La comparación directa de características de espectrogramas de longitud fija (`max_len`) sin considerar la alineación temporal podía llevar a puntuaciones de similitud bajas o inexactas si las pronunciaciones no estaban perfectamente sincronizadas o tenían diferentes velocidades.
    *   **Impacto:** Esta dificultad se evidenció en los resultados iniciales de similitud sin DTW, que podían ser muy bajos (como -0.01%).

### 6. Combinación de DTW y Red Siamesa para Comparación Alineada

La solución final para una comparación robusta implicó combinar la alineación temporal de DTW con la extracción de características de la Red Siamesa.

*   **Alineación de Espectrogramas con DTW Path:** Utilizando la ruta de alineación obtenida de DTW, se "alinearon" los espectrogramas del usuario y la referencia, creando nuevas versiones que tenían la misma longitud temporal (la longitud de la ruta de alineación).
    *   **Solución:** La alineación temporal previa a la extracción de características asegura que la CNN procese frames que corresponden perceptualmente entre sí, independientemente de su posición original en el tiempo.
*   **Cálculo de Similitud Post-DTW:** Los espectrogramas *alineados* se pasaron a través de la CNN base de la Red Siamesa para obtener vectores de características alineados. Luego se calculó la similitud coseno entre estos vectores alineados.
    *   **Solución:** Comparar los vectores de características extraídos de los espectrogramas temporalmente alineados permite una evaluación de similitud que es robusta tanto a las variaciones de timing (manejadas por DTW) como a las complejidades acústicas (manejadas por la CNN).
    *   **Impacto:** Este enfoque demostró ser significativamente más efectivo. Los resultados mostraron un aumento drástico en la puntuación de similitud (por ejemplo, de -0.01% a 89.31%), confirmando que la alineación temporal es crucial antes de la comparación de características de alto nivel para esta tarea. El cálculo preciso del `flat_dim` para la capa lineal de la CNN fue un ajuste técnico necesario para manejar las dimensiones aplanadas después de las capas convolucionales/pooling.

### 7. Conclusiones

El desarrollo del algoritmo fue un proceso iterativo que abordó las dificultades de comparar secuencias de audio variables. La combinación de Dynamic Time Warping para la alineación temporal y una Red Siamesa con una base CNN para la extracción y comparación de características de alto nivel demostró ser un enfoque efectivo para evaluar la similitud de pronunciaciones, superando las limitaciones de los métodos de comparación directa sin alineación temporal. La robustez del preprocesamiento inicial también contribuyó a simplificar el pipeline general.
