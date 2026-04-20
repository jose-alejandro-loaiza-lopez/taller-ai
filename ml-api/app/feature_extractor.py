import cv2
import numpy as np

def get_color_histogram(image_bytes):
    """
    Extrae un vector de características RGB basado en histogramas de color.
    - Redimensiona la imagen a 128x128.
    - Calcula un histograma de 32 bins para cada canal R, G, B.
    - Normaliza L2 el vector completo para escalar (0-1).
    - Retorna vector de 96 dimensiones (R, G, B).
    
    NOTA: OpenCV carga en BGR, se convierte a RGB para consistencia.
    """
    # Convertir bytes a imagen de OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("No se pudo decodificar la imagen")

    # Normalizar tamaño de imagen para consistencia
    img = cv2.resize(img, (128, 128))
    
    # Convertir BGR (formato OpenCV) a RGB (formato estándar)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Calcular histogramas para cada canal de color (en orden RGB)
    hist_r = cv2.calcHist([img_rgb], [0], None, [32], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_rgb], [1], None, [32], [0, 256]).flatten()
    hist_b = cv2.calcHist([img_rgb], [2], None, [32], [0, 256]).flatten()

    # Normalizar cada histograma por separado (distribución de probabilidad por canal)
    hist_r = hist_r / (np.sum(hist_r) + 1e-8)
    hist_g = hist_g / (np.sum(hist_g) + 1e-8)
    hist_b = hist_b / (np.sum(hist_b) + 1e-8)

    # Concatenar en un vector de características (R, G, B) => 32*3 = 96 dimensiones
    features = np.concatenate([hist_r, hist_g, hist_b])

    return features