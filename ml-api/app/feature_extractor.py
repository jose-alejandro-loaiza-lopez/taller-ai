import cv2
import numpy as np

def get_color_histogram(image_bytes):
    # Convertir bytes a imagen de OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128)) # Redimensión estándar [cite: 358]

    # Calcular histograma por cada canal (R, G, B) [cite: 360]
    hist_features = []
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [32], [0, 256])
        cv2.normalize(hist, hist)
        hist_features.extend(hist.flatten())

    return np.array(hist_features) # Vector de 96 dimensiones [cite: 361]