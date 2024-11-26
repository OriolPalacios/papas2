import os
import cv2
import numpy as np
import pandas as pd
import csv
from skimage import feature
import scipy.signal
from google.colab import drive

# Encabezados para el CSV -- No modificar
encabezados = ['forma', 'textura_contrast', 'textura_dissimilarity', 'textura_homogeneity', 'textura_energy', 'textura_correlation'] + \
               ['color_primario', 'color_secundario', 'color_terciario'] + \
               ['hue_bin_0', 'hue_bin_9', 'hue_bin_17', 'hue_bin_19', 'hue_bin_50',
                'hue_bin_75', 'hue_bin_83', 'hue_bin_90', 'hue_bin_96', 'hue_bin_105',
                'hue_bin_110', 'hue_bin_114', 'hue_bin_120', 'hue_bin_150', 'variedad']
               
               
# forma dict, para convertir sus datos categóricos a numéticos
forma_dict = {"Alargado": 0, "Comprimido": 1, "Esferico": 2, "Irregular": 3, "Largo-Oblongo": 4, "Oblongo": 5, "Obovoide": 6, "Ovoide": 7,"Reniforme":8}

# Asegúrate de que la función 'forma' esté correctamente definida
def forma(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, binarizada = cv2.threshold(gris, 240, 255, cv2.THRESH_BINARY_INV)
    contornos, _ = cv2.findContours(binarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contorno in contornos:
        epsilon = 0.02 * cv2.arcLength(contorno, True)
        aproximacion = cv2.approxPolyDP(contorno, epsilon, True)
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)
        if perimetro == 0:
            continue
        x, y, w, h = cv2.boundingRect(contorno)
        relacion_aspecto = w / float(h)
        circularidad = (4 * np.pi * area) / (perimetro ** 2)

        if circularidad > 0.85:
            forma = 'Esférico'
        elif relacion_aspecto < 1.2:
            if circularidad > 0.75:
                forma = 'Ovoide'
            elif len(aproximacion) > 8:
                forma = 'Obovoide'
            else:
                forma = 'Comprimido'
        elif 1.2 <= relacion_aspecto < 1.5:
            forma = 'Oblongo'
        elif 1.5 <= relacion_aspecto < 2:
            forma = 'Largo-Oblongo'
        elif 2 <= relacion_aspecto < 3:
            forma = 'Alargado'
        elif len(aproximacion) > 12 and w > h * 1.5:
            forma = 'Clavado'
        elif len(aproximacion) < 10 and relacion_aspecto > 1.2 and circularidad < 0.5:
            forma = 'Reniforme'
        elif len(aproximacion) > 8 and relacion_aspecto > 2.5:
            forma = 'Fusiforme'
        elif len(aproximacion) < 8 and circularidad < 0.3 and w < h:
            forma = 'Falcado'
        elif len(aproximacion) > 12 and circularidad < 0.3 and w > h * 2:
            forma = 'Enroscado'
        elif len(aproximacion) > 8 and circularidad < 0.6:
            forma = 'Digitado'
        elif len(aproximacion) > 15 and circularidad < 0.5:
            forma = 'Concertinoide'
        elif len(aproximacion) > 20 and circularidad < 0.4:
            forma = 'Tuberosado'
        else:
            forma = 'Irregular'

        return forma

# Función para extraer características de textura
def extraer_caracteristicas_textura(image: np.array) -> np.array:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = feature.graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = feature.graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = feature.graycoprops(glcm, 'homogeneity')[0, 0]
    energy = feature.graycoprops(glcm, 'energy')[0, 0]
    correlation = feature.graycoprops(glcm, 'correlation')[0, 0]
    return np.array([contrast, dissimilarity, homogeneity, energy, correlation])

def extract_gch(image, bins=180):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_hue = cv2.calcHist([hsv_image], [0], None, [bins], [0, 180])
    hist_hue = cv2.normalize(hist_hue, hist_hue, norm_type=cv2.NORM_L1).flatten()
    important_bins = [0, 9, 17, 19, 50, 75, 83, 90, 96, 105, 110, 114, 120, 150]
    selected_features = [hist_hue[i] for i in important_bins]
    return selected_features

def extract_histogram(image, bins=128):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist, _ = np.histogram(image_hsv[..., 0], bins=bins, range=(0, 180))
    hist = hist / np.sum(hist)
    return hist

def extract_3_colores(image, bins=180):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_hue = cv2.calcHist([hsv_image], [0], None, [bins], [0, 180])
    hist_hue = hist_hue.flatten()
    peaks, _ = scipy.signal.find_peaks(hist_hue)
    sorted_peaks = sorted(peaks, key=lambda x: hist_hue[x], reverse=True)
    top_peaks = sorted_peaks[:3]
    top_3_colors_hue = [int((peak / bins) * 360) for peak in top_peaks]
    return top_3_colors_hue


### ---------- EXTRAER CARACTERÍSTICAS --------------
def generar_datos(img: np.array):
    if img is not None:
        element = []
        forma_str = forma(img)
        forma_num = forma_dict.get(forma_str, 0)  # Obtener el valor numérico de la forma
        element.append(forma_str)  # Asegúrate de que la función forma esté definida
        element.extend(list(map(lambda x: '{0:.20f}'.format(x), extraer_caracteristicas_textura(img))))  # Textura
        element.extend(extract_3_colores(img))
        element.extend(list(map(lambda x: '{0:.20f}'.format(x), extract_gch(img))))
    return element