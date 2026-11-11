
import numpy as np

from PIL import Image
from scipy.stats import skew, kurtosis

from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel
from skimage.measure import shannon_entropy


def extract_features(image_path):

    # Open image and convert to RGB
    image = Image.open(image_path).convert("RGB")

    # Convert image into a NumPy array
    rgb = np.array(image)

    # Separate RGB channels
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    # Convert image into grayscale
    gray = np.array(image.convert("L"))

    features = {}

    # 1. RGB statistical features
    for channel_name, channel in [
        ("red", red),
        ("green", green),
        ("blue", blue)
    ]:

        features[f"{channel_name}_mean"] = np.mean(channel)
        features[f"{channel_name}_std"] = np.std(channel)

    # 2. Grayscale statistical features
    pixels = gray.flatten()

    features["gray_mean"] = np.mean(pixels)
    features["gray_std"] = np.std(pixels)
    features["gray_variance"] = np.var(pixels)

    # 3. Statistical distribution features
    features["entropy"] = shannon_entropy(gray)
    features["skewness"] = skew(pixels)
    features["kurtosis"] = kurtosis(pixels)

    # 4. Edge density
    edges = sobel(gray)

    edge_threshold = np.mean(edges) + np.std(edges)

    features["edge_density"] = np.mean(
        edges > edge_threshold
    )

    # 5. GLCM texture features
    # Reduce grayscale intensity levels from 256 to 16
    gray_quantized = (gray // 16).astype(np.uint8)

    glcm = graycomatrix(
        gray_quantized,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=16,
        symmetric=True,
        normed=True
    )

    for property_name in [
        "contrast",
        "homogeneity",
        "energy",
        "correlation"
    ]:

        values = graycoprops(glcm, property_name)

        features[f"glcm_{property_name}"] = np.mean(values)

    return features
