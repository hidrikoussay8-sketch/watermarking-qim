import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from skimage.metrics import peak_signal_noise_ratio as psnr
import os

# -----------------------------
# 1. DCT et IDCT
# -----------------------------
def dct2(image):
    return dct(dct(image.T, norm='ortho').T, norm='ortho')

def idct2(coeff):
    return idct(idct(coeff.T, norm='ortho').T, norm='ortho')

# -----------------------------
# 2. Génération watermark
# -----------------------------
def generate_watermark(size):
    return np.random.randint(0, 2, size)

# -----------------------------
# 3. Insertion QIM
# -----------------------------
def embed_qim(dct_coeffs, watermark, delta, key):
    np.random.seed(key)
    flat = dct_coeffs.flatten()

    indices = np.random.choice(len(flat), len(watermark), replace=False)

    for i, bit in enumerate(watermark):
        idx = indices[i]

        if bit == 0:
            flat[idx] = delta * np.floor(flat[idx] / delta)
        else:
            flat[idx] = delta * np.floor(flat[idx] / delta) + delta / 2

    return flat.reshape(dct_coeffs.shape), indices

# -----------------------------
# 4. Extraction QIM
# -----------------------------
def extract_qim(dct_coeffs, indices, delta):
    flat = dct_coeffs.flatten()
    extracted = []

    for idx in indices:
        value = flat[idx]
        extracted.append(0 if (value % delta) < (delta / 2) else 1)

    return np.array(extracted)

# -----------------------------
# 5. Attaques
# -----------------------------
def add_noise(image):
    noise = np.random.normal(0, 10, image.shape)
    return np.clip(image + noise, 0, 255)

def compress_jpeg(image):
    cv2.imwrite("temp.jpg", image.astype(np.uint8))
    return cv2.imread("temp.jpg", 0)

# -----------------------------
# 6. BER
# -----------------------------
def compute_ber(original, extracted):
    return np.sum(original != extracted) / len(original)

# -----------------------------
# 7. MAIN
# -----------------------------

image = cv2.imread("image.jpg", 0)

if image is None:
    print("Erreur image !")
    exit()

image = image.astype(np.float64)

# DCT
dct_img = dct2(image)

# Watermark
wm = generate_watermark(64)

# Embed
delta = 10
key = 123
dct_watermarked, indices = embed_qim(dct_img, wm, delta, key)

# Reconstruction
watermarked_img = idct2(dct_watermarked)

# IMPORTANT : normalisation
watermarked_img = np.clip(watermarked_img, 0, 255)

# Attaques
noisy_img = add_noise(watermarked_img)
jpeg_img = compress_jpeg(watermarked_img)

# Extraction
wm_extracted_clean = extract_qim(dct2(watermarked_img), indices, delta)
wm_extracted_noise = extract_qim(dct2(noisy_img), indices, delta)
wm_extracted_jpeg = extract_qim(dct2(jpeg_img), indices, delta)

# PSNR (FIX IMPORTANT)
psnr_value = psnr(image, watermarked_img, data_range=255)

# BER
ber_clean = compute_ber(wm, wm_extracted_clean)
ber_noise = compute_ber(wm, wm_extracted_noise)
ber_jpeg = compute_ber(wm, wm_extracted_jpeg)

# -----------------------------
# DISPLAY
# -----------------------------
plt.figure(figsize=(16,8))

plt.subplot(2,4,1)
plt.imshow(image, cmap='gray')
plt.title("Image Originale")
plt.axis('off')

plt.subplot(2,4,2)
plt.imshow(watermarked_img, cmap='gray')
plt.title(f"Image tatouée\nPSNR = {psnr_value:.2f}")
plt.axis('off')

plt.subplot(2,4,3)
plt.imshow(noisy_img, cmap='gray')
plt.title("Bruit Gaussien")
plt.axis('off')

plt.subplot(2,4,4)
plt.imshow(jpeg_img, cmap='gray')
plt.title("JPEG")
plt.axis('off')

def plot_stem(data, title):
    plt.stem(data)
    plt.ylim(-0.1, 1.5)
    plt.title(title)

plt.subplot(2,4,5)
plot_stem(wm, "Watermark original")

plt.subplot(2,4,6)
plot_stem(wm_extracted_clean, f"Clean BER={ber_clean:.4f}")

plt.subplot(2,4,7)
plot_stem(wm_extracted_noise, f"Noise BER={ber_noise:.4f}")

plt.subplot(2,4,8)
plot_stem(wm_extracted_jpeg, f"JPEG BER={ber_jpeg:.4f}")

plt.suptitle("Tatouage numérique QIM", fontsize=16)
plt.tight_layout()

plt.savefig("resultat.png", dpi=300)

print("Saved:", os.path.abspath("resultat.png"))