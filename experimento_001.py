import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.utils.data import download_file

# Fonte oficial dos dados
base = "https://irsa.ipac.caltech.edu/ibe/data/spherex/"

url = (
    base
    + "dr1/level2/2025W18_1B/l2b-v27-2026-196/1/"
    + "level2_2025W18_1B_0074_2D1_spx_l2b-v27-2026-196.fits"
)

# Download do arquivo científico
arquivo = download_file(url, cache=True, timeout=180)

# Leitura dos dados
with fits.open(arquivo) as dados:
    dados.info()
    imagem = dados["IMAGE"].data.copy()

# Análise estatística preliminar
pixels = imagem[np.isfinite(imagem)]

print("Dimensões:", imagem.shape)
print("Média:", np.mean(pixels))
print("Mediana:", np.median(pixels))
print("Desvio padrão:", np.std(pixels))

# Visualização astronômica
plt.figure(figsize=(10, 10))
plt.imshow(
    imagem,
    cmap="gray",
    origin="lower",
    vmin=np.percentile(pixels, 2),
    vmax=np.percentile(pixels, 98)
)

plt.colorbar(label="Brilho superficial (MJy/sr)")
plt.title("SPHEREx — Experimento 001")
plt.xlabel("Pixel X")
plt.ylabel("Pixel Y")
plt.show()