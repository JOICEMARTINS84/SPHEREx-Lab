import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.utils.data import download_file

base = "https://irsa.ipac.caltech.edu/ibe/data/spherex/"

url = (
    base
    + "dr1/level2/2025W18_1B/l2b-v27-2026-196/1/"
    + "level2_2025W18_1B_0074_2D1_spx_l2b-v27-2026-196.fits"
)

arquivo = download_file(url, cache=True, timeout=180)

with fits.open(arquivo) as dados:
    imagem = dados["IMAGE"].data.copy()
    flags = dados["FLAGS"].data.copy().astype(np.uint32)
    zodi = dados["ZODI"].data.copy()
    variancia = dados["VARIANCE"].data.copy()

# Correção do fundo zodiacal
imagem = imagem - zodi

validos = (
    np.isfinite(imagem)
    & np.isfinite(variancia)
    & (variancia > 0)
)

# Identificação individual dos indicadores
transient = (flags & (1 << 0)) != 0
outlier = (flags & (1 << 19)) != 0
source = (flags & (1 << 21)) != 0

print("\nEXPERIMENTO 003")
print("Total de pixels:", imagem.size)
print("Pixels numericamente válidos:", validos.sum())
print("TRANSIENT:", transient.sum())
print("OUTLIER:", outlier.sum())
print("SOURCE:", source.sum())

# Máscara experimental de problemas instrumentais
bits = [1, 2, 6, 7, 9, 10, 11, 14, 15]
mascara = sum(1 << bit for bit in bits)

base_valida = validos & ((flags & mascara) == 0)

# Estimativa robusta do fundo
fundo = imagem[base_valida & ~source & ~outlier & ~transient]

if fundo.size == 0:
    raise ValueError("Amostra de fundo insuficiente.")

mediana = np.median(fundo)
mad = np.median(np.abs(fundo - mediana))
sigma = 1.4826 * mad

if sigma <= 0:
    raise ValueError("Dispersão robusta inválida.")

# Comparação entre filtros
filtros = {
    "Base": base_valida,
    "Sem OUTLIER": base_valida & ~outlier,
    "Sem TRANSIENT": base_valida & ~transient,
    "Flags iguais a zero": base_valida & (flags == 0)
}

for k in [3, 5, 7, 10]:
    candidatos = imagem > mediana + k * sigma
    total = np.sum(candidatos & base_valida)

    print(f"\nLimiar: {k} sigma")

    for nome, filtro in filtros.items():
        restantes = np.sum(candidatos & filtro)
        perda = 100 * (1 - restantes / total) if total else np.nan

        print(f"{nome}: {restantes} | Perda: {perda:.4f}%")

# Distribuição espacial dos indicadores
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, mapa, nome in zip(
    axes,
    [transient, outlier, source],
    ["TRANSIENT", "OUTLIER", "SOURCE"]
):
    ax.imshow(mapa, cmap="inferno", origin="lower")
    ax.set_title(nome)

plt.tight_layout()
plt.savefig("experimento_003.png", dpi=300)
plt.show()