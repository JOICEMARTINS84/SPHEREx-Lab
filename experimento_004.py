import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.utils.data import download_file

BASE = "https://irsa.ipac.caltech.edu/ibe/data/spherex/"

URL = (
    BASE
    + "dr1/level2/2025W18_1B/l2b-v27-2026-196/1/"
    + "level2_2025W18_1B_0074_2D1_spx_l2b-v27-2026-196.fits"
)

arquivo = download_file(URL, cache=True, timeout=180)

with fits.open(arquivo) as dados:
    imagem = dados["IMAGE"].data.astype(float)
    zodi = dados["ZODI"].data.astype(float)
    variancia = dados["VARIANCE"].data.astype(float)
    flags = dados["FLAGS"].data.astype(np.uint32)

print("\nEXPERIMENTO 004", flush=True)

# Subtração da luz zodiacal
imagem = imagem - zodi

transient = (flags & (1 << 0)) != 0
outlier = (flags & (1 << 19)) != 0
source = (flags & (1 << 21)) != 0

# Mesma máscara instrumental do experimento 003
bits = [1, 2, 6, 7, 9, 10, 11, 14, 15]
mascara = sum(1 << bit for bit in bits)

validos = (
    np.isfinite(imagem)
    & np.isfinite(variancia)
    & (variancia > 0)
    & ((flags & mascara) == 0)
)

# Amostra para estimativa do fundo residual
amostra = validos & ~transient & ~outlier

fundo = np.full(imagem.shape, np.nan)
dispersao = np.full(imagem.shape, np.nan)

altura, largura = imagem.shape
bloco = 64

regioes = 0

# Processamento matemático por regiões
for y in range(0, altura, bloco):
    for x in range(0, largura, bloco):

        regiao = np.s_[
            y:y + bloco,
            x:x + bloco
        ]

        valores = imagem[regiao][amostra[regiao]]

        if valores.size < 100:
            continue

        mediana = np.median(valores)

        mad = np.median(
            np.abs(valores - mediana)
        )

        sigma = 1.4826 * mad

        if sigma <= 0:
            continue

        fundo[regiao] = mediana
        dispersao[regiao] = sigma

        regioes += 1

# Normalização pela dispersão local
residual = imagem - fundo
s_local = residual / dispersao

# Normalização pela incerteza instrumental
s_instrumental = residual / np.sqrt(
    np.where(validos, variancia, np.nan)
)

comuns = (
    validos
    & np.isfinite(s_local)
    & np.isfinite(s_instrumental)
)

print("Regiões analisadas:", regioes)
print("Pixels analisáveis:", comuns.sum())
print("Mediana do fundo:", np.nanmedian(fundo))

filtros = {
    "Base": comuns,
    "Sem TRANSIENT": comuns & ~transient,
    "Sem OUTLIER": comuns & ~outlier,
    "FLAGS = 0": comuns & (flags == 0)
}

for nome, matriz in [
    ("DISPERSÃO LOCAL", s_local),
    ("INCERTEZA INSTRUMENTAL", s_instrumental)
]:

    print("\n", nome)

    for k in [3, 5, 7, 10]:

        candidatos = comuns & (matriz > k)
        total = candidatos.sum()

        print(f"\nLimiar: {k}")

        for filtro, selecao in filtros.items():

            restantes = (
                candidatos & selecao
            ).sum()

            perda = (
                100 * (1 - restantes / total)
                if total else np.nan
            )

            print(
                f"{filtro}: {restantes} "
                f"| Perda: {perda:.4f}%"
            )

# Visualização científica
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

for ax, dados, titulo in zip(
    axes,
    [fundo, s_local, s_instrumental],
    [
        "Fundo local",
        "Dispersão local",
        "Incerteza instrumental"
    ]
):

    valores = dados[comuns]
    minimo, maximo = np.percentile(valores, [2, 98])

    mapa = ax.imshow(
        np.where(comuns, dados, np.nan),
        cmap="inferno",
        origin="lower",
        vmin=minimo,
        vmax=maximo
    )

    ax.set_title(titulo)
    fig.colorbar(mapa, ax=ax, shrink=0.7)

plt.tight_layout()
plt.savefig("experimento_004.png", dpi=300)
plt.show()