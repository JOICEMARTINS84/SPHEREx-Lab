import csv
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.utils.data import download_file
from scipy.ndimage import maximum_filter, map_coordinates

BASE = "https://irsa.ipac.caltech.edu/ibe/data/spherex/"

URL = (
    BASE
    + "dr1/level2/2025W18_1B/l2b-v27-2026-196/1/"
    + "level2_2025W18_1B_0074_2D1_spx_l2b-v27-2026-196.fits"
)

arquivo = download_file(URL, cache=True, timeout=180)

print("\nEXPERIMENTO 005")

with fits.open(arquivo) as dados:

    imagem = dados["IMAGE"].data.astype(float)
    zodi = dados["ZODI"].data.astype(float)
    variancia = dados["VARIANCE"].data.astype(float)
    flags = dados["FLAGS"].data.astype(np.uint32)

    tabela = dados["EPSF"].data.copy()
    cabecalho = dados["EPSF"].header.copy()

imagem -= zodi

# Indicadores de qualidade

transient = (flags & 1) != 0

bits = [1, 2, 6, 7, 9, 10, 11, 14, 15]
mascara = sum(1 << bit for bit in bits)

validos = (
    np.isfinite(imagem)
    & np.isfinite(variancia)
    & (variancia > 0)
    & ((flags & mascara) == 0)
)

# Modelos ópticos

cx = tabela["XCENTER"]
cy = tabela["YCENTER"]

ox = int(cabecalho["OVSMPX"])
oy = int(cabecalho["OVSMPY"])

print("Modelos EPSF:", len(tabela))
print("Superamostragem:", ox, oy)

# Coordenadas da janela de análise

r = 4

Y, X = np.mgrid[-r:r+1, -r:r+1]

anel = (X**2 + Y**2 >= 9)

fases = [-0.4, 0, 0.4]

# Construção dos modelos na resolução do detector

modelos = {}

for zona in range(len(tabela)):

    epsf = np.array(
        tabela["EPSF"][zona],
        dtype=float
    ).reshape(33, 33)

    lista = []

    for dy in fases:
        for dx in fases:

            p = map_coordinates(
                epsf,
                [
                    16 + (Y - dy) * oy,
                    16 + (X - dx) * ox
                ],
                order=1,
                mode="constant",
                cval=0,
                prefilter=False
            )

            soma = p.sum()

            if np.isfinite(p).all() and soma > 0:
                lista.append(p / soma)

    if lista:
        modelos[zona] = np.array(lista)

# Identificação dos máximos locais

segura = np.where(validos, imagem, -np.inf)

picos = (
    validos
    & (segura == maximum_filter(segura, size=3))
)

picos[:r, :] = False
picos[-r:, :] = False
picos[:, :r] = False
picos[:, -r:] = False

# Amostragem reproduzível

rng = np.random.default_rng(2026)

resultados = []

for grupo, selecao in [
    ("TRANSIENT", picos & transient),
    ("SEM_TRANSIENT", picos & ~transient)
]:

    coordenadas = np.argwhere(selecao)

    print("\nGrupo:", grupo)
    print("Máximos locais:", len(coordenadas))

    indices = rng.choice(
        len(coordenadas),
        size=min(2000, len(coordenadas)),
        replace=False
    )

    for y, x in coordenadas[indices]:

        regiao = np.s_[
            y-r:y+r+1,
            x-r:x+r+1
        ]

        patch = imagem[regiao]
        var = variancia[regiao]
        ok = validos[regiao]

        if ok.sum() < 65:
            continue

        borda = patch[anel & ok]

        if borda.size < 15:
            continue

        fundo = np.median(borda)

        snr = (
            imagem[y, x] - fundo
        ) / np.sqrt(variancia[y, x])

        if snr < 5:
            continue

        # Seleção do modelo óptico mais próximo

        zona = np.argmin(
            (cx - x)**2 + (cy - y)**2
        )

        if zona not in modelos:
            continue

        d = patch[ok]
        w = 1 / var[ok]

        media = np.average(d, weights=w)

        chi0 = np.sum(w * (d - media)**2)

        if chi0 <= 0:
            continue

        melhor_chi = np.inf
        melhor_amplitude = 0

        # Ajuste matemático ponderado

        for modelo in modelos[zona]:

            p = modelo[ok]

            pm = np.average(p, weights=w)

            q = p - pm

            denominador = np.sum(w * q**2)

            if denominador <= 0:
                continue

            amplitude = np.sum(
                w * q * (d - media)
            ) / denominador

            amplitude = max(0, amplitude)

            b = media - amplitude * pm

            previsto = b + amplitude * p

            chi = np.sum(
                w * (d - previsto)**2
            )

            if chi < melhor_chi:

                melhor_chi = chi
                melhor_amplitude = amplitude

        if not np.isfinite(melhor_chi):
            continue

        # Fração da variação explicada pelo modelo

        r2 = 1 - melhor_chi / chi0

        resultados.append([
            grupo,
            int(x),
            int(y),
            float(snr),
            float(r2),
            float(melhor_amplitude),
            int(zona)
        ])

    print("Amostrados:", len(indices))

# Exportação dos resultados

with open(
    "experimento_005.csv",
    "w",
    newline="",
    encoding="utf-8"
) as arquivo:

    writer = csv.writer(arquivo)

    writer.writerow([
        "grupo",
        "x",
        "y",
        "snr_pixel",
        "r2",
        "amplitude",
        "zona"
    ])

    writer.writerows(resultados)

# Análise estatística

plt.figure(figsize=(10, 6))

for grupo in ["TRANSIENT", "SEM_TRANSIENT"]:

    valores = np.array([
        linha[4]
        for linha in resultados
        if linha[0] == grupo
    ])

    print("\nRESULTADOS:", grupo)
    print("Candidatos analisados:", valores.size)

    if valores.size:

        print("R2 médio:", np.mean(valores))
        print("R2 mediano:", np.median(valores))

        print(
            "R2 acima de 0.8:",
            np.sum(valores > 0.8)
        )

        plt.hist(
            valores,
            bins=np.linspace(0, 1, 41),
            density=True,
            alpha=0.5,
            label=grupo
        )

plt.xlabel("Compatibilidade com o modelo ePSF (R²)")
plt.ylabel("Densidade")
plt.title("SPHEREx — Análise morfológica")
plt.legend()
plt.tight_layout()
plt.savefig("experimento_005.png", dpi=300)
plt.show()