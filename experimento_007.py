from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.utils.data import download_file
from scipy.ndimage import map_coordinates

PASTA = Path(__file__).resolve().parent

BASE = "https://irsa.ipac.caltech.edu/ibe/data/spherex/"

URL = (
    BASE
    + "dr1/level2/2025W18_1B/l2b-v27-2026-196/1/"
    + "level2_2025W18_1B_0074_2D1_spx_l2b-v27-2026-196.fits"
)

arquivo = download_file(URL, cache=True, timeout=180)

# Seleção dos candidatos

df = pd.read_csv(PASTA / "experimento_005.csv")

candidatos = df[
    (df.grupo == "TRANSIENT")
    & (df.r2 > 0.8)
].sort_values("r2", ascending=False)

print("\nEXPERIMENTO 007")
print("Candidatos selecionados:", len(candidatos))

# Leitura dos dados astronômicos

with fits.open(arquivo) as dados:

    imagem = dados["IMAGE"].data.astype(float)
    zodi = dados["ZODI"].data.astype(float)
    variancia = dados["VARIANCE"].data.astype(float)
    flags = dados["FLAGS"].data.astype(np.uint32)

    header = dados["IMAGE"].header.copy()

    tabela = dados["EPSF"].data.copy()

    ox = int(dados["EPSF"].header["OVSMPX"])
    oy = int(dados["EPSF"].header["OVSMPY"])

    wcs = WCS(header).celestial

    spectral_wcs = WCS(
        header,
        fobj=dados,
        key="W"
    )

    spectral_wcs.sip = None

imagem -= zodi

data_obs = header.get("DATE-AVG", "")
mjd_obs = header.get("MJD-AVG", np.nan)

print("Observação:", header.get("OBSID"))
print("Data central:", data_obs)
print("MJD central:", mjd_obs)

# Máscara instrumental

bits = [1, 2, 6, 7, 9, 10, 11, 14, 15]
mascara = sum(1 << bit for bit in bits)

validos = (
    np.isfinite(imagem)
    & np.isfinite(variancia)
    & (variancia > 0)
    & ((flags & mascara) == 0)
)

# Configuração das matrizes

r = 4

Y, X = np.mgrid[-r:r+1, -r:r+1]

fases = [-0.4, 0, 0.4]

resultados = []

destino = PASTA / "candidatos"
destino.mkdir(exist_ok=True)

fig_geral, axes = plt.subplots(
    3, 3,
    figsize=(12, 12)
)

for ax in axes.flat:
    ax.set_visible(False)

# Investigação individual

for i, linha in enumerate(
    candidatos.itertuples(),
    start=1
):

    x = int(linha.x)
    y = int(linha.y)

    zona = int(linha.zona)

    regiao = np.s_[
        y-r:y+r+1,
        x-r:x+r+1
    ]

    patch = imagem[regiao]
    var = variancia[regiao]
    ok = validos[regiao]

    epsf = np.array(
        tabela["EPSF"][zona],
        dtype=float
    ).reshape(33, 33)

    d = patch[ok]
    w = 1 / var[ok]

    media = np.average(d, weights=w)

    chi0 = np.sum(w * (d - media)**2)

    melhor = None

    # Reconstrução matemática da fonte

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

            if (
                not np.isfinite(p).all()
                or p.sum() <= 0
            ):
                continue

            p /= p.sum()

            pm = np.average(p[ok], weights=w)

            q = p[ok] - pm

            denominador = np.sum(w * q**2)

            if denominador <= 0:
                continue

            A = np.sum(
                w * q * (d - media)
            ) / denominador

            A = max(0, A)

            B = media - A * pm

            modelo = B + A * p

            chi = np.sum(
                w * (d - modelo[ok])**2
            )

            if melhor is None or chi < melhor["chi"]:

                melhor = {
                    "chi": chi,
                    "modelo": modelo,
                    "dx": dx,
                    "dy": dy,
                    "amplitude": A,
                    "fundo": B
                }

    if melhor is None or chi0 <= 0:
        continue

    r2 = 1 - melhor["chi"] / chi0

    # Conversão para coordenadas celestes

    x_centro = x + melhor["dx"]
    y_centro = y + melhor["dy"]

    coordenada = wcs.pixel_to_world(
        x_centro,
        y_centro
    ).icrs

    ra = coordenada.ra.deg
    dec = coordenada.dec.deg

    wavelength, bandwidth = (
        spectral_wcs.pixel_to_world(
            x_centro,
            y_centro
        )
    )

    lambda_um = wavelength.to_value(u.um)
    banda_um = bandwidth.to_value(u.um)

    # Resíduo do modelo

    modelo = melhor["modelo"]

    residuo = np.where(
        ok,
        (patch - modelo) / np.sqrt(var),
        np.nan
    )

    nome = f"SPX-{i:03d}"

    print(f"\n{nome}")
    print("Pixel:", x, y)
    print(f"RA: {ra:.8f}")
    print(f"DEC: {dec:.8f}")
    print(f"Lambda: {lambda_um:.6f} um")
    print(f"R2: {r2:.8f}")
    print(f"SNR: {linha.snr_pixel:.4f}")

    if abs(r2 - linha.r2) > 1e-8:
        print("ATENÇÃO: R2 diferente do Exp. 005")

    resultados.append({
        "candidato": nome,
        "obsid": header.get("OBSID"),
        "data_obs": data_obs,
        "mjd_obs": mjd_obs,
        "x": x,
        "y": y,
        "x_centro": x_centro,
        "y_centro": y_centro,
        "ra": ra,
        "dec": dec,
        "lambda_um": lambda_um,
        "banda_um": banda_um,
        "r2": r2,
        "snr_pixel": linha.snr_pixel,
        "amplitude": melhor["amplitude"],
        "fundo": melhor["fundo"],
        "flags_central": int(flags[y, x])
    })

    # Visualização científica individual

    fig, ax = plt.subplots(
        1, 3,
        figsize=(13, 4)
    )

    observado = np.where(ok, patch, np.nan)

    vmin, vmax = np.nanpercentile(
        observado, [2, 100]
    )

    for painel, matriz, titulo in zip(
        ax[:2],
        [observado, modelo],
        ["Observação", "Modelo ePSF"]
    ):

        painel.imshow(
            matriz,
            origin="lower",
            cmap="inferno",
            vmin=vmin,
            vmax=vmax
        )

        painel.set_title(titulo)

    limite = max(
        1,
        np.nanpercentile(np.abs(residuo), 98)
    )

    ax[2].imshow(
        residuo,
        origin="lower",
        cmap="coolwarm",
        vmin=-limite,
        vmax=limite
    )

    ax[2].set_title("Resíduo normalizado")

    fig.suptitle(
        f"{nome} | R² = {r2:.4f}"
    )

    plt.tight_layout()

    fig.savefig(
        destino / f"{nome}.png",
        dpi=300
    )

    plt.close(fig)

    # Painel geral dos candidatos

    painel = axes.flat[i-1]

    painel.set_visible(True)

    painel.imshow(
        observado,
        origin="lower",
        cmap="inferno"
    )

    painel.set_title(
        f"{nome} | R² = {r2:.3f}"
    )

# Exportação das coordenadas

pd.DataFrame(resultados).to_csv(
    PASTA / "experimento_007.csv",
    index=False
)

fig_geral.suptitle(
    "SPHEREx — Candidatos TRANSIENT"
)

fig_geral.tight_layout()

fig_geral.savefig(
    PASTA / "experimento_007.png",
    dpi=300
)

print("\nEXPERIMENTO CONCLUÍDO")

plt.show()