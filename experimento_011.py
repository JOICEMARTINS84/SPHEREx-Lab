from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.utils.data import download_file
from scipy.ndimage import maximum_filter, map_coordinates
from scipy.optimize import minimize

PASTA = Path(__file__).resolve().parent

catalogo = pd.read_csv(
    PASTA / "experimento_010_imagens.csv"
)

rng = np.random.default_rng(2026)

resultados = []

R = 4
Y, X = np.mgrid[-R:R+1, -R:R+1]

ANEL = X**2 + Y**2 >= 9

FASES = [-0.4, 0, 0.4]

AMOSTRA = 200

BITS = [1, 2, 6, 7, 9, 10, 11, 14, 15]

MASCARA = sum(1 << bit for bit in BITS)


# Construção matemática do modelo ePSF

def construir_modelo(epsf, dx, dy, ox, oy):

    cy, cx = (np.array(epsf.shape) - 1) / 2

    p = map_coordinates(
        epsf,
        [
            cy + (Y - dy) * oy,
            cx + (X - dx) * ox
        ],
        order=1,
        mode="constant",
        cval=0,
        prefilter=False
    )

    soma = p.sum()

    if not np.isfinite(p).all() or soma <= 0:
        return None

    return p / soma


# Ajuste por mínimos quadrados ponderados

def ajustar(patch, var, ok, epsf, ox, oy):

    d = patch[ok]
    w = 1 / var[ok]

    media = np.average(d, weights=w)

    chi0 = np.sum(w * (d - media)**2)

    if chi0 <= 0 or not np.isfinite(chi0):
        return None

    def avaliar(posicao):

        dx, dy = posicao

        modelo = construir_modelo(
            epsf, dx, dy, ox, oy
        )

        if modelo is None:
            return np.inf, np.nan, np.nan

        p = modelo[ok]

        pm = np.average(p, weights=w)

        q = p - pm

        denominador = np.sum(w * q**2)

        if denominador <= 0:
            return np.inf, np.nan, np.nan

        A = np.sum(
            w * q * (d - media)
        ) / denominador

        A = max(0, A)

        B = media - A * pm

        chi = np.sum(
            w * (d - B - A * p)**2
        )

        return chi, A, B

    # Método discreto do experimento 005

    grade = []

    for dx in FASES:
        for dy in FASES:

            chi, A, B = avaliar((dx, dy))

            if np.isfinite(chi):

                grade.append(
                    (chi, dx, dy, A, B)
                )

    if not grade:
        return None

    grade.sort()

    discreto = grade[0]

    # Otimização contínua

    def objetivo(posicao):

        return avaliar(posicao)[0] / chi0

    melhor = discreto

    convergiu = False

    # Duas condições iniciais

    for inicial in grade[:2]:

        otimo = minimize(
            objetivo,
            x0=[inicial[1], inicial[2]],
            method="Powell",
            bounds=[
                (-0.5, 0.5),
                (-0.5, 0.5)
            ],
            options={
                "xtol": 1e-5,
                "ftol": 1e-8,
                "maxiter": 100
            }
        )

        if not otimo.success:
            continue

        convergiu = True

        dx, dy = otimo.x

        chi, A, B = avaliar((dx, dy))

        if chi < melhor[0]:

            melhor = (
                chi, dx, dy, A, B
            )

    return {
        "r2_discreto": 1 - discreto[0] / chi0,
        "r2_continuo": 1 - melhor[0] / chi0,
        "dx_discreto": discreto[1],
        "dy_discreto": discreto[2],
        "dx_continuo": melhor[1],
        "dy_continuo": melhor[2],
        "amplitude": melhor[3],
        "fundo": melhor[4],
        "convergiu": convergiu
    }


# Processamento das seis imagens reais

print("EXPERIMENTO 011")

for linha in catalogo.itertuples():

    detector = int(linha.detector)

    print(f"\nDETECTOR {detector}", flush=True)

    arquivo = download_file(
        linha.url,
        cache=True,
        timeout=300
    )

    with fits.open(arquivo) as dados:

        imagem = dados["IMAGE"].data.astype(float)
        zodi = dados["ZODI"].data.astype(float)
        var = dados["VARIANCE"].data.astype(float)

        flags = dados["FLAGS"].data.astype(
            np.uint32
        )

        epsf = dados["EPSF"].data.copy()

        ox = int(dados["EPSF"].header["OVSMPX"])
        oy = int(dados["EPSF"].header["OVSMPY"])

    imagem -= zodi

    cx = epsf["XCENTER"]
    cy = epsf["YCENTER"]

    validos = (
        np.isfinite(imagem)
        & np.isfinite(var)
        & (var > 0)
        & ((flags & MASCARA) == 0)
    )

    transient = (flags & 1) != 0

    segura = np.where(validos, imagem, -np.inf)

    picos = (
        validos
        & (segura == maximum_filter(segura, size=3))
    )

    picos[:R, :] = False
    picos[-R:, :] = False
    picos[:, :R] = False
    picos[:, -R:] = False

    for grupo, selecao in [

        ("TRANSIENT", picos & transient),
        ("SEM_TRANSIENT", picos & ~transient)

    ]:

        coordenadas = np.argwhere(selecao)

        indices = rng.choice(
            len(coordenadas),
            size=min(AMOSTRA, len(coordenadas)),
            replace=False
        )

        contador = 0

        for y, x in coordenadas[indices]:

            regiao = np.s_[
                y-R:y+R+1,
                x-R:x+R+1
            ]

            patch = imagem[regiao]
            variancia = var[regiao]
            ok = validos[regiao]

            if ok.sum() < 65:
                continue

            borda = patch[ANEL & ok]

            if borda.size < 15:
                continue

            fundo = np.median(borda)

            snr = (
                imagem[y, x] - fundo
            ) / np.sqrt(var[y, x])

            if snr < 5:
                continue

            zona = np.argmin(
                (cx - x)**2 + (cy - y)**2
            )

            psf = np.asarray(
                epsf["EPSF"][zona],
                dtype=float
            ).reshape(33, 33)

            resultado = ajustar(
                patch,
                variancia,
                ok,
                psf,
                ox,
                oy
            )

            if resultado is None:
                continue

            # Deslocamento entre os métodos

            deslocamento = np.hypot(
                resultado["dx_continuo"]
                - resultado["dx_discreto"],

                resultado["dy_continuo"]
                - resultado["dy_discreto"]
            )

            # Indicadores de efeitos instrumentais

            flags_locais = flags[regiao]

            bloom = np.any(
                flags_locais & (1 << 26)
            )

            halo = np.any(
                flags_locais & (1 << 28)
            )

            saturacao_indicio = np.any(
                flags_locais & (
                    (1 << 1) | (1 << 15)
                )
            )

            resultados.append({

                "detector": detector,
                "exposicao": linha.exposicao,
                "grupo": grupo,
                "x": int(x),
                "y": int(y),
                "snr": snr,
                "flags_central": int(flags[y, x]),
                "bloom": bool(bloom),
                "halo": bool(halo),
                "saturacao_indicio": bool(
                    saturacao_indicio
                ),
                "deslocamento_pixel": deslocamento,
                "deslocamento_arcsec": deslocamento * 6.15,

                **resultado

            })

            contador += 1

        print(grupo, "Candidatos:", contador)

    # Salvamento após cada detector

    pd.DataFrame(resultados).to_csv(
        PASTA / "experimento_011.csv",
        index=False
    )


# Análise estatística

df = pd.DataFrame(resultados)

if df.empty:
    raise RuntimeError("Nenhum candidato analisado.")

df["ganho_r2"] = (
    df.r2_continuo - df.r2_discreto
)

resumo = df.groupby(
    ["detector", "grupo"]
).agg(

    candidatos=("snr", "size"),
    r2_discreto=("r2_discreto", "mean"),
    r2_continuo=("r2_continuo", "mean"),
    ganho_medio=("ganho_r2", "mean"),
    deslocamento_mediano=("deslocamento_arcsec", "median"),
    convergencia=("convergiu", "mean")

).reset_index()

print("\nRESULTADOS")
print(resumo.to_string(index=False))

resumo.to_csv(
    PASTA / "experimento_011_resumo.csv",
    index=False
)

df.to_csv(
    PASTA / "experimento_011.csv",
    index=False
)

# Visualização científica

fig, ax = plt.subplots(1, 2, figsize=(13, 6))

for grupo in ["TRANSIENT", "SEM_TRANSIENT"]:

    g = df[df.grupo == grupo]

    ax[0].scatter(
        g.r2_discreto,
        g.r2_continuo,
        s=8,
        alpha=0.35,
        label=grupo
    )

    ax[1].hist(
        g.deslocamento_arcsec,
        bins=40,
        alpha=0.5,
        density=True,
        label=grupo
    )

ax[0].plot([0, 1], [0, 1], "k--")

ax[0].set_xlabel("R² discreto")
ax[0].set_ylabel("R² contínuo")

ax[1].set_xlabel("Deslocamento angular aproximado (arcsec)")
ax[1].set_ylabel("Densidade")

for painel in ax:
    painel.legend()
    painel.grid(alpha=0.15)

plt.tight_layout()

plt.savefig(
    PASTA / "experimento_011.png",
    dpi=300
)

print("\nEXPERIMENTO CONCLUÍDO")

plt.show()