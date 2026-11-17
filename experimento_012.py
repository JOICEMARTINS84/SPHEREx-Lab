from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
from astropy.utils.data import download_file
from astroquery.gaia import Gaia
from scipy.ndimage import map_coordinates, maximum_filter
from scipy.optimize import minimize

PASTA = Path(__file__).resolve().parent

catalogo = pd.read_csv(
    PASTA / "experimento_010_imagens.csv"
)

rng = np.random.default_rng(2026)

R = 4
Y, X = np.mgrid[-R:R+1, -R:R+1]

ANEL = X**2 + Y**2 >= 9
NUCLEO = X**2 + Y**2 <= 4

FASES = [-0.4, 0, 0.4]

MAX_ESTRELAS = 500

# Máscara conservadora para estrelas de referência

BITS = [
    0, 1, 2, 6, 7, 9, 10, 11, 12,
    14, 15, 17, 19, 20, 22, 23, 24,
    25, 26, 27, 28, 29
]

MASCARA = sum(1 << bit for bit in BITS)

resultados = []
resumos = []
status = []


# Ajuste matemático da ePSF

def ajustar(patch, var, ok, epsf, ox, oy):

    d = patch[ok]
    w = 1 / var[ok]

    media = np.average(d, weights=w)

    chi0 = np.sum(w * (d - media)**2)

    if chi0 <= 0:
        return None

    cy, cx = (np.array(epsf.shape) - 1) / 2

    def objetivo(pos):

        dx, dy = pos

        modelo = map_coordinates(
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

        if modelo.sum() <= 0:
            return np.inf

        modelo /= modelo.sum()

        p = modelo[ok]

        pm = np.average(p, weights=w)

        q = p - pm

        den = np.sum(w * q**2)

        if den <= 0:
            return np.inf

        A = max(
            0,
            np.sum(w * q * (d - media)) / den
        )

        B = media - A * pm

        chi = np.sum(
            w * (d - B - A * p)**2
        )

        return chi / chi0

    grade = sorted(
        (objetivo((dx, dy)), dx, dy)
        for dx in FASES
        for dy in FASES
    )

    discreto = grade[0]

    if not np.isfinite(discreto[0]):
        return None

    continuo = discreto

    convergiu = False

    for inicial in grade[:2]:

        opt = minimize(
            objetivo,
            x0=[inicial[1], inicial[2]],
            method="Powell",
            bounds=[(-0.5, 0.5), (-0.5, 0.5)],
            options={
                "xtol": 1e-5,
                "ftol": 1e-8,
                "maxiter": 100
            }
        )

        if opt.success and np.isfinite(opt.fun):

            convergiu = True

            if opt.fun < continuo[0]:

                continuo = (
                    opt.fun,
                    opt.x[0],
                    opt.x[1]
                )

    return discreto, continuo, convergiu


# Processamento individual das exposições

def processar(linha):

    detector = int(linha.detector)

    arquivo = download_file(
        linha.url,
        cache=True,
        timeout=300
    )

    with fits.open(arquivo) as dados:

        imagem = dados["IMAGE"].data.astype(float)
        imagem -= dados["ZODI"].data

        var = dados["VARIANCE"].data.astype(float)
        flags = dados["FLAGS"].data.astype(np.uint32)

        header = dados["IMAGE"].header.copy()

        epsf = dados["EPSF"].data.copy()

        ox = int(dados["EPSF"].header["OVSMPX"])
        oy = int(dados["EPSF"].header["OVSMPY"])

    wcs = WCS(header).celestial

    epoca = Time(
        header["MJD-AVG"],
        format="mjd",
        scale="utc"
    ).jyear

    altura, largura = imagem.shape

    centro = wcs.pixel_to_world(
        (largura - 1) / 2,
        (altura - 1) / 2
    ).icrs

    cantos = wcs.pixel_to_world(
        [0, largura-1, 0, largura-1],
        [0, 0, altura-1, altura-1]
    ).icrs

    raio = centro.separation(cantos).deg.max() + 0.1

    # Consulta ao catálogo Gaia DR3

    destino = PASTA / f"gaia_D{detector}.ecsv"

    if destino.exists():

        tabela = Table.read(destino)

    else:

        consulta = f"""
        SELECT source_id, ra, dec,
               pmra, pmdec, ref_epoch,
               phot_g_mean_mag, ruwe,
               parallax
        FROM gaiadr3.gaia_source
        WHERE phot_g_mean_mag <= 17.5
        AND 1 = CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE(
                'ICRS',
                {centro.ra.deg},
                {centro.dec.deg},
                {raio}
            )
        )
        """

        print("Consultando Gaia DR3...")

        tabela = Gaia.launch_job_async(
            consulta
        ).get_results()

        tabela.write(destino, overwrite=True)

    df = tabela.to_pandas()

    print("Fontes Gaia:", len(df))

    if len(df) < 2:
        return []

    # Correção do movimento próprio

    dt = epoca - df.ref_epoch.fillna(2016)

    referencias = SkyCoord(
        ra=df.ra.to_numpy() * u.deg,
        dec=df.dec.to_numpy() * u.deg
    )

    referencias = referencias.spherical_offsets_by(
        (df.pmra.fillna(0) * dt).to_numpy() * u.mas,
        (df.pmdec.fillna(0) * dt).to_numpy() * u.mas
    )

    # Verificação do isolamento das estrelas

    _, vizinho, _ = referencias.match_to_catalog_sky(
        referencias,
        nthneighbor=2
    )

    gx, gy = wcs.all_world2pix(
        referencias.ra.deg,
        referencias.dec.deg,
        0,
        ra_dec_order=True,
        quiet=True
    )

    selecao = (
        df.phot_g_mean_mag.between(9, 15)
        & df.ruwe.between(0, 1.4)
        & df.pmra.notna()
        & df.pmdec.notna()
        & (df.parallax.abs() < 10)
        & (vizinho.arcsec > 15)
        & (gx > 8)
        & (gy > 8)
        & (gx < largura-9)
        & (gy < altura-9)
    )

    indices = np.flatnonzero(selecao)

    indices = rng.permutation(indices)[:MAX_ESTRELAS]

    print("Estrelas de referência:", len(indices))

    validos = (
        np.isfinite(imagem)
        & np.isfinite(var)
        & (var > 0)
        & ((flags & MASCARA) == 0)
    )

    segura = np.where(validos, imagem, -np.inf)

    maximos = validos & (
        segura == maximum_filter(segura, size=3)
    )

    cx = epsf["XCENTER"]
    cy = epsf["YCENTER"]

    registros = []
    utilizados = set()

    for i in indices:

        x0 = int(round(gx[i]))
        y0 = int(round(gy[i]))

        # Encontrar máximo local próximo à referência

        candidatos = []

        for y in range(y0-2, y0+3):
            for x in range(x0-2, x0+3):

                if not maximos[y, x]:
                    continue

                dist = np.hypot(
                    x - gx[i],
                    y - gy[i]
                )

                if dist <= 1.5:

                    candidatos.append(
                        (imagem[y, x], x, y)
                    )

        if not candidatos:
            continue

        _, x, y = max(candidatos)

        if (x, y) in utilizados:
            continue

        regiao = np.s_[
            y-R:y+R+1,
            x-R:x+R+1
        ]

        patch = imagem[regiao]
        variancia = var[regiao]
        ok = validos[regiao]

        # Excluir núcleos afetados por flags

        if not ok[NUCLEO].all():
            continue

        if ok.sum() < 65:
            continue

        borda = patch[ANEL & ok]

        fundo = np.median(borda)

        snr = (
            imagem[y, x] - fundo
        ) / np.sqrt(var[y, x])

        if snr < 10:
            continue

        zona = np.argmin(
            (cx - x)**2 + (cy - y)**2
        )

        modelo = np.asarray(
            epsf["EPSF"][zona],
            dtype=float
        ).reshape(33, 33)

        ajuste = ajustar(
            patch,
            variancia,
            ok,
            modelo,
            ox,
            oy
        )

        if ajuste is None:
            continue

        discreto, continuo, convergiu = ajuste

        registro = {
            "detector": detector,
            "exposicao": linha.exposicao,
            "gaia_id": str(df.source_id.iloc[i]),
            "x": x,
            "y": y,
            "snr": snr,
            "mag_g": df.phot_g_mean_mag.iloc[i],
            "convergiu": convergiu
        }

        for nome, fit in [
            ("discreto", discreto),
            ("continuo", continuo)
        ]:

            chi, dx, dy = fit

            posicao = wcs.pixel_to_world(
                x + dx,
                y + dy
            ).icrs

            da, dd = referencias[i].spherical_offsets_to(
                posicao
            )

            registro[f"r2_{nome}"] = 1 - chi
            registro[f"ra_{nome}"] = posicao.ra.deg
            registro[f"dec_{nome}"] = posicao.dec.deg
            registro[f"erro_ra_{nome}"] = da.arcsec
            registro[f"erro_dec_{nome}"] = dd.arcsec

            registro[f"erro_{nome}"] = (
                referencias[i].separation(posicao).arcsec
            )

        registros.append(registro)

        utilizados.add((x, y))

    return registros


# Execução nos seis detectores

print("EXPERIMENTO 012")

for linha in catalogo.itertuples():

    detector = int(linha.detector)

    print(f"\nDETECTOR {detector}", flush=True)

    try:

        registros = processar(linha)

        print("Estrelas analisadas:", len(registros))

        if len(registros) < 40:

            status.append({
                "detector": detector,
                "estrelas": len(registros),
                "status": "AMOSTRA INSUFICIENTE"
            })

            resultados.extend(registros)

        else:

            df = pd.DataFrame(registros)

            # Separação entre calibração e teste

            ordem = rng.permutation(len(df))

            metade = len(df) // 2

            df["conjunto"] = "teste"

            df.loc[
                ordem[:metade],
                "conjunto"
            ] = "calibracao"

            cal = df.conjunto == "calibracao"
            teste = df.conjunto == "teste"

            resumo = {
                "detector": detector,
                "n_calibracao": int(cal.sum()),
                "n_teste": int(teste.sum())
            }

            for nome in ["discreto", "continuo"]:

                da = df.loc[
                    cal, f"erro_ra_{nome}"
                ].median()

                dd = df.loc[
                    cal, f"erro_dec_{nome}"
                ].median()

                corrigido = np.hypot(
                    df[f"erro_ra_{nome}"] - da,
                    df[f"erro_dec_{nome}"] - dd
                )

                df[f"corrigido_{nome}"] = corrigido

                resumo[f"offset_ra_{nome}"] = da
                resumo[f"offset_dec_{nome}"] = dd

                resumo[f"mediana_bruta_{nome}"] = (
                    df.loc[teste, f"erro_{nome}"].median()
                )

                resumo[f"mediana_corrigida_{nome}"] = (
                    corrigido[teste].median()
                )

                resumo[f"rmse_{nome}"] = np.sqrt(
                    np.mean(corrigido[teste]**2)
                )

            resumos.append(resumo)

            resultados.extend(
                df.to_dict("records")
            )

            status.append({
                "detector": detector,
                "estrelas": len(df),
                "status": "OK"
            })

    except Exception as erro:

        print("ERRO:", erro)

        status.append({
            "detector": detector,
            "status": str(erro)
        })

    # Salvamento progressivo

    pd.DataFrame(resultados).to_csv(
        PASTA / "experimento_012.csv",
        index=False
    )

    pd.DataFrame(resumos).to_csv(
        PASTA / "experimento_012_resumo.csv",
        index=False
    )

    pd.DataFrame(status).to_csv(
        PASTA / "experimento_012_status.csv",
        index=False
    )


# Visualização científica

resumo = pd.DataFrame(resumos)

if not resumo.empty:

    print("\nRESULTADOS")
    print(resumo.to_string(index=False))

    x = np.arange(len(resumo))

    plt.figure(figsize=(10, 6))

    plt.bar(
        x - 0.18,
        resumo.mediana_corrigida_discreto,
        width=0.36,
        label="Discreto"
    )

    plt.bar(
        x + 0.18,
        resumo.mediana_corrigida_continuo,
        width=0.36,
        label="Contínuo"
    )

    plt.xticks(
        x,
        ["D" + str(d) for d in resumo.detector]
    )

    plt.ylabel("Erro angular mediano (arcsec)")
    plt.xlabel("Detector")
    plt.title("SPHEREx — Validação astrométrica")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        PASTA / "experimento_012.png",
        dpi=300
    )

    plt.show()

print("\nEXPERIMENTO CONCLUÍDO")