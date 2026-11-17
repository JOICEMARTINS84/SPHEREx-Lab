from pathlib import Path

import re
import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u
from astroquery.ipac.irsa import Irsa

PASTA = Path(__file__).resolve().parent

rng = np.random.default_rng(2026)

registros = []
falhas = []

# Seleção de regiões em coordenadas galácticas

longitudes = [0, 90, 180, 270]
latitudes = [-45, 0, 45]

inicio = Time("2025-05-01").mjd

print("EXPERIMENTO 009 — SPHEREx DR1")

for l in longitudes:
    for b in latitudes:

        regiao = f"L{l}_B{b}"

        coord = SkyCoord(
            l=l * u.deg,
            b=b * u.deg,
            frame="galactic"
        ).icrs

        print("\nConsultando:", regiao)

        try:

            tabela = Irsa.query_sia(
                pos=(coord, 1 * u.arcsec),
                collection=["spherex_dr1"]
            )

        except Exception as erro:

            print("Erro:", erro)
            falhas.append(regiao)
            continue

        print("Imagens encontradas:", len(tabela))

        for linha in tabela:

            url = str(linha["access_url"])
            banda = str(linha["energy_bandpassname"])

            try:
                mjd = float(linha["t_min"])
            except (ValueError, TypeError):
                continue

            if not np.isfinite(mjd) or mjd < inicio:
                continue

            if "/dr1/level2/" not in url:
                continue

            # Identificação da exposição

            padrao = re.search(
                r"level2_(.+?)_([0-9]+)_[0-9]D([1-6])_",
                url
            )

            if padrao is None:
                continue

            periodo, numero, detector = padrao.groups()

            exposicao = f"{periodo}_{numero}"

            registros.append({

                "regiao": regiao,
                "l_gal": l,
                "b_gal": b,

                "ra": float(linha["s_ra"]),
                "dec": float(linha["s_dec"]),

                "mjd": mjd,
                "banda": banda,
                "detector": int(detector),

                "exposicao": exposicao,
                "url": url

            })

# Organização dos metadados

if not registros:
    raise RuntimeError("Nenhuma imagem encontrada.")

df = pd.DataFrame(registros)

df = df.drop_duplicates("url")

df = df.sort_values("url").reset_index(drop=True)

df.to_csv(
    PASTA / "experimento_009_catalogo.csv",
    index=False
)

# Amostragem estratificada por detector

amostra = []
utilizados = set()

for detector in range(1, 7):

    quantidade = 17 if detector <= 4 else 16

    dados = df[
        df.detector == detector
    ]

    grupos = {}

    for regiao, grupo in dados.groupby("regiao"):

        indices = rng.permutation(
            grupo.index.to_numpy()
        )

        grupos[regiao] = list(indices)

    selecionados = 0

    while selecionados < quantidade:

        progresso = False

        for regiao in rng.permutation(
            sorted(grupos)
        ):

            while grupos[regiao]:

                indice = grupos[regiao].pop()

                linha = df.loc[indice]

                if linha.exposicao in utilizados:
                    continue

                utilizados.add(linha.exposicao)

                amostra.append(linha.to_dict())

                selecionados += 1
                progresso = True

                break

            if selecionados >= quantidade:
                break

        if not progresso:
            break

    print(
        f"Detector {detector}: "
        f"{selecionados}/{quantidade}"
    )

final = pd.DataFrame(amostra)

final.to_csv(
    PASTA / "experimento_009_amostra.csv",
    index=False
)

# Relatório experimental

print("\nRESULTADOS")
print("Imagens disponíveis:", len(df))
print("Imagens selecionadas:", len(final))
print("Falhas nas consultas:", falhas)

if not final.empty:

    print("\nDistribuição por detector:")
    print(final.detector.value_counts().sort_index())

    print("\nDistribuição por região:")
    print(final.regiao.value_counts())

print("\nEXPERIMENTO CONCLUÍDO")