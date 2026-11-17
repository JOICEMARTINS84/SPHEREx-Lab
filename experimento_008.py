from pathlib import Path
import numpy as np
import pandas as pd

from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u

PASTA = Path(__file__).resolve().parent

dados = pd.read_csv(
    PASTA / "experimento_007.csv"
)

resultados = []

print("EXPERIMENTO 008 — GAIA DR3")

for linha in dados.itertuples():

    print("\nInvestigando:", linha.candidato)

    centro = SkyCoord(
        ra=linha.ra * u.deg,
        dec=linha.dec * u.deg
    )

    epoca = Time(
        linha.mjd_obs,
        format="mjd",
        scale="utc"
    ).jyear

    consulta = f"""
    SELECT source_id, ra, dec,
           pmra, pmdec, ref_epoch,
           phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS',
            {linha.ra},
            {linha.dec},
            {60/3600}
        )
    )
    """

    try:

        tabela = Gaia.launch_job(
            consulta
        ).get_results()

    except Exception as erro:

        print("Falha na consulta:", erro)
        continue

    encontrados = []

    for estrela in tabela:

        posicao = SkyCoord(
            ra=float(estrela["ra"]) * u.deg,
            dec=float(estrela["dec"]) * u.deg
        )

        corrigido = False

        pmra = estrela["pmra"]
        pmdec = estrela["pmdec"]

        if (
            not np.ma.is_masked(pmra)
            and not np.ma.is_masked(pmdec)
            and np.isfinite(pmra)
            and np.isfinite(pmdec)
        ):

            dt = epoca - float(
                estrela["ref_epoch"]
            )

            posicao = posicao.spherical_offsets_by(
                pmra * dt * u.mas,
                pmdec * dt * u.mas
            )

            corrigido = True

        distancia = centro.separation(
            posicao
        ).arcsec

        if distancia <= 30:

            encontrados.append({
                "candidato": linha.candidato,
                "gaia_id": str(estrela["source_id"]),
                "separacao_arcsec": distancia,
                "mag_g": estrela["phot_g_mean_mag"],
                "movimento_corrigido": corrigido
            })

    encontrados.sort(
        key=lambda x: x["separacao_arcsec"]
    )

    resultados.extend(encontrados)

    print("Fontes próximas:", len(encontrados))

    for fonte in encontrados[:5]:
        print(fonte)

pd.DataFrame(
    resultados,
    columns=[
        "candidato",
        "gaia_id",
        "separacao_arcsec",
        "mag_g",
        "movimento_corrigido"
    ]
).to_csv(
    PASTA / "experimento_008.csv",
    index=False
)

print("\nEXPERIMENTO CONCLUÍDO")