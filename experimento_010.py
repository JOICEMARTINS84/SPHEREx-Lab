from pathlib import Path

import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.utils.data import download_file

PASTA = Path(__file__).resolve().parent

catalogo = pd.read_csv(
    PASTA / "experimento_009_catalogo.csv"
)

amostra = pd.read_csv(
    PASTA / "experimento_009_amostra.csv"
)

# Separação por regiões celestes

desenvolvimento = [
    "L0_B-45", "L0_B0", "L0_B45",
    "L180_B0", "L180_B45", "L270_B0"
]

validacao = [
    "L180_B-45", "L270_B-45", "L90_B0"
]

amostra["conjunto"] = np.select(
    [
        amostra.regiao.isin(desenvolvimento),
        amostra.regiao.isin(validacao)
    ],
    ["desenvolvimento", "validacao"],
    default="teste"
)

amostra.to_csv(
    PASTA / "divisao_amostra.csv",
    index=False
)

# Imagens adicionais para auditoria

disponiveis = catalogo[
    catalogo.regiao.isin(desenvolvimento)
    & ~catalogo.exposicao.isin(amostra.exposicao)
    & (catalogo.exposicao != "2025W18_1B_0074")
]

selecionadas = []
utilizadas = set()

for detector in range(1, 7):

    grupo = disponiveis[
        (disponiveis.detector == detector)
        & ~disponiveis.exposicao.isin(utilizadas)
    ].sort_values("url")

    linha = grupo.sample(
        n=1,
        random_state=2026 + detector
    ).iloc[0]

    selecionadas.append(linha)

    utilizadas.add(linha.exposicao)

pd.DataFrame(selecionadas).to_csv(
    PASTA / "experimento_010_imagens.csv",
    index=False
)

# Auditoria instrumental

resultados = []
indicadores = []

print("EXPERIMENTO 010")

for linha in selecionadas:

    detector = int(linha.detector)

    print(f"\nANALISANDO DETECTOR {detector}")

    try:

        arquivo = download_file(
            linha.url,
            cache=True,
            timeout=300
        )

        with fits.open(arquivo) as dados:

            dados.info()

            imagem = dados["IMAGE"].data
            flags = dados["FLAGS"].data.astype(np.uint32)
            variancia = dados["VARIANCE"].data
            zodi = dados["ZODI"].data

            epsf = dados["EPSF"].data
            header = dados["EPSF"].header

            modelos = np.asarray(epsf["EPSF"])

            ox = int(header["OVSMPX"])
            oy = int(header["OVSMPY"])

            nx = len(np.unique(epsf["BINX"]))
            ny = len(np.unique(epsf["BINY"]))

            # Validação das matrizes

            dimensoes_ok = all(
                matriz.shape == (2040, 2040)
                for matriz in [
                    imagem, flags, variancia, zodi
                ]
            )

            validos = (
                np.isfinite(imagem)
                & np.isfinite(zodi)
                & np.isfinite(variancia)
                & (variancia > 0)
            )

            # Validação dos modelos ePSF

            modelos_validos = np.isfinite(
                modelos.reshape(len(epsf), -1)
            ).all(axis=1)

            grade_esperada = (
                (11, 41) if detector == 3
                else (21, 21)
            )

            epsf_ok = (
                (nx, ny) == grade_esperada
                and len(epsf) == nx * ny
                and modelos[0].size == 1089
                and (ox, oy) == (5, 5)
                and modelos_validos.all()
            )

            # Coordenadas celestes

            wcs = WCS(
                dados["IMAGE"].header
            ).celestial

            centro = wcs.pixel_to_world(
                1019.5, 1019.5
            ).icrs

            # Coordenadas espectrais

            espectral = WCS(
                dados["IMAGE"].header,
                fobj=dados,
                key="W"
            )

            espectral.sip = None

            onda, banda = espectral.pixel_to_world(
                1019.5, 1019.5
            )

            comprimento = onda.to_value(u.um)

            # Frequência dos indicadores

            for bit in range(32):

                quantidade = np.count_nonzero(
                    flags & (1 << bit)
                )

                indicadores.append({
                    "detector": detector,
                    "bit": bit,
                    "pixels": quantidade,
                    "percentual":
                        100 * quantidade / flags.size
                })

            resultado = {
                "detector": detector,
                "exposicao": linha.exposicao,
                "dimensoes_ok": dimensoes_ok,
                "pixels_validos": int(validos.sum()),
                "modelos_epsf": len(epsf),
                "modelos_validos": int(
                    modelos_validos.sum()
                ),
                "zonas_x": nx,
                "zonas_y": ny,
                "oversampling_x": ox,
                "oversampling_y": oy,
                "epsf_ok": epsf_ok,
                "ra_central": centro.ra.deg,
                "dec_central": centro.dec.deg,
                "lambda_central_um": comprimento,
                "status": "OK" if (
                    dimensoes_ok and epsf_ok
                ) else "REVISAR"
            }

            resultados.append(resultado)

            for chave, valor in resultado.items():
                print(f"{chave}: {valor}")

    except Exception as erro:

        resultados.append({
            "detector": detector,
            "status": "ERRO",
            "erro": str(erro)
        })

        print("ERRO:", erro)

    # Salvamento progressivo

    pd.DataFrame(resultados).to_csv(
        PASTA / "experimento_010_resumo.csv",
        index=False
    )

    pd.DataFrame(indicadores).to_csv(
        PASTA / "experimento_010_flags.csv",
        index=False
    )

print("\nEXPERIMENTO CONCLUÍDO")