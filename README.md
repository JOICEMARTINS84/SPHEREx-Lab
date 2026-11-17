# SPHEREx-Lab

**Joice Souza Martins Santos** — Matemática, astrônoma amadora e pesquisadora independente (Brasil).

Investigação exploratória sobre sinais, ruído, morfologia e astrometria em dados públicos do telescópio espacial SPHEREx.

## Organização

- Scripts Python dos experimentos 001 e 003–017, incluindo a variante 012 D6.
- A numeração 002 foi pulada intencionalmente; não corresponde a um experimento realizado.
- `experimento_012_D6.csv` e `resultados_016/dados_individuais.csv`: entradas do Experimento 017.
- `resultados_017/`: métricas, ajustes individuais, auditoria, offsets, bootstrap e metadados.

## Experimento 017

Uma exposição do detector 6, com 130 fontes para calibração e 130 para avaliação. O método de múltiplas células reduziu o erro de 29 fontes e aumentou o de 101. O controle Gaia não representa uma avaliação cega independente.

- [Métricas](resultados_017/metricas.csv)
- [Ajustes individuais](resultados_017/ajustes_individuais.csv)
- [Protocolo e limitações](resultados_017/metadados.json)
- [Código](experimento_017.py)

## Reprodução

Instale numpy, pandas, scipy, matplotlib e astropy em um ambiente Python compatível.

```bash
python experimento_017.py --self-test
python experimento_017.py --csv experimento_012_D6.csv --fase resultados_016/dados_individuais.csv --out reproducoes/017
```

A execução integral exige o FITS original, que pode ser obtido no IRSA (ou fornecido com `--fits`). O script verifica hashes SHA-256 das entradas. Não foram reexecutados todos os pipelines durante a organização do repositório. Os demais arquivos históricos e figuras estão sendo organizados; nem todas as etapas possuem todos os insumos recuperados.

## Fontes e diário

- [Matemática & Contexto](https://matematicacontexto.wpcomstaging.com/)
- [Dados FITS no IRSA](https://irsa.ipac.caltech.edu/ibe/data/spherex/dr1/level2/2026W13_2A/l2b-v27-2026-237/6/level2_2026W13_2A_0238_1D6_spx_l2b-v27-2026-237.fits)
- [Notebook de visualização](https://colab.research.google.com/drive/1PmGdeceVYhbPEFtjJ-uv10mHdCB2urrS) (permissões públicas ainda não verificadas).

Resultados exploratórios, sem revisão por pares. Não demonstram descoberta de novos objetos nem redução da taxa de falsos positivos.
