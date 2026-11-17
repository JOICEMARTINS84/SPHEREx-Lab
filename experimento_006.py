from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

PASTA = Path(__file__).resolve().parent

D = pd.read_csv(PASTA / 'experimento_005.csv')

G = ['TRANSIENT', 'SEM_TRANSIENT']

# Auditoria dos dados

assert set(D.grupo) == set(G)
assert not D.isna().any().any()
assert not D.duplicated(['x', 'y']).any()

assert np.isfinite(
    D[['x', 'y', 'snr_pixel', 'r2', 'amplitude', 'zona']]
).all().all()

assert D.r2.between(-1e-10, 1 + 1e-10).all()
assert (D.snr_pixel >= 5).all()

print('EXPERIMENTO 006 — Auditoria exploratória')
print('Registros:', len(D))

# Comparação estatística inicial

a, b = [
    D.loc[D.grupo == g, 'r2'].to_numpy()
    for g in G
]

u, p = mannwhitneyu(
    a, b, alternative='two-sided'
)

print('Diferença bruta de médias:', a.mean() - b.mean())
print('Mann-Whitney U:', u)
print('Valor-p nominal:', p)
print('Delta de Cliff:', 2*u/(len(a)*len(b))-1)

print('\nMedianas SNR:')
print(D.groupby('grupo').snr_pixel.median())

# Estratificação e bootstrap espacial

def analisar(limites, tamanho, repeticoes=4000):

    k = len(limites) - 1

    faixa = np.searchsorted(
        limites, D.snr_pixel, side='right'
    ) - 1

    grupo = (
        D.grupo == G[1]
    ).astype(int).to_numpy()

    bloco = (
        (D.x // tamanho).astype(str)
        + ':'
        + (D.y // tamanho).astype(str)
    )

    _, idx = np.unique(
        bloco, return_inverse=True
    )

    nb = idx.max() + 1

    n = np.zeros((nb, k, 2))
    s = np.zeros_like(n)

    np.add.at(n, (idx, faixa, grupo), 1)

    np.add.at(
        s,
        (idx, faixa, grupo),
        D.r2.to_numpy()
    )

    N = n.sum(axis=0)
    S = s.sum(axis=0)

    ok = (N > 0).all(axis=1)

    N = N[ok]
    S = S[ok]
    n = n[:, ok]
    s = s[:, ok]

    # Pesos de sobreposição entre os grupos

    w = N[:, 0] * N[:, 1] / N.sum(axis=1)

    w /= w.sum()

    medias = S / N

    delta = w @ (
        medias[:, 0] - medias[:, 1]
    )

    tabela = pd.DataFrame({

        'snr_min': np.array(limites[:-1])[ok],
        'snr_max': np.array(limites[1:])[ok],

        'n_transient': N[:, 0].astype(int),
        'n_sem_transient': N[:, 1].astype(int),

        'r2_transient': medias[:, 0],
        'r2_sem_transient': medias[:, 1],

        'peso': w
    })

    # Reamostragem por blocos espaciais

    rng = np.random.default_rng(2026)

    boot = []

    for _ in range(repeticoes):

        ii = rng.integers(0, nb, nb)

        nn = n[ii].sum(axis=0)
        ss = s[ii].sum(axis=0)

        if (nn == 0).any():
            continue

        m = ss / nn

        boot.append(
            w @ (m[:, 0] - m[:, 1])
        )

    ci = (
        np.quantile(boot, [.025, .975])
        if boot else [np.nan, np.nan]
    )

    print(f'\nBlocos: {tamanho} pixels')
    print(f'Blocos ocupados: {nb}')
    print(f'Delta ajustado: {delta:.6f}')
    print(f'IC95%: [{ci[0]:.6f}, {ci[1]:.6f}]')
    print(f'Bootstrap: {len(boot)}/{repeticoes}')

    return tabela, {

        'bloco': tamanho,
        'blocos_ocupados': nb,
        'delta': delta,
        'ic_inf': ci[0],
        'ic_sup': ci[1],
        'bootstrap_validos': len(boot),
        'bootstrap_total': repeticoes
    }


# Análise de sensibilidade

resultados = []

configuracoes = [

    (
        'principal',
        [5, 10, 20, 40, 80, 160, np.inf]
    ),

    (
        'sensibilidade',
        [
            5, 7.5, 10, 15, 20, 30, 40,
            60, 80, 120, 160, 200, np.inf
        ]
    )
]

for nome, limites in configuracoes:

    print('\nFAIXAS:', nome)

    for tamanho in [64, 128, 256]:

        tabela, resumo = analisar(
            limites, tamanho
        )

        resultados.append({
            'faixas': nome,
            **resumo
        })

        if tamanho == 128:

            print(tabela.to_string(index=False))

            tabela.to_csv(
                PASTA / f'experimento_006_{nome}_faixas.csv',
                index=False
            )

pd.DataFrame(resultados).to_csv(
    PASTA / 'experimento_006_resumo.csv',
    index=False
)

print('\nEXPERIMENTO CONCLUÍDO')