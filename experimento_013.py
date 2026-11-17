from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

PASTA = Path(__file__).resolve().parent
ARQUIVO = PASTA / 'experimento_012_D6.csv'
SAIDA = PASTA / 'resultados_013'
SAIDA.mkdir(exist_ok=True)

# Preservar IDs Gaia como texto, evitando perda de precisão.
dados = pd.read_csv(ARQUIVO, dtype={'gaia_id': str})
cal = dados.loc[dados.conjunto == 'calibracao'].copy()
t = dados.loc[dados.conjunto == 'teste'].copy()
assert len(cal) and len(t), 'Conjuntos de calibração e teste necessários.'
assert not dados.duplicated(['detector', 'exposicao', 'gaia_id']).any()
assert set(cal.gaia_id).isdisjoint(t.gaia_id), 'Fontes repetidas entre conjuntos.'
for metodo in ['discreto', 'continuo']:
    for eixo in ['ra', 'dec']:
        col = f'erro_{eixo}_{metodo}'
        t[f'res_{eixo}_{metodo}'] = t[col] - cal[col].median()
    rec = np.hypot(t[f'res_ra_{metodo}'], t[f'res_dec_{metodo}'])
    assert np.allclose(rec, t[f'corrigido_{metodo}']), 'Correção não reproduzida.'
    t[f'corrigido_{metodo}'] = rec

t['ganho'] = t.corrigido_discreto - t.corrigido_continuo
t['delta_r2'] = t.r2_continuo - t.r2_discreto
t['no_limite'] = np.maximum(t.dx_continuo.abs(), t.dy_continuo.abs()) >= 0.499

def resumo(g):
    return pd.Series({'n':len(g), 'mediana_discreto':g.corrigido_discreto.median(),
        'mediana_continuo':g.corrigido_continuo.median(),
        'mediana_ganho':g.ganho.median(), 'rmse_continuo':np.sqrt(np.mean(g.corrigido_continuo**2)),
        'melhoraram':(g.ganho>0).sum(), 'pioraram':(g.ganho<0).sum()})

# Exploração pós-hoc: quartis definidos na própria amostra de teste.
variaveis = ['snr','mag_g','x','y','r2_discreto','r2_continuo','delta_r2']
cor = []
for v in variaveis:
    for alvo in ['corrigido_continuo','ganho']:
        rho,p = spearmanr(t[v],t[alvo])
        cor.append({'variavel':v,'alvo':alvo,'rho':rho,'p_bruto':p})
corr = pd.DataFrame(cor)
# Holm para as 14 correlações exploratórias.
ordem = np.argsort(corr.p_bruto.to_numpy())
ps = corr.p_bruto.to_numpy()[ordem]
pa = np.empty(len(ps)); pa[ordem] = np.minimum(1,np.maximum.accumulate(ps*(len(ps)-np.arange(len(ps)))))
corr['p_holm'] = pa
corr.to_csv(SAIDA/'correlacoes.csv',index=False)

quadros = []
for v in ['snr','mag_g','r2_continuo','x','y']:
    grupos = pd.qcut(t[v],4,duplicates='drop')
    for intervalo,g in t.groupby(grupos,observed=True):
        quadros.append({'variavel':v,'grupo':str(intervalo),**resumo(g).to_dict()})
for limite,g in t.groupby('no_limite'):
    quadros.append({'variavel':'no_limite','grupo':str(limite),**resumo(g).to_dict()})
quadros = pd.DataFrame(quadros)
quadros.to_csv(SAIDA/'grupos.csv',index=False)
t.to_csv(SAIDA/'estrelas_teste.csv',index=False)
t.nlargest(20,'corrigido_continuo').to_csv(SAIDA/'20_maiores_erros.csv',index=False)

# Um gráfico por diagnóstico.
for v,rotulo in [('snr','SNR do pico'),('mag_g','Magnitude G (Gaia)'),('r2_continuo','R² ponderado do ajuste contínuo')]:
    fig,ax=plt.subplots(figsize=(8,5))
    ax.scatter(t[v], t.corrigido_continuo,alpha=.65)
    ax.set(xlabel=rotulo,ylabel='Erro angular corrigido (arcsec)',title='D6 — erro residual e '+rotulo)
    if v=='snr': ax.set_xscale('log')
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(SAIDA/f'erro_{v}.png',dpi=180); plt.close(fig)
fig,ax=plt.subplots(figsize=(8,7))
p = ax.scatter(t.x,t.y,c=t.corrigido_continuo,cmap='viridis',s=32)
ax.set(xlabel='x (pixel)',ylabel='y (pixel)',title='D6 — distribuição espacial dos erros'); ax.set_aspect('equal')
fig.colorbar(p,ax=ax,label='Erro angular corrigido (arcsec)'); fig.tight_layout(); fig.savefig(SAIDA/'mapa_residuos.png',dpi=180); plt.close(fig)
fig,ax=plt.subplots(figsize=(8,5))
ax.scatter(t.delta_r2,t.ganho,alpha=.65); ax.axhline(0,ls='--',color='gray')
ax.set(xlabel='R² contínuo − discreto',ylabel='Erro discreto − contínuo (arcsec)',title='D6 — qualidade do ajuste e ganho astrométrico')
ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(SAIDA/'ganho_r2.png',dpi=180); plt.close(fig)
print('EXPERIMENTO 013 — diagnóstico exploratório D6')
print('Calibração:',len(cal),'| Teste:',len(t))
print(resumo(t).to_string()); print('\nCORRELAÇÕES\n',corr.to_string(index=False)); print('\nGRUPOS\n',quadros.to_string(index=False))
print('\nArquivos:',SAIDA)
