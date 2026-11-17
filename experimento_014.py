"""SPHEREx-Lab | 014 — auditoria exploratória de associações D6.
Entrada: resultados individuais do 012 D6, catálogo Gaia em cache e FITS original.
Não detecta fontes novas e não mede a taxa real de falsos positivos.
Execução: py -u experimento_014.py [--fits caminho_do_fits]
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import zipfile

import numpy as np
import pandas as pd
import scipy
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import astropy
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
from astropy.utils.data import download_file
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 01 — Protocolo fixado antes desta execução
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'resultados_014'
METHODS = ('discreto', 'continuo')
RADII = (2., 4., 6.15, 10., 15.)  # arcsec; diagnósticos, não limiares validados
SHIFT_RADII = (60., 120.)
ANGLES = tuple(range(0, 360, 45))
LOCAL_RADII = (30., 60.)
G_LIMIT = 17.5
EXPECTED_N = 780443
EXPECTED_SPLIT = (130, 130)
TOL = 0.02  # arcsec; tolerância para reprodução numérica, não precisão científica


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sky(ra, dec):
    return SkyCoord(ra=np.asarray(ra, float)*u.deg,
                    dec=np.asarray(dec, float)*u.deg, frame='icrs')


def xyz(coords):
    return coords.cartesian.xyz.value.T


def chord(arcsec):
    return 2*np.sin(np.deg2rad(np.asarray(arcsec)/3600)/2)


def angular(ch):
    return np.rad2deg(2*np.arcsin(np.clip(ch/2, 0, 1)))*3600


def holm(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    order = finite[np.argsort(p[finite])]
    # Mantém a família de testes planejada, mesmo se algum teste for indefinido.
    out[order] = np.minimum(1, np.maximum.accumulate(p[order]*(len(p)-np.arange(len(order))))) if len(order) else []
    return out


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 3 or len(np.unique(x[ok])) < 2 or len(np.unique(y[ok])) < 2:
        return n, np.nan, np.nan
    r, p = spearmanr(x[ok], y[ok])
    return n, float(r), float(p)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fits', type=Path, help='FITS original da exposição D6; sem argumento usa URL/cache do 010')
    opts = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    print('EXPERIMENTO 014 — AUDITORIA DE ASSOCIAÇÕES D6', flush=True)

    # 02 — Identidade, integridade e proveniência dos insumos
    csv = ROOT / 'experimento_012_D6.csv'
    img_csv = ROOT / 'experimento_010_imagens.csv'
    d = pd.read_csv(csv, dtype={'gaia_id': 'string', 'exposicao': 'string'})
    needed = {'detector','exposicao','gaia_id','conjunto','x','y','snr','mag_g'}
    for m in METHODS:
        needed.update(f'{k}_{m}' for k in ('ra','dec','dx','dy','erro_ra','erro_dec','corrigido','r2'))
    require(needed.issubset(d.columns), f'Colunas ausentes: {sorted(needed-set(d.columns))}')
    require(d.detector.eq(6).all() and d.exposicao.nunique()==1, 'Esperado D6, uma única exposição.')
    exp = str(d.exposicao.iloc[0])
    require(d.gaia_id.str.fullmatch(r'[0-9]+').fillna(False).all(), 'IDs Gaia inválidos ou arredondados.')
    require(not d.gaia_id.duplicated().any() and not d.duplicated(['x','y']).any(), 'IDs ou picos repetidos.')
    require(set(d.conjunto)=={'calibracao','teste'}, 'Divisão original não encontrada.')
    cal = d.loc[d.conjunto=='calibracao'].copy()
    test = d.loc[d.conjunto=='teste'].copy().reset_index(drop=True)
    require((len(cal),len(test))==EXPECTED_SPLIT, f'Divisão inesperada: {len(cal)}/{len(test)}. Confira a versão do CSV.')
    numeric = sorted(needed-{'exposicao','gaia_id','conjunto'})
    require(np.isfinite(d[numeric].to_numpy(float)).all(), 'Dados não finitos no CSV.')
    cache = ROOT / 'gaia_cache_012_v3' / f'D6_{exp}' / 'campo_completo.ecsv'
    require(cache.exists(), f'Cache integral do 012 não encontrado: {cache}. Não use gaia_D6.ecsv antigo.')
    print('Lendo catálogo Gaia DR3 em cache...', flush=True)
    tab = Table.read(cache, format='ascii.ecsv')
    require(str(tab.meta.get('exposicao'))==exp, 'Exposição do cache não coincide.')
    require(len(tab)==EXPECTED_N, f'Cache com {len(tab)} fontes; esperadas {EXPECTED_N}. Confira a edição.')
    require(tab['source_id'].dtype.kind in 'iu', 'source_id do ECSV não é inteiro.')
    g = tab.to_pandas().sort_values('source_id').reset_index(drop=True)
    ids = g.source_id.astype(str).to_numpy()
    require(not g.source_id.duplicated().any(), 'IDs Gaia duplicados no cache.')
    require(np.isfinite(g[['ra','dec','phot_g_mean_mag']].to_numpy(float)).all(), 'Posições ou magnitudes inválidas no cache.')
    require((g.phot_g_mean_mag<=G_LIMIT+1e-6).all(), 'Filtro de magnitude diferente do 012.')
    index_by_id = pd.Index(ids)
    d_idx = index_by_id.get_indexer(d.gaia_id.astype(str))
    ref_idx = index_by_id.get_indexer(test.gaia_id.astype(str))
    require((d_idx>=0).all(), 'IDs originais ausentes no catálogo.')
    require(np.allclose(g.phot_g_mean_mag.to_numpy()[d_idx],d.mag_g.to_numpy(),atol=1e-5,rtol=0), 'Magnitudes não reproduzidas.')
    pm_ok = np.isfinite(g.pmra) & np.isfinite(g.pmdec)
    require(pm_ok.iloc[d_idx].all(), 'Referência original sem movimento próprio completo.')
    require(np.isfinite(g.ref_epoch.iloc[d_idx]).all(), 'Referência original sem época.')

    # 03 — Recuperar a época e WCS do mesmo FITS (sem recorrer a datas aproximadas)
    images = pd.read_csv(img_csv, dtype={'exposicao': str})
    row = images.loc[(images.detector.astype(int)==6)&(images.exposicao==exp)]
    require(len(row)==1, 'Exposição D6 não localizada univocamente no CSV 010.')
    url = str(row.url.iloc[0])
    print('Abrindo FITS da mesma exposição (cache se disponível)...', flush=True)
    fpath = opts.fits.resolve() if opts.fits else Path(download_file(url,cache=True,timeout=300))
    with fits.open(fpath) as f:
        header = f['IMAGE'].header.copy()
    wcs = WCS(header).celestial
    width, height = int(header['NAXIS1']), int(header['NAXIS2'])
    epoch = Time(header['MJD-AVG'],format='mjd',scale='utc').jyear
    center = wcs.pixel_to_world((width-1)/2,(height-1)/2).icrs
    corners = wcs.pixel_to_world([0,width-1,0,width-1],[0,0,height-1,height-1]).icrs
    radius = float(center.separation(corners).deg.max()+0.1)
    require(np.allclose([center.ra.deg,center.dec.deg,radius],
        [tab.meta.get('ra',np.nan),tab.meta.get('dec',np.nan),tab.meta.get('raio',np.nan)],rtol=0,atol=1e-8), 'Geometria FITS/cache divergente.')
    print(f'Exposição {exp}; época {epoch:.9f}; Gaia: {len(g)}; teste: {len(test)}', flush=True)

    # 04 — Mesma aproximação de propagação angular utilizada no 012.
    # Para movimento próprio ausente, repete a hipótese do 012 (zero por componente).
    # Depois repete a busca com apenas movimentos próprios completos.
    dt = epoch - g.ref_epoch.fillna(2016).to_numpy(float)
    ref = sky(g.ra,g.dec).spherical_offsets_by(
        g.pmra.fillna(0).to_numpy(float)*dt*u.mas,
        g.pmdec.fillna(0).to_numpy(float)*dt*u.mas)
    ref_xyz = xyz(ref)
    tree = cKDTree(ref_xyz)
    valid_pm_indices = np.flatnonzero(pm_ok.to_numpy())
    require(len(valid_pm_indices)>=2, 'Catálogo com movimento próprio insuficiente.')
    tree_pm = cKDTree(ref_xyz[valid_pm_indices])

    # 05 — Reproduzir astrometria e congelar calibração anterior
    positions = {}
    offsets = {}
    for m in METHODS:
        orig = sky(d[f'ra_{m}'],d[f'dec_{m}'])
        fits_pos = wcs.pixel_to_world(d.x.to_numpy()+d[f'dx_{m}'].to_numpy(),
                                     d.y.to_numpy()+d[f'dy_{m}'].to_numpy()).icrs
        require(np.max(orig.separation(fits_pos).arcsec)<TOL, f'WCS não reproduziu {m}.')
        a,b = ref[d_idx].spherical_offsets_to(orig)
        require(np.allclose(a.arcsec,d[f'erro_ra_{m}'],atol=TOL,rtol=0) and
                np.allclose(b.arcsec,d[f'erro_dec_{m}'],atol=TOL,rtol=0), f'Época/cache não reproduziram {m}.')
        da = float(cal[f'erro_ra_{m}'].median()); db = float(cal[f'erro_dec_{m}'].median())
        residual = np.hypot(test[f'erro_ra_{m}']-da,test[f'erro_dec_{m}']-db)
        require(np.allclose(residual,test[f'corrigido_{m}'],rtol=0,atol=1e-6),f'Correção original não reproduzida: {m}')
        corrected = sky(test[f'ra_{m}'],test[f'dec_{m}']).spherical_offsets_by(-da*u.arcsec,-db*u.arcsec)
        require(np.allclose(corrected.separation(ref[ref_idx]).arcsec,residual,rtol=0,atol=TOL),f'Correção esférica incompatível: {m}')
        positions[m] = corrected
        offsets[m] = {'ra_arcsec':da,'dec_arcsec':db,'mediana_teste':float(np.median(residual)), 'rmse_teste':float(np.sqrt(np.mean(residual**2)))}
    print('Astrometria anterior reproduzida numericamente.', flush=True)

    # 06 — Indicadores ambientais centrados na referência: não usam o erro residual.
    audited = test.copy()
    ref_test = ref[ref_idx]
    near_dist, _ = tree.query(xyz(ref_test), k=2)
    audited['vizinho_referencia_arcsec'] = angular(near_dist[:,1])
    print(f'Referências com vizinha <=15 arcsec: {(audited.vizinho_referencia_arcsec<=15).sum()}', flush=True)
    for r in LOCAL_RADII:
        n = tree.query_ball_point(xyz(ref_test), chord(r), return_length=True)-1
        audited[f'vizinhas_ref_{int(r)}arcsec'] = n
    # 07 — Competição em torno da posição medida/corrigida
    for m,c in positions.items():
        dist, ix = tree.query(xyz(c),k=2)
        alt_ix = np.where(ix[:,0]==ref_idx,ix[:,1],ix[:,0])
        alt_dist = np.where(ix[:,0]==ref_idx,dist[:,1],dist[:,0])
        target = c.separation(ref_test).arcsec
        audited[f'ra_corr_{m}'] = c.ra.deg; audited[f'dec_corr_{m}'] = c.dec.deg
        audited[f'id_mais_proximo_{m}'] = ids[ix[:,0]]
        audited[f'd1_{m}_arcsec'] = angular(dist[:,0])
        audited[f'd2_{m}_arcsec'] = angular(dist[:,1])
        audited[f'id_alternativo_{m}'] = ids[alt_ix]
        audited[f'g_alternativo_{m}'] = g.phot_g_mean_mag.to_numpy()[alt_ix]
        audited[f'pm_alternativo_completo_{m}'] = pm_ok.to_numpy()[alt_ix]
        audited[f'd_alternativo_{m}_arcsec'] = angular(alt_dist)
        audited[f'd_original_{m}_arcsec'] = target
        audited[f'margem_alt_menos_original_{m}'] = angular(alt_dist)-target
        audited[f'original_mais_proximo_{m}'] = ix[:,0]==ref_idx
        audited[f'alternativa_mais_proxima_{m}'] = (angular(alt_dist)+1e-8<target)
        audited[f'razao_d1_d2_{m}'] = np.divide(angular(dist[:,0]),angular(dist[:,1]),out=np.full(len(c),np.nan),where=dist[:,1]>0)
        _, ipm = tree_pm.query(xyz(c),k=1)
        audited[f'id_mais_proximo_pm_completo_{m}'] = ids[valid_pm_indices[ipm]]
        for r in RADII:
            audited[f'n_{m}_{r:g}arcsec'] = tree.query_ball_point(xyz(c),chord(r),return_length=True)
    audited['metodos_mesmo_id_mais_proximo'] = audited.id_mais_proximo_discreto==audited.id_mais_proximo_continuo
    audited.to_csv(OUT/'associacoes_individuais.csv',index=False)
    # Não transformar proximidade geométrica em rótulo de verdade astronômica.
    summary=[]
    for m in METHODS:
        for r in RADII:
            n = audited[f'n_{m}_{r:g}arcsec']
            summary.append({'metodo':m,'raio_arcsec':r,'n_teste':len(n),
                'sem_candidato':int((n==0).sum()),'um_candidato':int((n==1).sum()),
                'multiplos_candidatos':int((n>=2).sum()),
                'original_dentro_raio':int((audited[f'd_original_{m}_arcsec']<=r).sum()),
                'alternativa_mais_proxima_dentro_raio':int((audited[f'alternativa_mais_proxima_{m}'] & (audited[f'd_alternativo_{m}_arcsec']<=r)).sum())})
    summary=pd.DataFrame(summary)
    summary.to_csv(OUT/'associacoes_por_raio.csv',index=False)

    # 08 — Controles posicionais deslocados, sem fluxos simulados.
    # Subamostra comum a todas as direções e ambos métodos; margem conservadora de 4 pixels
    # e 60 arcsec no cone para minimizar efeitos de borda.
    def inside(c):
        x,y=wcs.world_to_pixel(c)
        return (np.isfinite(x)&np.isfinite(y)&(x>=4)&(y>=4)&(x<=width-5)&(y<=height-5)&
            (center.separation(c).arcsec<=radius*3600-60))
    controls={}; eligible=np.ones(len(test),dtype=bool)
    for m,c in positions.items():
        eligible &= inside(c)
        for dr in SHIFT_RADII:
            for angle in ANGLES:
                shifted=c.directional_offset_by(angle*u.deg,dr*u.arcsec)
                controls[m,dr,angle]=shifted
                eligible &= inside(shifted)
    require(eligible.any(),'Nenhuma posição elegível para todos os controles.')
    control_rows=[]; individual=[]
    for m,c in positions.items():
        for shift,angle in [(0.,0)]+[(r,a) for r in SHIFT_RADII for a in ANGLES]:
            cc=(c if shift==0 else controls[m,shift,angle])[eligible]
            dd, ii=tree.query(xyz(cc),k=1)
            dd=angular(dd)
            for r in RADII:
                ns=tree.query_ball_point(xyz(cc),chord(r),return_length=True)
                control_rows.append({'metodo':m,'deslocamento_arcsec':shift,'angulo_deg':angle,'raio_arcsec':r,
                                     'n_elegivel':len(ns),'n_coincidencias':int((ns>0).sum()),'fracao_coincidencias':float((ns>0).mean()),
                                     'fracao_multiplos':float((ns>=2).mean())})
            for gid,near,idx in zip(test.gaia_id[eligible],dd,ii):
                individual.append({'metodo':m,'gaia_id_original':gid,'deslocamento_arcsec':shift,'angulo_deg':angle,
                                   'id_mais_proximo':ids[idx],'d1_arcsec':near})
    control_df=pd.DataFrame(control_rows)
    control_df.to_csv(OUT/'controles_por_direcao.csv',index=False)
    pd.DataFrame(individual).to_csv(OUT/'controles_individuais.csv',index=False)
    ctrl_summary=control_df.groupby(['metodo','deslocamento_arcsec','raio_arcsec'],as_index=False).agg(
        n_direcoes=('angulo_deg','count'),n_elegivel=('n_elegivel','first'),media=('fracao_coincidencias','mean'),
        minimo=('fracao_coincidencias','min'),maximo=('fracao_coincidencias','max'))
    ctrl_summary.to_csv(OUT/'controles_resumo.csv',index=False)
    print(f'Controles posicionais: {eligible.sum()} de {len(test)} estrelas na subamostra comum.',flush=True)

    # 09 — Correlações exploratórias (indicadores centrados na referência, não no centro ajustado)
    correlations=[]
    for v in ('vizinho_referencia_arcsec','vizinhas_ref_30arcsec','vizinhas_ref_60arcsec'):
        for m in METHODS:
            n,r,p=spearman(audited[v],audited[f'corrigido_{m}'])
            correlations.append({'indicador':v,'metodo':m,'n':n,'rho':r,'p_nominal':p})
    corr=pd.DataFrame(correlations)
    corr['p_holm_nominal']=holm(corr.p_nominal)
    corr.to_csv(OUT/'correlacoes_ambiente.csv',index=False)

    # 10 — Figuras e metadados de reprodução
    for m in METHODS:
        plt.figure(figsize=(7,5))
        x=audited[f'd_original_{m}_arcsec']; y=audited[f'd_alternativo_{m}_arcsec']
        plt.scatter(x,y,s=18,alpha=.75)
        top=float(max(x.max(),y.max()))
        plt.plot([0,top],[0,top],'--',label='Distâncias iguais')
        plt.xlabel('Distância à referência original (arcsec)'); plt.ylabel('Distância à melhor alternativa (arcsec)')
        plt.title(f'D6 — auditoria posicional ({m})'); plt.legend(); plt.tight_layout()
        plt.savefig(OUT/f'distancias_{m}.png',dpi=180); plt.close()
        plt.figure(figsize=(7,5))
        for shift in (0.,)+SHIFT_RADII:
            s=ctrl_summary[(ctrl_summary.metodo==m)&(ctrl_summary.deslocamento_arcsec==shift)].sort_values('raio_arcsec')
            plt.plot(s.raio_arcsec,s.media,marker='o',label='Original' if shift==0 else f'Deslocado {shift:g}″')
            if shift: plt.fill_between(s.raio_arcsec.to_numpy(),s.minimo.to_numpy(),s.maximo.to_numpy(),alpha=.15)
        plt.xlabel('Raio de associação (arcsec)'); plt.ylabel('Fração com pelo menos um candidato'); plt.ylim(0,1.04)
        plt.title(f'D6 — coincidências posicionais ({m})'); plt.legend(); plt.tight_layout()
        plt.savefig(OUT/f'controles_{m}.png',dpi=180); plt.close()
    counts={m:{'original_mais_proximo':int(audited[f'original_mais_proximo_{m}'].sum()),
               'alternativa_estritamente_mais_proxima':int(audited[f'alternativa_mais_proxima_{m}'].sum()),
               'mudanca_ao_restringir_pm':int((audited[f'id_mais_proximo_{m}']!=audited[f'id_mais_proximo_pm_completo_{m}']).sum())} for m in METHODS}
    metadata={'experimento':'014','escopo':'exploratório D6, uma exposição; posições condicionadas ao Gaia',
        'exposicao':exp,'mjd_avg':float(header['MJD-AVG']),'epoca_jyear_utc':float(epoch),'url_fits':url,
        'n_gaia':len(g),'n_gaia_pm_incompleto':int((~pm_ok).sum()),'limite_g':G_LIMIT,
        'n_calibracao':len(cal),'n_teste':len(test),'n_controle_comum':int(eligible.sum()),
        'raios_arcsec':RADII,'raios_deslocamento_arcsec':SHIFT_RADII,'angulos_deg':ANGLES,'raios_densidade_arcsec':LOCAL_RADII,
        'tolerancia_reproducao_arcsec':TOL,'offsets':offsets,'contagens':counts,
        'metodos_id_mais_proximo_diferente':int((~audited.metodos_mesmo_id_mais_proximo).sum()),
        'sha256':{str(p.name):sha256(p) for p in (csv,img_csv,cache,fpath,Path(__file__))},
        'versoes':{'python':sys.version,'plataforma':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,
                   'scipy':scipy.__version__,'astropy':astropy.__version__,'matplotlib':matplotlib.__version__},
        'limites':['Gaia G<=17.5, não é catálogo completo de fontes infravermelhas','A seleção anterior foi guiada pelo Gaia e exigiu isolamento e SNR; não estima taxa de falsos positivos.',
                   'Controles deslocados não amostram detecções nem quantificam a taxa de erro real; direções não são independentes.',
                   'p-valores nominais, dependência espacial e confusão não modeladas; nenhuma inferência confirmatória.',
                   'Propagação de primeira ordem, sem paralaxe anual nem velocidade radial; PM ausente aproximado por zero por componente.']}
    (OUT/'metadados.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nASSOCIAÇÕES POR RAIO\n'+summary.to_string(index=False))
    print('\nCONTROLES\n'+ctrl_summary.to_string(index=False))
    print('\nCORRELAÇÕES EXPLORATÓRIAS\n'+corr.to_string(index=False))
    print('\nCONTAGENS\n'+json.dumps(counts,indent=2,ensure_ascii=False))
    archive=ROOT/'experimento_014_resultados.zip'
    # Lista explícita para não incluir resultados antigos ou arquivos estranhos.
    files=['associacoes_individuais.csv','associacoes_por_raio.csv','controles_por_direcao.csv','controles_individuais.csv',
           'controles_resumo.csv','correlacoes_ambiente.csv','metadados.json']+[f'{stem}_{m}.png' for stem in ('distancias','controles') for m in METHODS]
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in files: z.write(OUT/name,arcname=name)
        z.write(Path(__file__),arcname=Path(__file__).name)
    print(f'\nEXPERIMENTO 014 CONCLUÍDO — {archive}',flush=True)

if __name__=='__main__':
    main()
