"""SPHEREx-Lab 016 — diagnóstico exploratório da fase subpixel.
py -u experimento_016.py
py -u experimento_016.py --somente-csv (apenas parte A)
"""
from pathlib import Path
import argparse, hashlib, json, sys, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 01. Protocolo e identidade dos dados
ROOT = Path(__file__).resolve().parent
EXP = '2026W13_2A_0238'
CSV_HASH = 'c71a632e01b95c67b0ae1027278d24df1b70e575bc862d4f59ee6f672678a0c0'
GAIA_HASH = '50156bdd2241aae40af07a6cc2c6d802fe139da5310ce6b371b387ad318076da'
IMG_HASH = '912fd53dd2bdfee1a78b2dd45bf63d1f3ebf9034978cc474479e09a504b11b24'
FITS_HASH = '682a1380468124627784520960a176899b036d0d262422fae8737d840a49fe90'
METHODS = ('discreto','continuo')
SIZE, BUFFER, LIMIT = 2040, 64, .499

def require(ok,msg):
    if not ok: raise ValueError(msg)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def load(p):
    require(sha(p)==CSV_HASH,'CSV 012 diferente do auditado no 014.')
    d=pd.read_csv(p,dtype={'gaia_id':str,'exposicao':str})
    require(len(d)==260 and d.detector.eq(6).all() and d.exposicao.eq(EXP).all(),'Identidade inválida.')
    require(d.conjunto.value_counts().to_dict()=={'calibracao':130,'teste':130},'Divisão inválida.')
    require(not d.gaia_id.duplicated().any() and not d.duplicated(['x','y']).any(),'Duplicatas.')
    require(np.isfinite(d.select_dtypes(include='number')).all().all() and (d.snr>0).all(),'Valores inválidos.')
    d['ganho']=d.corrigido_discreto-d.corrigido_continuo
    d['delta_r2']=d.r2_continuo-d.r2_discreto
    require(d.delta_r2.min()>=-1e-9,'R2 incompatível.')
    for m in METHODS:
        require(d[[f'dx_{m}',f'dy_{m}']].abs().max().max()<=.500001,'Deslocamento inválido.')
        for axis in ['ra','dec']:
            k=f'erro_{axis}_{m}'
            d[f'res_{axis}_{m}']=d[k]-d.loc[d.conjunto=='calibracao',k].median()
        require(np.allclose(np.hypot(d[f'res_ra_{m}'],d[f'res_dec_{m}']),d[f'corrigido_{m}'],atol=1e-8,rtol=0),'Erro corrigido não reproduzido.')
    d['no_limite']=d[['dx_continuo','dy_continuo']].abs().max(axis=1)>=LIMIT
    return d

# 02. Parte A: diagnóstico da saída do algoritmo, não da fase verdadeira

def metrics(z):
    e=np.linalg.norm(z,axis=1) if np.ndim(z)==2 else np.asarray(z)
    return {'n':len(e),'mediana':float(np.median(e)),'rmse':float(np.sqrt(np.mean(e**2)))}

def part_a(d,out):
    rows=[]
    for (s,b),g in d.groupby(['conjunto','no_limite']):
        rows.append({'conjunto':s,'no_limite':bool(b),'n':len(g),'mediana_snr':g.snr.median(),'mediana_ganho':g.ganho.median(),'melhoraram':int((g.ganho>0).sum()),'mediana_delta_r2':g.delta_r2.median(),**{f'{k}_{m}':v for m in METHODS for k,v in metrics(g[f'corrigido_{m}']).items() if k!='n'}})
    a=pd.DataFrame(rows);a.to_csv(out/'limites_otimizacao.csv',index=False)
    t=d[d.conjunto=='teste'];fig,ax=plt.subplots(figsize=(7,6))
    sc=ax.scatter(t.dx_continuo,t.dy_continuo,c=t.corrigido_continuo,cmap='viridis');fig.colorbar(sc,ax=ax,label='Erro corrigido (arcsec)')
    ax.set(xlim=(-.52,.52),ylim=(-.52,.52),xlabel='dx ajustado (pixel)',ylabel='dy ajustado (pixel)',title='D6 — fases estimadas; amostra de teste',aspect='equal');fig.tight_layout();fig.savefig(out/'fase_estimada.png',dpi=180);plt.close(fig)
    print('\nPARTE A — limites do ajuste\n',a.to_string(index=False),flush=True)
    return ['limites_otimizacao.csv','fase_estimada.png']

# 03. Parte B: Gaia propagado para a época do FITS, mesma regra do 012

def reference(d,opts,meta):
    import astropy
    from astropy.io import fits
    from astropy.table import Table
    from astropy.wcs import WCS
    from astropy.time import Time
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from astropy.utils.data import download_file
    cache=ROOT/'gaia_cache_012_v3'/f'D6_{EXP}'/'campo_completo.ecsv'
    img=ROOT/'experimento_010_imagens.csv'
    require(sha(cache)==GAIA_HASH,'Cache diferente do auditado no 014.')
    require(sha(img)==IMG_HASH,'Catálogo 010 diferente do auditado no 014.')
    catalog=pd.read_csv(img,dtype={'exposicao':str});row=catalog[(catalog.detector==6)&(catalog.exposicao==EXP)]
    require(len(row)==1,'Exposição não encontrada univocamente.')
    url=str(row.url.iloc[0]);fpath=opts.fits or Path(download_file(url,cache=True,timeout=300))
    require(sha(fpath)==FITS_HASH,'FITS diferente do auditado no 014; não substituir por outra versão.')
    with fits.open(fpath) as f: h=f['IMAGE'].header.copy()
    w=WCS(h).celestial
    require((h['NAXIS1'],h['NAXIS2'])==(SIZE,SIZE),'Dimensões inesperadas.')
    epoch=Time(h['MJD-AVG'],format='mjd',scale='utc').jyear
    tab=Table.read(cache,format='ascii.ecsv')
    require(len(tab)==780443 and str(tab.meta.get('exposicao'))==EXP and tab['source_id'].dtype.kind in 'iu','Cache incompatível.')
    g=tab.to_pandas();require(not g.source_id.duplicated().any(),'IDs repetidos no Gaia.')
    g=g.set_index(g.source_id.astype(str)).loc[d.gaia_id].reset_index(drop=True)
    require(np.isfinite(g[['ra','dec','pmra','pmdec','ref_epoch']]).all().all(),'Referências incompletas.')
    require(np.allclose(g.phot_g_mean_mag,d.mag_g,rtol=0,atol=1e-5),'Magnitude divergente.')
    dt=epoch-g.ref_epoch.to_numpy();ref=SkyCoord(ra=g.ra.to_numpy()*u.deg,dec=g.dec.to_numpy()*u.deg).spherical_offsets_by(g.pmra.to_numpy()*dt*u.mas,g.pmdec.to_numpy()*dt*u.mas)
    for m in METHODS:
        fitpos=w.pixel_to_world(d.x.to_numpy()+d[f'dx_{m}'].to_numpy(),d.y.to_numpy()+d[f'dy_{m}'].to_numpy()).icrs
        orig=SkyCoord(ra=d[f'ra_{m}'].to_numpy()*u.deg,dec=d[f'dec_{m}'].to_numpy()*u.deg)
        require(float(orig.separation(fitpos).arcsec.max())<.02,'WCS divergente.')
        a,b=ref.spherical_offsets_to(orig)
        require(np.allclose(a.arcsec,d[f'erro_ra_{m}'],rtol=0,atol=.02) and np.allclose(b.arcsec,d[f'erro_dec_{m}'],rtol=0,atol=.02),'Gaia/época divergente.')
    gx,gy=w.world_to_pixel(ref)
    require(np.isfinite([gx,gy]).all(),'Projeção WCS não finita.')
    require(((gx>=0)&(gx<SIZE)&(gy>=0)&(gy<SIZE)).all(),'Referência fora da imagem.')
    d['gaia_x'],d['gaia_y']=gx,gy
    d['fase_x']=phase(gx);d['fase_y']=phase(gy)
    d['dist_borda']=.5-np.maximum(abs(d.fase_x),abs(d.fase_y))
    d['pico_mesmo_pixel_gaia']=(np.floor(gx+.5)==d.x)&(np.floor(gy+.5)==d.y)
    meta.update({'astropy':astropy.__version__,'epoca_jyear':float(epoch),'mjd_avg':float(h['MJD-AVG']),'url_fits':url,'sha256_fits':sha(fpath),'sha256_gaia':sha(cache),'sha256_catalogo_010':sha(img)})
    return d

def phase(x):return (np.asarray(x)+.5)%1-.5

# 04. Modelos diagnósticos aninhados: controles espaciais e intensidade;
# fase periódica (seno e cosseno em cada eixo), sem usar dx/dy ajustados.

def design(d,with_phase):
    x=(d.x.to_numpy()-1019.5)/1020;y=(d.y.to_numpy()-1019.5)/1020
    a=[np.ones(len(d)),x,y,np.log10(d.snr.to_numpy())-2,(d.mag_g.to_numpy()-12)/3]
    if with_phase:
        for k in ('fase_x','fase_y'):
            p=2*np.pi*d[k].to_numpy();a.extend([np.sin(p),np.cos(p)])
    return np.column_stack(a)

def fit(train,valid,m,with_phase):
    X=design(train,with_phase);Y=train[[f'erro_ra_{m}',f'erro_dec_{m}']].to_numpy()
    require(len(X)>=3*X.shape[1] and np.linalg.matrix_rank(X)==X.shape[1] and np.linalg.cond(X)<1e4,'Ajuste insuficiente ou instável.')
    b=np.linalg.lstsq(X,Y,rcond=None)[0]
    z=valid[[f'erro_ra_{m}',f'erro_dec_{m}']].to_numpy()-design(valid,with_phase)@b
    return z

def spatial(cal,val,m,with_phase,k):
    cell=SIZE/k;ix=np.floor(val.x.to_numpy()/cell).astype(int);iy=np.floor(val.y.to_numpy()/cell).astype(int)
    group=ix+k*iy;res=np.full((len(val),2),np.nan)
    for g in np.unique(group):
        a,b=g%k,g//k
        keep=~((cal.x>=a*cell-BUFFER)&(cal.x<(a+1)*cell+BUFFER)&(cal.y>=b*cell-BUFFER)&(cal.y<(b+1)*cell+BUFFER))
        res[group==g]=fit(cal.loc[keep],val.loc[group==g],m,with_phase)
    require(np.isfinite(res).all(),'Predições incompletas.')
    return res

# 05. Avaliação e gráficos. Não selecionar modelo usando o teste.

def part_b(d,out):
    cal=d[d.conjunto=='calibracao'].reset_index(drop=True);t=d[d.conjunto=='teste'].reset_index(drop=True)
    rows=[];preds=[]
    for m in METHODS:
        for ph in (False,True):
            name='controles_mais_fase' if ph else 'controles'
            for kind,frame,z in [('cv_3x3',cal,spatial(cal,cal,m,ph,3)),('cv_2x2',cal,spatial(cal,cal,m,ph,2)),('teste_original',t,fit(cal,t,m,ph)),('teste_espacial_3x3',t,spatial(cal,t,m,ph,3))]:
                rows.append({'metodo':m,'modelo':name,'avaliacao':kind,**metrics(z)})
                f=frame[['gaia_id','x','y','fase_x','fase_y']].copy();f['metodo']=m;f['modelo']=name;f['avaliacao']=kind;f['res_ra'],f['res_dec']=z.T;f['erro']=np.linalg.norm(z,axis=1);preds.append(f)
    r=pd.DataFrame(rows);r.to_csv(out/'modelos_fase.csv',index=False);pd.concat(preds,ignore_index=True).to_csv(out/'predicoes_fase.csv',index=False)
    print('\nPARTE B — modelos aninhados\n',r.to_string(index=False),flush=True)
    fig,axes=plt.subplots(1,2,figsize=(11,5));vmax=max(t.corrigido_continuo.max(),t.corrigido_discreto.max())
    for ax,m in zip(axes,METHODS):
        s=ax.scatter(t.fase_x,t.fase_y,c=t[f'corrigido_{m}'],vmin=0,vmax=vmax,cmap='viridis');ax.set(xlim=(-.51,.51),ylim=(-.51,.51),xlabel='Fase Gaia X (pixel)',ylabel='Fase Gaia Y (pixel)',title=m,aspect='equal')
    fig.colorbar(s,ax=axes,label='Erro corrigido (arcsec)');fig.subplots_adjust(wspace=.35,right=.85);fig.savefig(out/'fase_gaia.png',dpi=180);plt.close(fig)
    return ['modelos_fase.csv','predicoes_fase.csv','fase_gaia.png']

# 06. Execução, testes mínimos e documentação

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--csv',type=Path,default=ROOT/'experimento_012_D6.csv');p.add_argument('--fits',type=Path);p.add_argument('--out',type=Path);p.add_argument('--somente-csv',action='store_true');a=p.parse_args()
    assert np.allclose(phase([0,.25,.5,1.75]),[0,.25,-.5,-.25])
    out=a.out or ROOT/('resultados_016A' if a.somente_csv else 'resultados_016');out.mkdir(parents=True,exist_ok=True)
    d=load(a.csv);meta={'experimento':'016','exposicao':EXP,'modo':'A' if a.somente_csv else 'A+B','sha256_csv':sha(a.csv),'sha256_codigo':sha(__file__),'limite_px':LIMIT,'buffer_px':BUFFER,'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__,'limites':['Uma única exposição; amostra selecionada com Gaia; conjunto de teste já explorado.','Fase ajustada é endógena; fase Gaia depende do WCS e da propagação aproximada 012, não é verdade absoluta.','Comparações descritivas, sem p-valores nem inferência causal.','Não altera a ePSF, nem compara interpoladores; isso requer novo ajuste dos pixels.']}
    names=part_a(d,out)
    if not a.somente_csv:
        d=reference(d,a,meta);names+=part_b(d,out)
    d.to_csv(out/'dados_individuais.csv',index=False);(out/'metadados.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8');names+=['dados_individuais.csv','metadados.json']
    zp=out.parent/('experimento_016A_resultados.zip' if a.somente_csv else 'experimento_016_resultados.zip')
    with zipfile.ZipFile(zp,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for n in names:z.write(out/n,n)
        z.write(__file__,'experimento_016.py')
    print('\nResultados:',zp,flush=True)

if __name__=='__main__':main()
