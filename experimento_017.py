"""SPHEREx-Lab 017: centragem da ePSF; mesma janela, quatro estratégias.
Executar: py -u experimento_017.py
Opcional: --fits CAMINHO --fase CAMINHO --csv CAMINHO --out PASTA
Teste sintético (sem FITS): py -u experimento_017.py --self-test
"""
# 01. Protocolo
from pathlib import Path
import argparse, hashlib, json, sys, zipfile
import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
EXP='2026W13_2A_0238'
URL='https://irsa.ipac.caltech.edu/ibe/data/spherex/dr1/level2/2026W13_2A/l2b-v27-2026-237/6/level2_2026W13_2A_0238_1D6_spx_l2b-v27-2026-237.fits'
CSV_HASH='c71a632e01b95c67b0ae1027278d24df1b70e575bc862d4f59ee6f672678a0c0'
FITS_HASH='682a1380468124627784520960a176899b036d0d262422fae8737d840a49fe90'
SIZE=2040; R=4; PHASES=(-.4,0,.4); SEED=20261117; BOOT=4000
Y,X=np.mgrid[-R:R+1,-R:R+1]; CORE=X*X+Y*Y<=4; ANNULUS=X*X+Y*Y>=9
BITS=[0,1,2,6,7,9,10,11,12,14,15,17,19,20,22,23,24,25,26,27,28,29]
MASK=sum(1<<b for b in BITS)
METHODS=('original','gaia_controle','ampliado','multicelulas')
OPTIONS={'xtol':1e-5,'ftol':1e-8,'maxiter':100}

def require(ok,msg):
    if not ok: raise RuntimeError(msg)
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

# 02. Identidade da amostra; fase de referência exportada no 016 B.
def read_inputs(a):
    require(sha(a.csv)==CSV_HASH,'CSV 012 diferente do arquivo auditado. Use o original.')
    d=pd.read_csv(a.csv,dtype={'gaia_id':str,'exposicao':str})
    require(len(d)==260 and d.detector.eq(6).all() and d.exposicao.eq(EXP).all(),'Amostra incompatível.')
    require(d.conjunto.value_counts().to_dict()=={'calibracao':130,'teste':130},'Divisão incompatível.')
    require(not d.gaia_id.duplicated().any() and not d.duplicated(['x','y']).any(),'Duplicatas.')
    require(d.gaia_id.str.fullmatch(r'\d+').all(),'IDs Gaia inválidos.')
    p=a.fase
    if p is None:
        candidates=[ROOT/'resultados_016'/'dados_individuais.csv',ROOT/'dados_individuais.csv']
        p=next((s for s in candidates if s.is_file()),candidates[0])
    f=pd.read_csv(p,dtype={'gaia_id':str,'exposicao':str})
    require(len(f)==260 and not f.gaia_id.duplicated().any() and set(f.gaia_id)==set(d.gaia_id),'IDs do 016 incompatíveis.')
    f=f.set_index('gaia_id').loc[d.gaia_id].reset_index(drop=True)
    for k in ['exposicao','conjunto']:
        require(np.array_equal(d[k],f[k]),f'Coluna {k} divergente.')
    for k in ['detector','x','y','snr','mag_g','dx_continuo','dy_continuo','r2_continuo','erro_ra_continuo','erro_dec_continuo']:
        require(np.allclose(d[k],f[k],rtol=0,atol=1e-9),f'Coluna {k} divergente.')
    d['gx']=f.gaia_x;d['gy']=f.gaia_y
    require(np.isfinite(d.select_dtypes(include='number')).all().all(),'Dados não finitos.')
    require(((d[['x','y','gx','gy']]>=8)&(d[['x','y','gx','gy']]<SIZE-8)).all().all(),'Fonte próxima da borda do detector.')
    require(np.max(np.hypot(d.gx-d.x,d.gy-d.y))<=1.5001,'Pico fora do raio de seleção 012.')
    d['mesmo_pixel']=(np.floor(d.gx+.5)==d.x)&(np.floor(d.gy+.5)==d.y)
    return d,p

# 03. Mesma função objetivo do 012: ePSF local, amplitude >=0 e fundo constante.
def objective(patch,var,ok,epsf,ox,oy):
    d=patch[ok];w=1/var[ok];mean=np.average(d,weights=w);chi0=np.sum(w*(d-mean)**2)
    require(np.isfinite(chi0) and chi0>0,'Variância ponderada do recorte inválida.')
    cy,cx=(np.array(epsf.shape)-1)/2
    def evaluate(pos,details=False):
        dx,dy=pos
        P=map_coordinates(epsf,[cy+(Y-dy)*oy,cx+(X-dx)*ox],order=1,mode='constant',cval=0,prefilter=False)
        total=P.sum()
        if not np.isfinite(total) or total<=0:return (np.inf,0.,np.nan) if details else np.inf
        P/=total;p=P[ok];pm=np.average(p,weights=w);q=p-pm;den=np.sum(w*q*q)
        if not np.isfinite(den) or den<=0:return (np.inf,0.,np.nan) if details else np.inf
        A=max(0.,np.sum(w*q*(d-mean))/den);B=mean-A*pm
        val=np.sum(w*(d-B-A*p)**2)/chi0
        return (float(val),float(A),float(B)) if details else float(val)
    return evaluate

# 04. Ajustes. Nunca usar a distância angular para selecionar C ou D.
def tile(fun,ix,iy):
    bounds=[(ix-.5,ix+.5),(iy-.5,iy+.5)]
    grid=sorted((fun((ix+dx,iy+dy)),ix+dx,iy+dy) for dx in PHASES for dy in PHASES)
    return improve(fun,grid[:2],bounds,grid[0])

def improve(fun,starts,bounds,best):
    calls=0; successes=0
    for _,dx,dy in starts:
        r=minimize(fun,[dx,dy],method='Powell',bounds=bounds,options=OPTIONS)
        calls+=1
        if r.success and np.isfinite(r.fun) and all(lo-1e-8<=x<=hi+1e-8 for x,(lo,hi) in zip(r.x,bounds)):
            successes+=1
            candidate=(float(r.fun),float(r.x[0]),float(r.x[1]))
            if candidate[0]<best[0]:best=candidate
    require(np.isfinite(best).all(),'Nenhuma solução finita.')
    return best,calls,successes

def strategies(fun,dx,dy,ix,iy):
    orig=(fun((dx,dy)),dx,dy)
    replay,rc,rs=tile(fun,0,0)
    gaia,gc,gs=tile(fun,ix,iy)
    if ix==0 and iy==0 and orig[0]<gaia[0]:gaia=orig
    # C: mesma posição inicial original; único domínio mais amplo.
    wide,wc,ws=improve(fun,[orig],[(-1.5,1.5)]*2,orig)
    # D: todas as células adjacentes; mantém original se não houver ganho.
    best=orig;dc=ds=0
    for a in (-1,0,1):
        for b in (-1,0,1):
            cand,c,s=tile(fun,a,b);dc+=c;ds+=s
            if cand[0]<best[0]:best=cand
    # União explícita: D também considera C; não garante ótimo global.
    if wide[0]<best[0]:best=wide
    return {'original':(orig,0,0),'gaia_controle':(gaia,gc,gs),'ampliado':(wide,wc,ws),'multicelulas':(best,dc+wc,ds+ws)},replay,rc,rs

# 05. FITS original; sem nova consulta ao Gaia. Auditar WCS antes dos ajustes.
def run_pixels(a,d):
    import astropy
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from astropy.utils.data import download_file
    path=a.fits or Path(download_file(URL,cache=True,timeout=300))
    digest=sha(path);require(digest==FITS_HASH,'FITS diferente da versão usada no 016. Não substituir por outra versão.')
    with fits.open(path) as f:
        image=f['IMAGE'].data.astype(float);image-=f['ZODI'].data
        var=f['VARIANCE'].data.astype(float);flags=f['FLAGS'].data.astype(np.uint32)
        h=f['IMAGE'].header.copy();epsf=f['EPSF'].data.copy();eh=f['EPSF'].header.copy()
    require(image.shape==var.shape==flags.shape==(SIZE,SIZE),'Dimensões FITS inesperadas.')
    ox=int(eh['OVSMPX']);oy=int(eh['OVSMPY']);require(ox==oy==5,'Superamostragem inesperada.')
    valid=np.isfinite(image)&np.isfinite(var)&(var>0)&((flags&MASK)==0)
    w=WCS(h).celestial;ref=w.pixel_to_world(d.gx.to_numpy(),d.gy.to_numpy()).icrs
    orig=w.pixel_to_world(d.x.to_numpy()+d.dx_continuo.to_numpy(),d.y.to_numpy()+d.dy_continuo.to_numpy()).icrs
    saved=SkyCoord(ra=d.ra_continuo.to_numpy()*u.deg,dec=d.dec_continuo.to_numpy()*u.deg)
    require(orig.separation(saved).arcsec.max()<.02,'WCS original não reproduzido.')
    ra,dec=ref.spherical_offsets_to(orig)
    require(np.allclose(ra.arcsec,d.erro_ra_continuo,atol=.02,rtol=0) and np.allclose(dec.arcsec,d.erro_dec_continuo,atol=.02,rtol=0),'Referência 016 não reproduz os resíduos 012.')
    cx=np.asarray(epsf['XCENTER'],float);cy=np.asarray(epsf['YCENTER'],float)
    rows=[];audits=[]
    for i,r in enumerate(d.itertuples(index=False)):
        x,y=int(r.x),int(r.y);sl=np.s_[y-R:y+R+1,x-R:x+R+1]
        patch=image[sl];v=var[sl];ok=valid[sl]
        require(ok.sum()>=65 and ok[CORE].all(),f'Máscara original divergente: {r.gaia_id}')
        snr=(image[y,x]-np.median(patch[ANNULUS&ok]))/np.sqrt(var[y,x])
        require(np.isclose(snr,r.snr,rtol=1e-6,atol=1e-6),f'SNR divergente: {r.gaia_id}')
        zone=int(np.argmin((cx-x)**2+(cy-y)**2));psf=np.asarray(epsf['EPSF'][zone],float).reshape(33,33)
        require(np.isfinite(psf).all(),'ePSF não finita.')
        fun=objective(patch,v,ok,psf,ox,oy)
        ix=int(np.floor(r.gx+.5))-x;iy=int(np.floor(r.gy+.5))-y
        require(max(abs(ix),abs(iy))<=1,'Pixel Gaia fora das nove células.')
        sol,replay,rc,rs=strategies(fun,r.dx_continuo,r.dy_continuo,ix,iy)
        old=sol['original'][0];diff=abs(1-old[0]-r.r2_continuo)
        require(diff<=1e-7,f'R2 original não reproduzido: {r.gaia_id}, diferença {diff}')
        audits.append({'gaia_id':r.gaia_id,'r2_012':r.r2_continuo,'r2_recalculado':1-old[0],'r2_reajuste':1-replay[0],'dx_reajuste':replay[1],'dy_reajuste':replay[2],'dist_reajuste_px':np.hypot(replay[1]-old[1],replay[2]-old[2]),'powell_sucessos':rs,'powell_tentativas':rc,'snr_recalculada':snr,'n_pixels':int(ok.sum())})
        for name,(s,c,success) in sol.items():
            chi,dx,dy=s;score,A,B=fun((dx,dy),True)
            sky=w.pixel_to_world(x+dx,y+dy).icrs;da,dd=ref[i].spherical_offsets_to(sky)
            rows.append({'gaia_id':r.gaia_id,'conjunto':r.conjunto,'x':x,'y':y,'gx':r.gx,'gy':r.gy,'snr':r.snr,'mag_g':r.mag_g,'mesmo_pixel_original':r.mesmo_pixel,'metodo':name,'dx':dx,'dy':dy,'r2':1-score,'amplitude':A,'fundo':B,'erro_ra':da.arcsec,'erro_dec':dd.arcsec,'distancia_bruta':ref[i].separation(sky).arcsec,'mudou_pixel':bool(int(np.floor(dx+.5))!=0 or int(np.floor(dy+.5))!=0),'limite_externo':bool(max(abs(dx),abs(dy))>=1.499),'powell_tentativas':c,'powell_sucessos':success,'n_pixels':int(ok.sum()),'zona_epsf':zone})
        if (i+1)%10==0 or i+1==len(d):print(f'Ajustes: {i+1}/{len(d)}',flush=True)
    out=pd.DataFrame(rows);audit=pd.DataFrame(audits)
    require(len(out)==1040 and not out.duplicated(['gaia_id','metodo']).any(),'Saída incompleta.')
    pivot=out.pivot(index='gaia_id',columns='metodo',values='r2')
    require((pivot.ampliado>=pivot.original-1e-9).all() and (pivot.multicelulas>=pivot.ampliado-1e-9).all(),'Soluções aninhadas incompatíveis.')
    meta={'sha256_fits':digest,'url_fits':URL,'mjd_avg':float(h['MJD-AVG']),'astropy':astropy.__version__,'ox':ox,'oy':oy}
    return out,audit,meta

# 06. Calibração independente por estratégia, avaliação emparelhada.
def summarize(raw,d):
    r=raw.copy();offsets=[];metrics=[]
    for m in METHODS:
        sel=r.metodo.eq(m);cal=r.loc[sel&r.conjunto.eq('calibracao')]
        da=cal.erro_ra.median();dd=cal.erro_dec.median()
        r.loc[sel,'corr_ra']=r.loc[sel,'erro_ra']-da;r.loc[sel,'corr_dec']=r.loc[sel,'erro_dec']-dd
        offsets.append({'metodo':m,'offset_ra':da,'offset_dec':dd})
    r['erro_corrigido']=np.hypot(r.corr_ra,r.corr_dec)
    old=r[r.metodo.eq('original')].set_index('gaia_id').loc[d.gaia_id]
    require(np.allclose(old.erro_corrigido,d.corrigido_continuo,rtol=0,atol=.04),'Erro corrigido original não reproduzido.')
    r=r.join(old[['erro_corrigido','r2']].rename(columns={'erro_corrigido':'erro_original','r2':'r2_original'}),on='gaia_id')
    r['ganho_angular']=r.erro_original-r.erro_corrigido;r['ganho_r2']=r.r2-r.r2_original
    t=r[r.conjunto.eq('teste')]
    for m in METHODS:
        g=t[t.metodo.eq(m)]
        for label,part in [('todas',g),('mesmo_pixel',g[g.mesmo_pixel_original]),('pixel_diferente',g[~g.mesmo_pixel_original])]:
            e=part.erro_corrigido.to_numpy();require(len(e)>0,'Subgrupo vazio.')
            metrics.append({'metodo':m,'grupo':label,'n':len(e),'mediana':np.median(e),'rmse':np.sqrt(np.mean(e*e)),'p90':np.quantile(e,.9),'p95':np.quantile(e,.95),'maximo':max(e),'mediana_r2':part.r2.median(),'mediana_ganho_r2':part.ganho_r2.median(),'melhoraram':int((part.ganho_angular>1e-8).sum()),'pioraram':int((part.ganho_angular< -1e-8).sum()),'mudaram_pixel':int(part.mudou_pixel.sum()),'limite_externo':int(part.limite_externo.sum())})
    return r,pd.DataFrame(offsets),pd.DataFrame(metrics)

# 07. Bootstrap espacial emparelhado, condicional aos offsets já ajustados.
def bootstrap(r):
    t=r[r.conjunto.eq('teste')];rows=[]
    for m in METHODS[1:]:
        q=t[t.metodo.eq(m)].sort_values('gaia_id');a=q.erro_original.to_numpy();b=q.erro_corrigido.to_numpy()
        for size in (256,512):
            cells=np.floor(q[['x','y']].to_numpy()/size).astype(int);_,group=np.unique(cells,axis=0,return_inverse=True)
            blocks=[np.flatnonzero(group==j) for j in range(group.max()+1)]
            rng=np.random.default_rng(SEED);samples=[]
            for _ in range(BOOT):
                ix=np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))]);aa,bb=a[ix],b[ix]
                samples.append([np.sqrt(np.mean(bb*bb))-np.sqrt(np.mean(aa*aa)),np.median(bb)-np.median(aa)])
            lo,hi=np.quantile(samples,[.025,.975],axis=0)
            rows.append({'metodo':m,'bloco_px':size,'n_blocos':len(blocks),'delta_rmse':np.sqrt(np.mean(b*b))-np.sqrt(np.mean(a*a)),'delta_rmse_lo':lo[0],'delta_rmse_hi':hi[0],'delta_mediana':np.median(b)-np.median(a),'delta_mediana_lo':lo[1],'delta_mediana_hi':hi[1]})
    return pd.DataFrame(rows)

# 08. Figuras de diagnóstico; mesmos pontos e escalas.
def plots(out,r,m):
    g=m[m.grupo.eq('todas')].set_index('metodo').loc[list(METHODS)];x=np.arange(len(g));fig,ax=plt.subplots(figsize=(9,5));ax.bar(x-.18,g.rmse,.36,label='RMSE');ax.bar(x+.18,g.mediana,.36,label='Mediana');ax.set_xticks(x,g.index,rotation=15);ax.set_ylabel('Erro angular corrigido (arcsec)');ax.legend();fig.tight_layout();fig.savefig(out/'metricas.png',dpi=180);plt.close(fig)
    q=r[(r.conjunto=='teste')&(r.metodo=='multicelulas')];fig,ax=plt.subplots(figsize=(7,5));s=ax.scatter(q.x,q.y,c=q.ganho_angular,cmap='coolwarm');fig.colorbar(s,ax=ax,label='Erro original − erro novo (arcsec)');ax.set(xlabel='X (pixel)',ylabel='Y (pixel)',aspect='equal',title='Ganho astrométrico — múltiplas células');fig.tight_layout();fig.savefig(out/'mapa_ganho.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5));ax.scatter(q.ganho_r2,q.ganho_angular,s=20);ax.axhline(0,color='gray',lw=1);ax.set(xlabel='Ganho de R²',ylabel='Ganho angular (arcsec)',title='Qualidade morfológica versus astrometria');fig.tight_layout();fig.savefig(out/'ganho_r2_astrometria.png',dpi=180);plt.close(fig)

# 09. Teste unitário com ePSF sintética, não observação astronômica.
def selftest():
    y,x=np.mgrid[-16:17,-16:17];psf=np.exp(-(x*x+y*y)/(2*3.2**2));truth=(.92,-.71)
    p=map_coordinates(psf,[16+(Y-truth[1])*5,16+(X-truth[0])*5],order=1,mode='constant',cval=0,prefilter=False);p/=p.sum()
    fun=objective(3+100*p,np.ones_like(p),np.ones_like(p,dtype=bool),psf,5,5)
    require(fun(truth)<1e-20,'Objetivo sintético inválido.')
    res,_,_,_=strategies(fun,.4,-.4,1,-1);a=res['original'][0];b=res['ampliado'][0];z=res['multicelulas'][0]
    require(z[0]<=b[0]<=a[0] and np.hypot(z[1]-truth[0],z[2]-truth[1])<1e-3,'Teste sintético de recuperação falhou.')
    print('Teste matemático sintético: OK',flush=True)

# 10. Execução rastreável.
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--csv',type=Path,default=ROOT/'experimento_012_D6.csv');p.add_argument('--fase',type=Path);p.add_argument('--fits',type=Path);p.add_argument('--out',type=Path,default=ROOT/'resultados_017');p.add_argument('--self-test',action='store_true');a=p.parse_args();selftest()
    if a.self_test:return
    d,phasepath=read_inputs(a);a.out.mkdir(parents=True,exist_ok=True)
    raw,audit,meta=run_pixels(a,d);r,offsets,m=summarize(raw,d);boot=bootstrap(r)
    for name,frame in [('ajustes_individuais',r),('auditoria_original',audit),('offsets',offsets),('metricas',m),('bootstrap_pareado',boot)]:frame.to_csv(a.out/(name+'.csv'),index=False)
    plots(a.out,r,m)
    meta.update({'experimento':'017','exposicao':EXP,'n_calibracao':130,'n_teste':130,'sha256_csv':sha(a.csv),'sha256_fase_016':sha(phasepath),'sha256_codigo':sha(__file__),'janela_px':9,'bits_excluidos':BITS,'interpolacao':'linear','otimizador':'Powell','opcoes':OPTIONS,'limite_ampliado_px':1.5,'bootstrap_repeticoes':BOOT,'semente':SEED,'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'limites':['Amostra guiada pelo Gaia; controle Gaia usa referência e não é comparação cega.','Uma exposição, teste já explorado; sem inferência confirmatória.','Janela e ePSF ancoradas no pico original; não há mudança de recorte nem de interpolação.','Original usa parâmetros salvos 012; reajuste auditado separadamente.','C e D não minimizam erro Gaia; D inclui C, logo ganho R2 é esperado.','Bootstrap descritivo condicional aos offsets, não cobre outras exposições.','Ausência de garantia de ótimo global; zeros fora do suporte ePSF reproduzem regra original.','Não estima taxa de artefatos nem identidade física.']})
    (a.out/'metadados.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    zp=a.out.parent/'experimento_017_resultados.zip'
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
        for n in ['ajustes_individuais.csv','auditoria_original.csv','offsets.csv','metricas.csv','bootstrap_pareado.csv','metricas.png','mapa_ganho.png','ganho_r2_astrometria.png','metadados.json']:z.write(a.out/n,n)
        z.write(__file__,'experimento_017.py')
    print(m.to_string(index=False));print('\nArquivo:',zp,flush=True)
if __name__=='__main__':main()
