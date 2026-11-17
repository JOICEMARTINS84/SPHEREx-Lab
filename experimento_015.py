"""SPHEREx-Lab — 015: auditoria exploratória de modelos de resíduos D6.
py -u experimento_015.py [--csv arquivo] [--out pasta]
Não altera FITS/WCS, não identifica fontes novas. Usa resíduos do experimento 012.
"""
from pathlib import Path
import argparse, hashlib, json, platform, sys, zipfile
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 01. Protocolo fixado a priori desta execução
ROOT = Path(__file__).resolve().parent
METHODS = ('discreto', 'continuo')
MODELS = ('constante', 'afim', 'afim_robusto', 'quadratico', 'quadratico_robusto')
NTERMS = dict(zip(MODELS, (1,3,3,6,6)))
SIZE = 2040.0
BUFFER = 64.0 # pixels, não uma escala de correlação demonstrada
F_SCALE = 2.0 # arcsec; soft_l1 por componente, fixado antes do ajuste
CV_TOL = .02 # preferência por modelo mais simples a até 2% do menor RMSE CV
BOOT = 4000
SEED = 20261117
EXPECTED_HASH = 'c71a632e01b95c67b0ae1027278d24df1b70e575bc862d4f59ee6f672678a0c0'
EXPOSURE = '2026W13_2A_0238'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok, msg):
    if not ok: raise ValueError(msg)

# 02. Validação de identidade e geometria dos dados

def load(path):
    d = pd.read_csv(path, dtype={'gaia_id':str,'exposicao':str})
    needed = {'detector','exposicao','gaia_id','x','y','conjunto','snr','mag_g'}
    needed |= {f'{v}_{m}' for v in ('erro_ra','erro_dec','corrigido') for m in METHODS}
    require(needed <= set(d), f'Colunas ausentes: {needed-set(d)}')
    require(sha(path)==EXPECTED_HASH, 'SHA-256 diferente do arquivo auditado no 014; confira a versão original.')
    require(len(d)==260 and d.detector.eq(6).all() and d.exposicao.eq(EXPOSURE).all(), 'Exposição/amostra inesperada.')
    require(d.gaia_id.str.fullmatch(r'\d+').fillna(False).all() and not d.gaia_id.duplicated().any() and not d.duplicated(['x','y']).any(), 'IDs ou picos inválidos/repetidos.')
    require(d.conjunto.value_counts().to_dict()=={'calibracao':130,'teste':130}, 'Divisão original inválida.')
    num=sorted(needed-{'exposicao','gaia_id','conjunto'})
    require(np.isfinite(d[num].to_numpy(float)).all(), 'Valores não finitos.')
    require(((d[['x','y']]>=0)&(d[['x','y']]<SIZE)).all().all(), 'Coordenadas fora do detector.')
    return d

def design(xy, model):
    x,y = ((np.asarray(xy,float)-(SIZE-1)/2)/(SIZE/2)).T
    return np.column_stack([np.ones(len(x)),x,y,x*x,x*y,y*y])[:,:NTERMS[model]]

# 03. Campo vetorial dos resíduos: Gaia -> SPHEREx, em arcsec.
# A correção é subtraída; trabalhamos na mesma aproximação angular do 012.
def fit(xy, z, model):
    X=design(xy,model)
    require(len(X)>=max(12,3*X.shape[1]), 'Amostra de ajuste insuficiente.')
    require(np.linalg.matrix_rank(X)==X.shape[1], 'Matriz de ajuste sem posto completo.')
    cond=float(np.linalg.cond(X)); require(cond<1e4,'Matriz mal condicionada.')
    if model=='constante': return np.median(z,axis=0)[None,:],cond
    b=np.linalg.lstsq(X,z,rcond=None)[0]
    if model.endswith('robusto'):
        opt=least_squares(lambda v:(X@v.reshape(b.shape)-z).ravel(), b.ravel(),loss='soft_l1', f_scale=F_SCALE,max_nfev=1000)
        require(opt.success and np.isfinite(opt.x).all(),'Otimização robusta não convergiu.')
        b=opt.x.reshape(b.shape)
    return b,cond

def predict(xy,b,model): return design(xy,model)@b

def errors(z,p): return np.linalg.norm(z-p,axis=1)

def metrics(e):
    e=np.asarray(e); require(len(e)>0 and np.isfinite(e).all(),'Métricas inválidas.')
    return {'n':len(e),'mediana':float(np.median(e)),'rmse':float(np.sqrt(np.mean(e**2))),'p90':float(np.quantile(e,.90)),'p95':float(np.quantile(e,.95)),'maximo':float(np.max(e))}

# 04. Blocos espaciais, definidos em pixels absolutos, sem observar os resíduos.
def blocks(xy,k):
    ij=np.floor(np.asarray(xy)*k/SIZE).astype(int)
    return ij[:,0]+k*ij[:,1]

def outside_buffer(xy,k,b):
    w=SIZE/k; ix=b%k; iy=b//k; x,y=np.asarray(xy).T
    return ~((x>=ix*w-BUFFER)&(x<(ix+1)*w+BUFFER)&(y>=iy*w-BUFFER)&(y<(iy+1)*w+BUFFER))

# 05. Predições espaciais: exclui todo o bloco de validação + faixa de proteção
# O conjunto de teste nunca entra em fit().
def spatial(xc,zc,xv,zv,model,k):
    g=blocks(xv,k); pred=np.full_like(zv,np.nan); rows=[]
    for b in np.unique(g):
        train=outside_buffer(xc,k,b); val=g==b
        beta,cond=fit(xc[train],zc[train],model)
        pred[val]=predict(xv[val],beta,model)
        rows.append({'bloco':int(b),'n_ajuste':int(train.sum()),'condicao':cond,**metrics(errors(zv[val],pred[val]))})
    require(np.isfinite(pred).all(),'Incompletude das predições espaciais.')
    return pred,rows

# 06. Incerteza exploratória: bootstrap emparelhado de células espaciais.
def paired_boot(xy,base,new,side,seed):
    g=np.floor(xy/side).astype(int); _,idx=np.unique(g,axis=0,return_inverse=True)
    groups=[np.flatnonzero(idx==j) for j in range(idx.max()+1)]
    if len(groups)<5: return {'n_blocos':len(groups),'delta_rmse_lo':None,'delta_rmse_hi':None,'delta_mediana_lo':None,'delta_mediana_hi':None}
    rng=np.random.default_rng(seed); delta=[]
    for _ in range(BOOT):
        i=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))]); a,b=base[i],new[i]
        delta.append((np.sqrt(np.mean(b*b))-np.sqrt(np.mean(a*a)),np.median(b)-np.median(a)))
    q=np.quantile(delta,[.025,.975],axis=0)
    return {'n_blocos':len(groups),'delta_rmse_lo':q[0,0],'delta_rmse_hi':q[1,0],'delta_mediana_lo':q[0,1],'delta_mediana_hi':q[1,1]}

# 07. Testes internos, dados sintéticos apenas para testar a implementação.
def selftest():
    rng=np.random.default_rng(123); xy=rng.uniform(0,SIZE,(80,2)); D=design(xy,'quadratico')
    truth=rng.normal(size=(6,2)); z=D@truth
    b,_=fit(xy,z,'quadratico'); assert np.allclose(b,truth,atol=1e-10)
    b,_=fit(xy,np.tile([2.,-3.],(80,1)),'constante'); assert np.allclose(b,[[2,-3]])
    b,_=fit(xy,z,'quadratico_robusto'); assert np.allclose(b,truth,atol=1e-8)
    assert np.all(blocks(np.array([[0,0],[2039,2039]]),3)==[0,8])
    assert not outside_buffer(np.array([[680,680]]),3,0)[0]
    assert np.allclose(errors(z,z),0)

# 08. Gráficos diagnósticos (mesma escala angular nos mapas por método).
def plots(out,summary,individual,selected):
    for m in METHODS:
        ss=summary[(summary.metodo==m)&(summary.avaliacao=='teste_original')].set_index('modelo').loc[list(MODELS)]
        fig,ax=plt.subplots(figsize=(10,5)); x=np.arange(len(ss)); ax.bar(x-.18,ss.rmse,.36,label='RMSE');ax.bar(x+.18,ss.mediana,.36,label='Mediana');ax.set_xticks(x,ss.index,rotation=20,ha='right');ax.set_ylabel('Erro residual (arcsec)');ax.set_title(f'D6 — teste original — {m}');ax.legend();fig.tight_layout();fig.savefig(out/f'metricas_{m}.png',dpi=180);plt.close(fig)
        a=individual[(individual.metodo==m)&(individual.avaliacao=='teste_original')]; alt=summary[(summary.metodo==m)&(summary.avaliacao=='cv_cal_3x3')&(summary.modelo!='constante')].sort_values('rmse').iloc[0].modelo; names=['constante',alt]
        fig,axes=plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
        vmax=max(a[a.modelo.isin(names)].erro.max(),1e-9)
        for ax,name in zip(axes,names):
            q=a[a.modelo==name]; s=ax.scatter(q.x,q.y,c=q.erro,vmin=0,vmax=vmax,cmap='viridis',s=30);ax.set_title(name);ax.set(xlim=(0,SIZE),ylim=(0,SIZE),xlabel='X (pixel)',aspect='equal')
        axes[0].set_ylabel('Y (pixel)');fig.colorbar(s,ax=axes,label='Erro (arcsec)',shrink=.7);fig.suptitle(f'D6 — {m}: distribuição dos resíduos — constante vs melhor alternativa no CV');fig.subplots_adjust(left=.08,right=.88,bottom=.12,top=.82,wspace=.15);fig.savefig(out/f'mapa_{m}.png',dpi=180);plt.close(fig)

# 09. Fluxo de análise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--csv',type=Path,default=ROOT/'experimento_012_D6.csv');p.add_argument('--out',type=Path,default=ROOT/'resultados_015');a=p.parse_args()
    selftest();d=load(a.csv);out=a.out;out.mkdir(parents=True,exist_ok=True)
    cal=d[d.conjunto=='calibracao'].reset_index(drop=True);t=d[d.conjunto=='teste'].reset_index(drop=True);xc=cal[['x','y']].to_numpy(float);xt=t[['x','y']].to_numpy(float)
    results=[];folds=[];individual=[];coefs=[];ci=[];selected={}
    def store(m,model,kind,frame,z,pred):
        e=errors(z,pred);results.append({'metodo':m,'modelo':model,'avaliacao':kind,**metrics(e)})
        f=frame[['gaia_id','x','y']].copy();f['metodo']=m;f['modelo']=model;f['avaliacao']=kind;f['pred_ra'],f['pred_dec']=pred.T;f['res_ra'],f['res_dec']=(z-pred).T;f['erro']=e;individual.append(f)
        return e
    for m in METHODS:
        cols=[f'erro_ra_{m}',f'erro_dec_{m}'];zc=cal[cols].to_numpy(float);zt=t[cols].to_numpy(float)
        base=np.linalg.norm(zt-np.median(zc,axis=0),axis=1)
        require(np.allclose(base,t[f'corrigido_{m}'],rtol=0,atol=1e-8),'Não reproduziu a calibração original.')
        cv={}
        for model in MODELS:
            for k in (3,2):
                pred,rows=spatial(xc,zc,xc,zc,model,k);kind=f'cv_cal_{k}x{k}';e=store(m,model,kind,cal,zc,pred)
                if k==3: cv[model]=metrics(e)['rmse']
                folds.extend({'metodo':m,'modelo':model,'avaliacao':kind,**r} for r in rows)
        # Seleção exclusivamente pelo CV espacial 3x3 de calibração.
        best=min(cv.values());selected[m]=next(q for q in MODELS if cv[q]<=best*(1+CV_TOL))
        for model in MODELS:
            beta,cond=fit(xc,zc,model);pred=predict(xt,beta,model);e=store(m,model,'teste_original',t,zt,pred)
            for term, (ra,dec) in zip(['1','x','y','x2','xy','y2'],beta):coefs.append({'metodo':m,'modelo':model,'termo':term,'ra_arcsec':ra,'dec_arcsec':dec,'condicao':cond})
            for k in (3,2):
                pred,rows=spatial(xc,zc,xt,zt,model,k);kind=f'teste_espacial_{k}x{k}';store(m,model,kind,t,zt,pred);folds.extend({'metodo':m,'modelo':model,'avaliacao':kind,**r} for r in rows)
            if model!='constante':
                for side in (512,256): ci.append({'metodo':m,'modelo':model,'selecionado_cv':model==selected[m],'lado_bloco_px':side,'delta_rmse':metrics(e)['rmse']-metrics(base)['rmse'],'delta_mediana':np.median(e)-np.median(base),**paired_boot(xt,base,e,side,SEED)})
        print(f'{m}: modelo selecionado no CV de calibração = {selected[m]}',flush=True)
    summary=pd.DataFrame(results); ind=pd.concat(individual,ignore_index=True)
    summary['selecionado_cv']=summary.apply(lambda r:r.modelo==selected[r.metodo],axis=1)
    summary.to_csv(out/'metricas.csv',index=False);ind.to_csv(out/'residuos_individuais.csv',index=False);pd.DataFrame(folds).to_csv(out/'validacao_por_bloco.csv',index=False);pd.DataFrame(coefs).to_csv(out/'coeficientes.csv',index=False);pd.DataFrame(ci).to_csv(out/'bootstrap_pareado.csv',index=False)
    plots(out,summary,ind,selected)
    meta={'experimento':'015','exposicao':EXPOSURE,'n_calibracao':len(cal),'n_teste':len(t),'sha256_csv':sha(a.csv),'sha256_codigo':sha(__file__),'grade_principal':'3x3','grade_sensibilidade':'2x2','buffer_px':BUFFER,'f_scale_arcsec':F_SCALE,'tolerancia_selecao_cv':CV_TOL,'semente':SEED,'bootstrap_repeticoes':BOOT,'selecionados':selected,'versoes':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'plataforma':platform.platform()},'limites':['Uma exposição e amostra guiada pelo Gaia; teste original já explorado nos experimentos 012-014.','Modelo do campo residual, não reestimativa do WCS FITS (nem identificação causal de distorção instrumental).','Bootstrap espacial emparelhado descritivo das alternativas versus constante; intervalos não simultâneos, condicionam nos ajustes e não capturam variação entre exposições.','Seleção por CV 3x3 e preferência por simplicidade; demais comparações exploratórias.','Sem descarte de estrelas ou reidentificação de fontes.']}
    (out/'metadados.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    # 10. Empacotamento com código e resultados, sem sobrescrever os insumos.
    names=['metricas.csv','residuos_individuais.csv','validacao_por_bloco.csv','coeficientes.csv','bootstrap_pareado.csv','metadados.json']+[f'{p}_{m}.png' for p in ('metricas','mapa') for m in METHODS]
    zip_path=out.parent/'experimento_015_resultados.zip'
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for n in names:z.write(out/n,n)
        z.write(__file__,'experimento_015.py')
    print(summary[summary.avaliacao=='teste_original'].to_string(index=False));print('\nResultados:',zip_path)

if __name__=='__main__': main()
