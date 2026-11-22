import sys,time,json; sys.argv=['x','0.139','0.045']
import numpy as np, ev_sim as S
ND=12
def kp(a, fixed_c=None):
    rev=a['rev']; gross=rev-a['cost_e']-a['cost_dev']-a['deg']-a['host_vom']
    out=dict(rel=a['delv'].sum()/a['dem0'].sum(), rel_c=a['delv']/a['dem0'], relP=a['delv'].sum()/a['demP'].sum(),
             loss=a['loss'].sum()/a['x'].sum(), comps={k:a[k].sum()/a['x'].sum() for k in ['l_dist','l_conv','l_std','l_st','l_pk']},
             rev=rev, gross=gross, delv=a['delv'], x=a['x'], dev=a['cost_dev'].sum(), cost_e=a['cost_e'].sum(),
             cyc=a['dis'].sum()/S.E_ST.sum(), reuse=a['ru'].sum()/a['REtot'].sum(), mapeD=a['mapeD'].mean(), mapeRE=a['mapeRE'].mean(),
             avgcost=(a['cost_e'].sum()+a['cost_dev'].sum())/a['delv'].sum())
    return out
t0=time.time(); R={}
for p in ['base','mlrule','opt','full']:
    t=time.time(); r=S.run_policy(p,ndays=ND); R[p]=(r,S.summarize(r)); print(p,'done %.0fs'%(time.time()-t),flush=True)
# fixed-opex calibration: baseline blended margin = 12.5% in aggregate; fixed allocated by MW
b=R['base'][1]; gross_b=(b['rev']-b['cost_e']-b['cost_dev']-b['deg']-b['host_vom']).sum()
FIX_day=gross_b-0.125*b['rev'].sum()           # Rs'000 per day
fix_c=FIX_day*S.CAP/S.CAP.sum()
print('fixed opex (Rs cr/yr): %.0f'%(FIX_day*365/1e4))
res={}
for p,(r,a) in R.items():
    k=kp(a); k['margin_c']=(k['gross']-fix_c)/k['rev']; k['margin']=(k['gross'].sum()-FIX_day)/k['rev'].sum()
    k['ebitda_cr']=(k['gross'].sum()-FIX_day)*365/1e4; k['rev_cr']=k['rev'].sum()*365/1e4
    res[p]=k
    print(f"\n== {p}: rel {k['rel']:.3f} (price-adj {k['relP']:.3f}) loss {k['loss']:.4f} margin {k['margin']:.3f} EBITDA Rs{k['ebitda_cr']:.0f}cr rev Rs{k['rev_cr']:.0f}cr")
    print(' rel_c',np.round(k['rel_c'],3),' margin_c',np.round(k['margin_c'],3))
    print(' loss comps',{a_:round(v,4) for a_,v in k['comps'].items()},' cyc/day %.2f reuse %.3f avgcost %.2f dev Rs%.0fk/day mapeD %.3f mapeRE %.3f'%(k['cyc'],k['reuse'],k['avgcost'],k['dev'],k['mapeD'],k['mapeRE']))
    for s in r: print('   ',s,'rel %.3f loss %.4f'%(r[s]['delv'].sum()/r[s]['dem0'].sum(), r[s]['loss'].sum()/r[s]['x'].sum()), ' margin %.3f'%(((r[s]['rev']-r[s]['cost_e']-r[s]['cost_dev']-r[s]['deg']-r[s]['host_vom']).sum()-FIX_day)/r[s]['rev'].sum()))
print('total %.0fs'%(time.time()-t0))
import pickle; pickle.dump(dict(res=res,FIX_day=FIX_day),open('res.pkl','wb'))
