import pandas as pd, sys
r=pd.read_csv(sys.argv[1]); r['seed']=r.seed.astype(str)
S=[str(s) for s in range(2021,2026)]
out=[]
for (d,p),q in r.groupby(['data','pred_len']):
    def st(m):
        s=q[(q.model==m)&q.seed.isin(S)].mse; return (s.mean(),s.std(),len(s))
    def one(m,seed):
        s=q[(q.model==m)&(q.seed==seed)].mse; return s.iloc[0] if len(s) else float('nan')
    row=dict(data=d,T=p)
    for m in ['DLinear','DLinear+sub','NLinear']:
        mu,sd,n=st(m); row[m+'_mean'],row[m+'_std'],row[m+'_n'],row[m+'_ens']=mu,sd,n,one(m,'ens')
    for m in ['Ridge','Ridge+sub']:
        row[m]=one(m,'-'); row[m+'_lam']=q[q.model==m].lam.iloc[0] if (q.model==m).any() else None
    out.append(row)
t=pd.DataFrame(out)
def better(a,sa,b,sb):  # a better than b by the rule: mean gap larger than the sum of stds
    sa=0 if pd.isna(sa) else sa; sb=0 if pd.isna(sb) else sb
    return 'better' if b-a>sa+sb else ('worse' if a-b>sa+sb else 'tie')
t['H1_vs_DLinear']=[better(x['DLinear+sub_mean'],x['DLinear+sub_std'],x['DLinear_mean'],x['DLinear_std']) for _,x in t.iterrows()]
t['H1_vs_NLinear']=[better(x['DLinear+sub_mean'],x['DLinear+sub_std'],x['NLinear_mean'],x['NLinear_std']) if x['NLinear_n']==5 else '' for _,x in t.iterrows()]
t['H2_vs_DLinear']=[better(x['Ridge'],0,x['DLinear_mean'],x['DLinear_std']) for _,x in t.iterrows()]
t['H2sub_vs_NLinear']=[better(x['Ridge+sub'],0,x['NLinear_mean'],x['NLinear_std']) if x['NLinear_n']==5 else '' for _,x in t.iterrows()]
t['H3_DLinear_gain_%']=100*(t['DLinear_mean']-t['DLinear_ens'])/t['DLinear_mean']
t.to_csv(sys.argv[2],index=False)
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
print(t[['data','T','DLinear_mean','DLinear_std','DLinear_ens','H3_DLinear_gain_%','DLinear+sub_mean','DLinear+sub_std','NLinear_mean','NLinear_std','Ridge','Ridge+sub','H1_vs_DLinear','H1_vs_NLinear','H2_vs_DLinear','H2sub_vs_NLinear']].round(4).to_string(index=False))
