import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
import scikit_posthocs as sp
import matplotlib as mpl
import itertools as it
from scipy.spatial.distance import euclidean


#file dei risultati
#resultsFilename = 'results9_budget10000.csv'
resultsFilename = 'results10_budget10000.csv'

#carica il dataframe dei risultati
df = pd.read_csv(resultsFilename, delimiter=';')

#modifica nomi degli algoritmi
df.alg.replace('OnePlusOne','ES',inplace=True)
#df.alg.replace('OnePlusOne','(1+1)-ES',inplace=True)
df.alg.replace('RealSpacePSO','PSO',inplace=True)
df.alg.replace('CauchyOnePlusOne','CES',inplace=True)
#df.alg.replace('CauchyOnePlusOne','C-(1+1)-ES',inplace=True)
df.alg.replace('DiagonalCMA','DCMA',inplace=True)
df.alg.replace('NelderMead','NM',inplace=True)
df.alg.replace('RandomSearch','RS',inplace=True)
df.alg.replace('ScrHammersleySearch','SHS',inplace=True)

#test kruskal-wallis fra gli algoritmi su test-accuracy
data = [group['acc_test'].values for name,group in df.groupby('alg')]
H,p = ss.kruskal(*data)
print(f'Kruskal-Wallis on Test-Accuracy - pvalue={p}')

#test posthoc conover (con fdr_bh Benjamini/Hochberg adjustment) e ottieni matrice dei pvalues
pvalues = sp.posthoc_conover(df, val_col='acc_test', group_col='alg', p_adjust='fdr_bh')

#crea tabella con test-accuracy
tab = df.groupby('alg').agg(['median','mean','std','min','max']).acc_test.sort_values('median')
print(tab)
tab.reset_index(inplace=True)
tab['pvalue'] = -1.0
refAlg = tab.iloc[0].alg
nalgs = tab.shape[0]
for i in range(nalgs):
    alg = tab.iloc[i].alg
    p = pvalues.loc[refAlg,alg]
    tab.pvalue.iloc[i] = p
tab.rename(columns={'alg': 'Algorithm'}, inplace=True)
tab = tab[['Algorithm','median','pvalue','mean','std','min','max']]
stab = tab.copy()
stab.pvalue = stab.pvalue.transform(lambda x: f'{x:.0e}')
stab.to_latex('tab_test.tex', index=False, float_format='%.2f', column_format='lrrrrrr')

#crea boxplot con test-accuracy
plt.figure()
ax = sns.boxplot(data=df, x='alg', y='acc_test', order=tab.Algorithm, palette='muted', width=0.8, linewidth=0.7, fliersize=2.5)
ax.set_yscale('log')
ax.set_xlabel('Algorithm')
ax.set_ylabel('Mean Percentage Relative Error')
ax.tick_params(axis='both', which='major', labelsize=10)
ax.set_yticks([6,10,15,20,25,50])
ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
plt.tight_layout()
plt.savefig(f'boxplot_test.pdf')
#plt.show()

#crea heatmap con pvalues
mapvalues = pvalues.copy()
algs = df.alg.unique()
for alg1 in algs:
    for alg2 in algs:
        med1 = tab.query(f'Algorithm=="{alg1}"')['median'].iloc[0]
        med2 = tab.query(f'Algorithm=="{alg2}"')['median'].iloc[0]
        mapvalues.loc[alg1,alg2] = 1 - mapvalues.loc[alg1,alg2]**0.3
        if med1>med2:
            mapvalues.loc[alg1,alg2] *= -1
plt.figure()
ax = sns.heatmap(
    data=mapvalues,
    vmin=-1.,
    vmax=+1.,
    center=0.,
    cmap='RdYlGn',#'coolwarm',
    cbar=False,
    linecolor='black',
    linewidths=0.8,
    annot=pvalues,
    annot_kws={'fontsize':5},
    square=True
    )
ax.tick_params(axis='both', which='major', labelsize=10)
plt.tight_layout()
plt.savefig(f'heatmap_pvalues.pdf')
#plt.show()

#crea correlation plot ... questo va messo per primo nel paper per mostrare che è sensato tutto
plt.figure()
ax = sns.regplot(
    data=df,
    x='acc_train',
    y='acc_test',
    scatter_kws={'s':3},
    line_kws={'color': 'black', 'lw':1}
    )
ax.set_xlim([0,80])
ax.set_ylim([0,80])
ax.set_xlabel('Perc. Rel. Error on Training Data')
ax.set_ylabel('Perc. Rel. Error on Test Data')
pearsonr = ss.pearsonr(df.acc_train,df.acc_test)[0]
#spearmanr = ss.spearmanr(df.acc_train,df.acc_test)[0]
#kendalltau = ss.kendalltau(df.acc_train,df.acc_test)[0]
plt.text(0.02, 0.98, f'Paerson $r$ = {pearsonr:.2f}', ha='left', va='top', transform=ax.transAxes)
plt.tight_layout()
plt.savefig(f'correlation_plot.pdf')
#plt.show()

#crea tabella distanze su search space
nparams = 3
params = [f'a{i}' for i in range(nparams)]
ddf = df.query('alg!="RS" and alg!="SHS"').groupby('alg').agg(['min','max'])[params]
ddf.columns = ['_'.join(col).strip() for col in ddf.columns.values]
for i in range(nparams):
     ddf.eval(f'a{i}_delta = a{i}_max - a{i}_min', inplace=True)
ddf['delta'] = -1.
df['a'] = df[params].apply( lambda x: x.to_numpy(), axis=1 )
for alg in set(algs)-{'RS','SHS'}:
    maxd = 0.
    for x,y in it.combinations(df.query(f'alg=="{alg}"').a,2):
        d = euclidean(x,y)
        if d>maxd: maxd = d
    ddf.loc[alg,'delta'] = maxd
ddf = ddf[ ['delta'] + [ f'a{i}_delta' for i in range(nparams) ] ]
f = open('tab_delta.tex','w')
print(r'\begin{tabular}{l'+('r'*8)+'}', file=f)
print(r'\toprule', file=f)
algs = ddf.index
s = r'\textbf{Algorithm} & ' + ' & '.join([ f'\\textbf{{{alg}}}' for alg in algs]) + r' \\'
print(s, file=f)
print(r'\midrule', file=f)
dists = ddf.delta.to_numpy()
s = r'\textbf{Max distance} & ' + ' & '.join([f'{dist:.2f}' for dist in dists]) + r' \\'
print(s, file=f)
print(r'\bottomrule', file=f)
print(r'\end{tabular}', file=f)
f.close()
