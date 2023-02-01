import numpy as np
import pandas as pd
import nevergrad as ng
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sys



#questa è la regression function (x è preso dal dataset, mentre i parametri a sono quelli che poi saranno l'incognita in objfun)
def regrfun(a,x):
    #costanti
    C_s = 0.5
    k = 0.41
    mu = 0.00131
    g = 9.81
    Ksmooth = 2.25
    Kr = 90
    #variabili x
    ks = x[0]
    h1 = x[1]
    h2 = x[2]
    F1 = x[3]
    #variabili calcolate
    V1 = F1*np.sqrt(g*h1)
    Beta = 0.42*(ks/h1)
    K_s_plus = (ks*ks)*Beta*V1/mu
    #valore di phi
    if K_s_plus <= Ksmooth:
        Phi = 0
    elif Ksmooth < K_s_plus <= Kr:
        Phi = (1/ks)*np.log(C_s*K_s_plus)*np.sin((np.pi/2)*((np.log(K_s_plus)-np.log(Ksmooth))/(np.log(Kr)-np.log(Ksmooth))))
    else: #K_s_plus > Kr
        Phi = (1/ks)*np.log(C_s*K_s_plus)
    '''
    #calcolo del valore Lr
    Lr = ( a[0]*Phi + a[1]*(h2/h1) + a[2]*F1 ) * h1
    #la y da ritornare è Lr
    return Lr
    '''
    #calcolo del valore Lr_su_h1
    Lr_su_h1 = a[0]*Phi + a[1]*(h2/h1) + a[2]*(F1-1.5)
    #la y da ritornare è Lr_su_h1
    return Lr_su_h1


#dato il training set (X_train,y_train), ritorna la funzione obiettivo (con solo il vettore a come parametro, ma training set considerato all'interno di objfun)
def create_objfun(X_train,y_train):
    nsamples = y_train.size
    #questa è la vera objfun che userà il training set al suo interno, senza prenderlo come parametro
    def objfun(a):
        loss = 0.
        for x,y in zip(X_train,y_train):
            y_pred = regrfun(a,x)
            #loss += (y_pred - y)**2
            loss += np.abs(y_pred - y) #per farlo nella stessa unità di misura in cm
        err_abs = loss/nsamples #errore assoluto
        err_rel = err_abs / np.mean(y_train) #errore relativo
        return err_rel*100 #in percentuale
    #ritorna la objfun agganciata al training set passato come parametro
    return objfun



#crea un algoritmo Nevergrad dato il nome della sua classe Nevergrad e il budget
#nomi delle classi DE si trovano qui: https://github.com/facebookresearch/nevergrad/blob/master/nevergrad/optimization/differentialevolution.py
#altri nomi delle classi di ottimizzatori si trovano qui: https://github.com/facebookresearch/nevergrad/blob/master/nevergrad/benchmark/optimizer_groups.txt (nota però che non tutti sono per problemi numerici single-objective unconstrained e unnoisy)
#algString in {DE, RealSpacePSO, BO, OnePlusOne, CMA}
def create_algorithm(algString, budget):
        n = 3                   #dimensionalità spazio di ricerca
        low, upp = -50., +50.   #lower e upper bound su ogni dimensione
        nw = 1                  #numero di workers
        param = ng.p.Array(shape=(n,))
        param.set_bounds(low,upp)
        algFullString = f'ng.optimizers.{algString}(parametrization=param, budget=budget, num_workers=nw)'
        alg = eval(algFullString)
        return alg



#esegui un singolo run di ottimizzazione
def run_singleExecution(alg, X_train, y_train, X_test, y_test, seed):
    alg.parametrization.random_state = np.random.RandomState(seed)
    objfun = create_objfun(X_train,y_train)
    recom = alg.minimize(objfun)
    a_star = recom.value
    #print(a_star) ####debug
    acc_train = objfun(a_star)
    objfun_test = create_objfun(X_test,y_test)
    acc_test = objfun_test(a_star)
    return acc_train, acc_test, a_star[0], a_star[1], a_star[2]



#main
if __name__=='__main__':
    #parametri per i run
    nrep = 10 #numero di ripetizione della repeated k-fold cross-validation
    nfol =  3 #numero di folds
    nrun = 10 #numero di run dell'algoritmo su ogni fold
    algs = [ 'OnePlusOne', 'DE', 'RealSpacePSO', 'CMA', 'CauchyOnePlusOne', 'DiagonalCMA', 'QrDE', 'NelderMead', 'RandomSearch', 'ScrHammersleySearch' ]
    budg = 10_000 #numero di valutazioni per run
    if len(sys.argv)<2: sys.exit('Immetti budget come parametro')
    budg = int(sys.argv[1])
    #carica dataset e rimuovi righe con NA
    ds = pd.read_csv('dataset.csv', delimiter=';')
    ds.dropna(inplace=True)
    X = ds[['ks','h1','h2','F1']].to_numpy()
    '''
    y = ds['Lr'].to_numpy()
    '''
    y = ds['Lr'].to_numpy() / ds['h1'].to_numpy() #così la y è Lr/h1
    #struttura dati per accumulare i risultati
    results = []
    #per ogni ripetizione della cross-validation
    for i_rep in range(1,nrep+1):
        kfold = KFold(n_splits=nfol, shuffle=True, random_state=i_rep)
        #per ogni fold della ripetizione attuale della cross-validation
        i_fol = 1
        for train_index,test_index in kfold.split(X,y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #per ogni algoritmo
            for algString in algs:
                #per ogni run dell'algoritmo
                for i_run in range(1,nrun+1):
                    #print('---')
                    print(f'i_rep={i_rep} i_fol={i_fol} alg={algString} i_run={i_run}', flush=True)
                    alg = create_algorithm(algString, budg)
                    acc_train, acc_test, a0,a1,a2 = run_singleExecution(alg, X_train, y_train, X_test, y_test, i_run)
                    print(f'acc_train={acc_train} acc_test={acc_test}', flush=True)
                    #print('---')
                    results.append({
                            'i_rep':        i_rep,
                            'i_fol':        i_fol,
                            'alg':          algString,
                            'i_run':        i_run,
                            'acc_train':    acc_train,
                            'acc_test':     acc_test,
                            'a0':           a0,
                            'a1':           a1,
                            'a2':           a2,
                        })
                    #fine singola esecuzione
                #fine di tutte le esecuzioni di un algoritmo
            #fine di tutti gli algoritmi su una fold
            i_fol += 1
        #fine di tutte le fold
    #fine di tutte le ripetizioni
    #salva i risultati
    df = pd.DataFrame(results)
    df.to_pickle(f'results10_budget{budg}.pickle')
    df.to_csv(f'results10_budget{budg}.csv', sep=';', index=False)
    #fine
