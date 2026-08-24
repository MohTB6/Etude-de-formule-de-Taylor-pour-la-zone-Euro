import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
def taux_longterme():
    ti_nominal=pd.read_csv('lending_facility.csv',sep=',',parse_dates=['DATE'])
    t_infl=pd.read_csv('inflation mensuel.csv', sep=',',parse_dates=['DATE'])
    t_infl = t_infl[t_infl['DATE'] >= '1999-01-01']
    ti_nominal['DATE'] = pd.to_datetime(ti_nominal['DATE'])
    t_infl['DATE'] = pd.to_datetime(t_infl['DATE'])
    ti_nominal['MOIS']=ti_nominal['DATE'].dt.to_period('M')
    t_infl['MOIS']=t_infl['DATE'].dt.to_period('M')
    t_calcul=pd.merge(t_infl,ti_nominal,on='MOIS',how='left').ffill()
    dates=ti_nominal['DATE']
    t_calcul['taux reel']=t_calcul['Rate']-t_calcul['Inflation rate']
    taux=t_calcul['taux reel'].mean()
    return taux
def nettoyage():
    ti_nominal=pd.read_csv('lending_facility.csv',sep=',',parse_dates=['DATE'])
    t_infl=pd.read_csv('inflation mensuel.csv', sep=',',parse_dates=['DATE'])
    croissance_reelle=pd.read_csv('real gdp.csv',sep=',',parse_dates=['observation_date'])
    croissance_potentielle=pd.read_csv('potential_gdp.csv',sep=',',parse_dates=['TIME_PERIOD'])
    t_infl = t_infl[t_infl['DATE'] >= '2002-01-01']
    ti_nominal=ti_nominal[ti_nominal['DATE']>='2002-01-01']
    t_infl=t_infl.reset_index(drop=True)
    ti_nominal['DATE'] = pd.to_datetime(ti_nominal['DATE'])
    t_infl['DATE'] = pd.to_datetime(t_infl['DATE'])
    ti_nominal['MOIS']=ti_nominal['DATE'].dt.to_period('M')
    t_infl['MOIS']=t_infl['DATE'].dt.to_period('M')
    croissance_reelle['observation_date'] = pd.to_datetime(croissance_reelle['observation_date'])
    croissance_potentielle['TIME_PERIOD'] = pd.to_datetime(croissance_potentielle['TIME_PERIOD'])
    croissance_reelle['ANNEE']=croissance_reelle['observation_date'].dt.to_period('Y')
    croissance_potentielle['ANNEE']=croissance_potentielle['TIME_PERIOD'].dt.to_period('Y')
    tableau=pd.DataFrame()
    tableau['MOIS']=pd.to_datetime(t_infl['DATE'])
    tableau=tableau.reset_index(drop=True)
    t_diff_croissance=pd.merge(croissance_reelle,croissance_potentielle,on='ANNEE',how='left')
    t_diff_croissance['output gap']=t_diff_croissance['variation']-t_diff_croissance['OBS_VALUE']
    t_output_gap=pd.DataFrame()
    t_output_gap['date']=pd.to_datetime(t_diff_croissance['observation_date'])
    t_output_gap['valeurs output gap']=t_diff_croissance['output gap']
    tableau['inflation']=t_infl["Inflation rate"]
    tableau['ANNEE'] = tableau['MOIS'].dt.year
    t_output_gap['ANNEE'] = t_output_gap['date'].dt.year
    t_final=pd.merge(tableau,t_output_gap,on='ANNEE',how='left')
    t_final=t_final[t_final['MOIS']<'2026-01-01']
    t_final=t_final.reset_index(drop=True)
    ti_nominal['MOIS']=ti_nominal['DATE'].dt.to_period('M')
    ti_final=ti_nominal.groupby('MOIS')['Rate'].mean()
    #ti_final=ti_final[ti_final['MOIS']>'2002-01']
    t_final['MOIS']=t_final['MOIS'].dt.to_period('M')
    t_final1=pd.merge(t_final,ti_final,on='MOIS',how='left')
    t_final1=t_final1.drop(columns=['ANNEE'])
    t_final1=t_final1.drop(columns=['date'])
    t_final1['taux neutre']=2
    t_final1['taux longterme']=taux_longterme()
    return t_final1, ti_final
    #il reste donc a calculer le Y=bX , ouY et X ont ete decrit sur une feuille, juste des soustractions vectorisées#
def tab_regression_lin_simple(taux):
    tab=nettoyage()
    if taux=="taux neutre":
        q=str(input("Voulez vous utiliser la veleur par défaut du taux directeur neutre (taper oui) ou une valeur autre (taper non) :")).upper()   #faire methode securisée!!!!
        if q!="OUI":
            neutre=float(input("Saisir votre valeur du taux neutre :"))
        else:
            neutre=2
        tab["X"]=tab['inflation']-2-tab['valeurs output gap']
        tab["Y"]=tab['Rate']-neutre-tab["valeurs output gap"]-tab['inflation']
        return tab["X"],tab["Y"]
def tab_regression_lin_multiple(taux):
    tab=nettoyage()
    if taux=="taux neutre":
        q=str(input("Voulez vous utiliser la veleur par défaut du taux directeur neutre (taper oui) ou une valeur autre (taper non) :")).upper()   #faire methode securisée!!!!
        if q!="OUI":
            neutre=float(input("Saisir votre valeur du taux neutre :"))
        else:
            neutre=2
    else:
        neutre=tab['taux longterme']
    tab["Y"]=tab["Rate"]-tab['inflation']-neutre
    tab["X1"]=tab["inflation"]-2
    tab["X2"]=tab['valeurs output gap']
    tab['Rate_lag'] = tab['Rate'].shift(1)
    modele=smf.ols(formula='Y ~ X1 + X2 ',data=tab)  #+ Rate_lag 
    resultat=modele.fit()
    #infos=resultat.summary()
    return resultat
        
def poly_regression(reg_lin):
    p=reg_lin
    return np.polyfit(np.array(p[0]),np.array(p[1]),1)
def coeff_determination(taux,reg_lin):
    tab=reg_lin
    moy_y=tab[1].mean()
    poly=poly_regression(tab)
    y_predi=np.polyval(poly,tab[0])
    residu=np.sum((tab[1]-y_predi)**2)
    diff=np.sum((tab[1]-moy_y)**2)
    return 1-residu/diff
def tracer_afficher():
    tab=tab_regression_lin_simple('taux neutre')
    r2=coeff_determination("taux neutre",tab)
    poly=poly_regression(tab)
    t=nettoyage()
    y=np.array(t['Rate'])
    x=np.array(t['inflation']+2+poly[0]*(t['inflation']-2)+(1-poly[0])*(t['valeurs output gap']))
    plt.scatter(x,y,alpha=0.7)
    plt.grid()
    y_pred=np.polyval(poly,tab[0])
    plt.plot(tab[0],y_pred)
    print("r2=",r2)
    plt.show()
    difer=np.sum((t['Rate']-(poly[0]*(t['inflation']-2-t['valeurs output gap'])+2+t['valeurs output gap']+t['inflation']))**2)
    return difer
# def tracer_afficher_multiple():
#     tab=tab_regression_lin_simple('taux neutre')
#     r2=coeff_determination("taux neutre",tab)
#     poly=poly_regression(tab)
#     t=nettoyage()
#     y=np.array(t['Rate'])
#     x=np.array(t['inflation']+2+poly[0]*(t['inflation']-2)+(1-poly[0])*(t['valeurs output gap']))
#     plt.scatter(x,y,alpha=0.7)
#     plt.grid()
#     y_pred=np.polyval(poly,tab[0])
#     plt.plot(tab[0],y_pred)
#     print("r2=",r2)
#     plt.show()
    
print(nettoyage())
#print(tab_regression_lin_multiple('longterme'))
        
        




