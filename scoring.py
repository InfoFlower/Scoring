################################################################################################################################################
############ Modules importés
################################################################################################################################################

from sklearn.tree import plot_tree
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV

################################################################################################################################################
############ FONCTION POUR CALCULE ET PREDICTION DES MODELES
################################################################################################################################################

def score(X_train,X_test,y_train,y_test,model):
    """
    Affichage des metrics
    """
    return {'accuracy' :        round(float(metrics.precision_score(y_test,model.predict(X_test))),2),
                   'auc' :      round(float(metrics.roc_auc_score(y_test,model.predict(X_test))),2),
                   'Gauc' :      round(float(metrics.roc_auc_score(y_test,model.predict_proba(X_test)[:,1], average='weighted')),2),
                   'f1-score' : round(float(metrics.f1_score(y_test,model.predict(X_test))),2)}

def score_cv(X_train,X_test,y_train,y_test,trained_model):
    return {'best params': trained_model.best_params_,
            'associed f1-score': score(X_train,X_test,y_train,y_test,trained_model)}


def all_calc(df,
             models=[(DecisionTreeClassifier(),'Arbre de décision'),(LogisticRegression(),'Logistique')],
             top_cv=False,
             test_size=0.33,
             rnd_state=11,
             new_data=None,
             cols=['workclass', 'education', 'marital-status', 'occupation','relationship', 'race', 'gender','native-country', 'income'],
             plot=False):
    """
    Segmente le jeu de données en train test size si besoin
    puis peut faire les predictions ou les scores dépendamment du besoin
    """
    df=pd.get_dummies(df,columns=cols)
    X=df[df.columns[:-2]]
    y=df['income_>50K']
    if new_data is None:
        X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=test_size,random_state=rnd_state)
    else:
        X_train,y_train=(X,y)
    ret={}
    for i in models:
        i[0].fit(X_train,y_train)
        if new_data is None:
            if top_cv==True:
                ret[i[1]]=score_cv(X_train,X_test,y_train,y_test,i[0])
            else:
                ret[i[1]]=score(X_train,X_test,y_train,y_test,i[0])
        if new_data is not None:
            ret[i[1]]=[i[0].predict(pd.get_dummies(new_data,columns=cols[:-1])),i[1]]
        if plot == True and i[1]=='Arbre de décision':
            plot_tree(i[0])
    return ret

################################################################################################################################################
############ FONCTION POUR TRAITEMENT DE DONNEES
################################################################################################################################################


def randint_exc(liste, exception):
    """
    Permet de faire un choix aléatoire dans une liste mis à part exception
    """
    res = random.choice(liste)
    while res == exception:
        res = random.choice(liste)
    return res

def repartition(df,i,interest):
    """
    Permet de savoir les modalitées à regrouper
    Puis calcule la répartition de chaques modalitées dépendamment de la variable expliquée pour la regle de decision
    """
    a_reg = []
    for m in df[i].unique():
        if sum(df[i] == m)/len(df[i]) <   0.05: # Si la modalité est inférieur au seuil
            a_reg.append(m) #On la met dans notre liste de modalités à regrouper
    cache = pd.crosstab(df[i], df[interest[0]])[interest[1]]# On calcule la répartition de chaques modalités dépendamment de la variable expliquée
    for ii in cache.index:
        cache[ii] = round(cache[ii] / sum(df[i] == ii), 3) # On divise par son nombre pour avoir une proportion et pr pouvoir comparer
    return cache,a_reg


def regroupement(df, sup, interest):
    """
    Fonction permettant de construire les dictionnaires de regroupement de variables à l'aide de la règle de décision suivante:
        regroupe une variable sous le seuil de rareté 0.05 avec 
        la variable ayant la répartition la plus semblable dépendamment 
        de la variable expliquée
    """
    regroup = {}
    for i in df.columns:
        if i not in sup:
            cache,a_reg=repartition(df,i,interest)
            reg = {}
            for sav in a_reg:
                val = cache[sav]
                savsav = randint_exc(df[i].unique(),sav) #Ici on choisi une modalité au hasard pour initialiser le minimum (on exlu la modalité analysée)
                for m in cache.index:
                    if abs(val - cache[m]) < abs(val - cache[savsav]) and sav!=m: #ici regle de décision
                        savsav = m
                reg[sav] = savsav
            regroup[i] = reg
    return regroup


def regroupement_dataframe(df,dico):
    """
    permet de faire le regroupement
    """
    for k,v in dico.items():
        for kk,vv in v.items():
            df[k]=df[k].replace(kk,vv)
    return df

def clean_df(df,too_much):
    """
    fonction permettant de clean le dataset, supprimer les colonnes non voulues... etc
    c'est ici qu'on fait le choix de supprimer toutes les personnes ne travaillant pas
    """
    return df.replace('?', np.nan).dropna().reset_index(drop=True).drop(too_much, axis=1)


def process_datacleaning(df,dico):
    """
    permet de faire le regroupement de variables en une fonction et en répétant le replace au cas ou
    """
    df_reg=df.copy()
    #df_reg=clean_df(df_reg,sup_var)
    for i in range(2):
        df_reg=regroupement_dataframe(df_reg,dico)
    return df_reg

################################################################################################################################################
############ FONCTION POUR REPRESENTATION DE VARIABLES
################################################################################################################################################

def show_repartition(df_regroup):
    """
    Montre la répartition des individus dans les modalités de chaque variable
    """
    for i in df_regroup.columns:
        col=i
        index= df_regroup[col].value_counts().index
        values=df_regroup[col].value_counts()
        plt.bar(index,values)
        plt.axhline(len(df_regroup)*0.05,color='red')
        print(i)
        plt.show()

def fancy_dico(dico):
    """
    Fonction pour rendre plus compréhensible la lecture des dictionnaires 
    de dictionnaires ou les listes de dictionnaires de dictionnaitres
    """
    rep=''
    if type(dico)==list:
        for dicoco in range(len(dico)):
            rep+=f"Itération numéro {dicoco + 1}" +'\n'
            for key,values in dico[dicoco].items():
                rep+=' '*5+key+'\n'
                for key,value in values.items():
                    rep+=' '*10+f'{key} : {value} '+'\n'
    else:
        for key,values in dico.items():
            rep+=key+'\n'
            for key,value in values.items():
                rep+=' '*5+f'{key} : {value} '+'\n'
    return rep