from sklearn.model_selection import train_test_split
import pandas as pd
from pandas.plotting import scatter_matrix
import mglearn
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import numpy as np
import os
import sys


if __name__ == '__main__':
    figFolder = './Figures/'
    if not os.path.exists(figFolder):
        os.makedirs(figFolder)

    df_orig = pd.read_csv('./Data/test_assignment_sim.csv')

    print("\nTesting Extrpolation performance of SVR on all sites\n")
    nBelow90 = 0
    nBelow95 = 0
    tools = []
    sites = []
    best_C = []
    best_gamma =[]
    best_scores = []
    test_scores = []
    param_grid = {'C': [1,10,1000], 'gamma': [0.1,0.2,0.3,1]}
    data = []
    for t in range(1,5):
        mask = df_orig['TOOL'] == t
        df = df_orig[mask]
        for i in range(49):
            tools.append(t)
            sites.append(i)
            scaler = StandardScaler()
            scaler.fit(df.loc[:,:'DEP TIME'])
            scaled_array = scaler.transform(df.loc[:,:'DEP TIME'])
            colnames = ['FLOWFACTOR','SPACING','DEP TIME']
            scaled_data = pd.DataFrame(scaled_array,columns=colnames)
            scaled_data['SITE_{}'.format(i)] = df['SITE_{}'.format(i)].to_numpy()
            mask2 = np.sqrt(scaled_data['FLOWFACTOR']**2 + scaled_data['SPACING']**2 + scaled_data['DEP TIME']**2)<2
            train = scaled_data[mask2]
            test = scaled_data[~mask2]
            valid_param=test.sample(frac=0.5, replace=False)
            test_real=test.drop(valid_param.index)

            score_list = []
            for c in [0.01,0.1,1,10,100,1000]:
                for g in [0.01,0.1,1,10,100,1000]:
                    # print("C = {} , gamma = {}".format(c,g))
                    svr = SVR(kernel='rbf',C=c,gamma=g).fit(train.loc[:,:'DEP TIME'],train['SITE_{}'.format(i)])
                    scores = []
                    for s in range(10):
                        test_sample = valid_param.sample(frac=0.7, replace=False)
                        scores.append(svr.score(test_sample.loc[:,:'DEP TIME'],test_sample['SITE_{}'.format(i)]))
                    score_list.append([c,g,np.mean(scores)])
            score_frame = pd.DataFrame(score_list,columns=['C','gamma','score'])
            row_max = score_frame.loc[score_frame['score'].idxmax()]
            c_max = row_max['C']
            gamma_max = row_max['gamma']
            max_score = row_max['score']

            svr = SVR(kernel='rbf',C=c_max,gamma=gamma_max).fit(train.loc[:,:'DEP TIME'],train['SITE_{}'.format(i)])
            test_score = svr.score(test_real.loc[:,:'DEP TIME'],test_real['SITE_{}'.format(i)])

            print(t,i,c_max,gamma_max,max_score,test_score)
            data.append([t,i,c_max,gamma_max,max_score,test_score])



    results = pd.DataFrame(data,columns=['TOOL','SITE','best_C','best_gamma','best_score','test_score'])

    results['test_score'].plot.hist(bins=30,range=(0.5,1),histtype='step')
    plt.savefig(figFolder + 'SVR_extrapolation_performance.png', bbox_inches='tight')
    plt.close()

    H,bins = np.histogram(results['best_C'])
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    results[['best_C']].plot.hist(bins=30,histtype='step',logx=True)
    # plt.xscale('log')
    plt.savefig(figFolder + 'SVR_extrapolation_best_C.png', bbox_inches='tight')
    plt.close()

    H,bins = np.histogram(results['best_gamma'])
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    results[['best_gamma']].plot.hist(bins=30,histtype='step',logx=True)
    # plt.xscale('log')
    plt.savefig(figFolder + 'SVR_extrapolation_best_gamma.png', bbox_inches='tight')
    plt.close()

    cmap = mpl.colors.ListedColormap(["navy", "crimson", "limegreen", "gold"])
    norm = mpl.colors.BoundaryNorm(np.arange(0.5,5), cmap.N)
    handles = [plt.plot([],[],color=cmap(i-1), ls="", marker=".", \
                    markersize=np.sqrt(10))[0] for i in range(1,5)]
    labels=["TOOL 1", "TOOL 2", "TOOL 3", "TOOL 4"]

    fig, ax = plt.subplots()
    results.plot.scatter(x='SITE', y='test_score',c='TOOL',marker='o', s=60,\
    alpha=.8,cmap=cmap, norm=norm, colorbar=False)
    plt.legend(handles, labels, loc=(1.02,0))
    plt.savefig(figFolder + 'SVR_extrapolation_score_scatter.png', bbox_inches='tight')

    fig, ax = plt.subplots()
    results.plot.scatter(x='best_C', y='best_gamma',c='TOOL',marker='o', s=60,\
    alpha=.8,cmap=cmap, norm=norm, colorbar=False,logx=True,logy=True)
    plt.legend(handles, labels, loc=(1.02,0))
    plt.savefig(figFolder + 'SVR_extrapolation_bestparams_scatter.png', bbox_inches='tight')
