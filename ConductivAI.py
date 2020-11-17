from sklearn.model_selection import train_test_split
import pandas as pd
from pandas.plotting import scatter_matrix
import mglearn
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import numpy as np
import os
import sys


if __name__ == '__main__':
    figFolder = './Figures/'
    if not os.path.exists(figFolder):
        os.makedirs(figFolder)
    # df = pd.read_csv('./Data/test_assignment_sim.csv',\
    # usecols= ['FLOWFACTOR','SPACING','DEP TIME','TOOL','SITE_0'])

    coordinates = pd.read_csv('./Data/site_coordinates.csv')
    # coordinates.plot.scatter('SITE_X','SITE_Y')
    # plt.show()
    coordinates['SITE_N'] = np.arange(0,49)

    coordinates.plot.scatter(x='SITE_X', y='SITE_Y',c='SITE_N',marker='o', s=60,\
    alpha=.8)
    plt.savefig(figFolder + 'coordinates_scatter.png', bbox_inches='tight')

    df_orig = pd.read_csv('./Data/test_assignment_sim.csv')
    mask = df_orig['TOOL'] == 1
    df = df_orig[mask]

    pd.plotting.scatter_matrix(df.loc[:,:'DEP TIME'],\
    figsize=(8,8),marker='o',hist_kwds={'bins': 20}, s=60,\
    alpha=.8)
    plt.savefig(figFolder + 'parameter_matrix.png', bbox_inches='tight')
    plt.close()
    sys.exit()

    for i in range(49):
        fig, axes = plt.subplots(1, 3, sharey=True, figsize=(16,9), squeeze=True)
        df.plot.hexbin(x='FLOWFACTOR', y='SITE_{}'.format(i), gridsize=15, bins='log',ax=axes[0])
        df.plot.hexbin(x='SPACING', y='SITE_{}'.format(i), gridsize=15, bins='log',ax=axes[1])
        df.plot.hexbin(x='DEP TIME', y='SITE_{}'.format(i), gridsize=15, bins='log',ax=axes[2])
        plt.savefig(figFolder + 'Correlation_site_{}.png'.format(i), bbox_inches='tight')
        plt.close()

        fig, axes = plt.subplots(1, 3, sharey=True, figsize=(16,9), squeeze=True)
        df.plot.scatter(x='FLOWFACTOR', y='SITE_{}'.format(i),ax=axes[0])
        df.plot.scatter(x='SPACING', y='SITE_{}'.format(i),ax=axes[1])
        df.plot.scatter(x='DEP TIME', y='SITE_{}'.format(i),ax=axes[2])
        plt.savefig(figFolder + 'Correlation_site_{}.png'.format(i), bbox_inches='tight')
        plt.close()



    print("\nTesting Linear Models on all sites\n")

    data = []
    param_grid = {'alpha': [0.01,0.1,1,10,100,1000]}
    for t in range(1,5):
        mask = df_orig['TOOL'] == t
        df = df_orig[mask]
        for i in range(49):
            lr = LinearRegression()
            scores = cross_val_score(lr,df.loc[:,:'DEP TIME'],df['SITE_{}'.format(i)], cv = 5)
            lr_score = scores.mean()

            train, test = train_test_split(df[['FLOWFACTOR','SPACING','DEP TIME','SITE_{}'.format(i)]], test_size=0.2)
            grid_search = GridSearchCV(Ridge(),param_grid,cv=5)
            grid_search.fit(train.loc[:,:'DEP TIME'],train['SITE_{}'.format(i)])
            ridge_score = grid_search.score(test.loc[:,:'DEP TIME'],test['SITE_{}'.format(i)])

            grid_search = GridSearchCV(Lasso(),param_grid,cv=5)
            grid_search.fit(train.loc[:,:'DEP TIME'],train['SITE_{}'.format(i)])
            lasso_score = grid_search.score(test.loc[:,:'DEP TIME'],test['SITE_{}'.format(i)])

            print(t,i,lr_score,ridge_score,lasso_score)
            data.append([t,i,lr_score,ridge_score,lasso_score])

    results = pd.DataFrame(data,columns=['TOOL','SITE','lr_score','ridge_score','lasso_score'])

    results[['lr_score','ridge_score','lasso_score']].plot.hist(bins=30,range=(0.5,1),histtype='step')
    plt.savefig(figFolder + 'linear_mode_performance.png', bbox_inches='tight')
    plt.close()

    cmap = mpl.colors.ListedColormap(["navy", "crimson", "limegreen", "gold"])
    norm = mpl.colors.BoundaryNorm(np.arange(0.5,5), cmap.N)
    handles = [plt.plot([],[],color=cmap(i-1), ls="", marker=".", \
                    markersize=np.sqrt(10))[0] for i in range(1,5)]
    labels=["TOOL 1", "TOOL 2", "TOOL 3", "TOOL 4"]

    fig, ax = plt.subplots()
    results.plot.scatter(x='SITE', y='lr_score',c='TOOL',marker='o', s=60,\
    alpha=.8,cmap=cmap, norm=norm, colorbar=False)
    plt.legend(handles, labels, loc=(1.02,0))
    plt.savefig(figFolder + 'lr_score_scatter.png', bbox_inches='tight')

    fig, ax = plt.subplots()
    results.plot.scatter(x='SITE', y='ridge_score',c='TOOL',marker='o', s=60,\
    alpha=.8,cmap=cmap, norm=norm, colorbar=False)
    plt.legend(handles, labels, loc=(1.02,0))
    plt.savefig(figFolder + 'ridge_score_scatter.png', bbox_inches='tight')

    fig, ax = plt.subplots()
    results.plot.scatter(x='SITE', y='lasso_score',c='TOOL',marker='o', s=60,\
    alpha=.8,cmap=cmap, norm=norm, colorbar=False)
    plt.legend(handles, labels, loc=(1.02,0))
    plt.savefig(figFolder + 'lasso_score_scatter.png', bbox_inches='tight')

    print("\nTesting SVR on all sites\n")

    param_grid = {'C': [0.1,1,10,100,1000], 'gamma': [0.1,1,10,100]}
    data = []
    for t in range(1,5):
        mask = df_orig['TOOL'] == t
        df = df_orig[mask]
        for i in range(49):
            line = [t,i]
            for method in ['linear', 'rbf', 'sigmoid']:
                for s in ['MinMax','Standard']:
                    train, test = train_test_split(df[['FLOWFACTOR','SPACING','DEP TIME','SITE_{}'.format(i)]], test_size=0.2)
                    if s== 'MinMax':
                        scaler = MinMaxScaler()
                    else:
                        scaler = StandardScaler()
                    scaler.fit(train.loc[:,:'DEP TIME'])
                    scaled_train = scaler.transform(train.loc[:,:'DEP TIME'])
                    scaled_test = scaler.transform(test.loc[:,:'DEP TIME'])
                    grid_search = GridSearchCV(SVR(kernel=method),param_grid,cv=5)
                    grid_search.fit(scaled_train,train['SITE_{}'.format(i)])
                    line.append(grid_search.score(scaled_test,test['SITE_{}'.format(i)]))
                    line.append(grid_search.best_params_['C'])
                    line.append(grid_search.best_params_['gamma'])

            data.append(line)


    results = pd.DataFrame(data,columns=['TOOL','SITE','MinMax_linear_score','MinMax_linear_best_C','MinMax_linear_best_gamma',
                                        'Standard_linear_score','Standard_linear_best_C','Standard_linear_best_gamma',
                                        'MinMax_rbf_score','MinMax_rbf_best_C','MinMax_rbf_best_gamma',
                                        'Standard_rbf_score','Standard_rbf_best_C','Standard_rbf_best_gamma',
                                        'MinMax_sigmoid_score','MinMax_sigmoid_best_C','MinMax_sigmoid_best_gamma',
                                        'Standard_sigmoid_score','Standard_sigmoid_best_C','Standard_sigmoid_best_gamma',])

    results[['MinMax_linear_score','Standard_linear_score','MinMax_rbf_score',
            'Standard_rbf_score','MinMax_sigmoid_score','Standard_sigmoid_score']].plot.hist(bins=30,range=(0.5,1),histtype='step')
    plt.savefig(figFolder + 'SVR_performance.png', bbox_inches='tight')
    plt.close()

    results[['Standard_rbf_best_C']].plot.hist(bins=30,histtype='step')
    plt.savefig(figFolder + 'SVR_best_C.png', bbox_inches='tight')
    plt.close()

    results[['Standard_rbf_best_gamma']].plot.hist(bins=30,histtype='step')
    plt.savefig(figFolder + 'SVR_best_gamma.png', bbox_inches='tight')
    plt.close()

    cmap = mpl.colors.ListedColormap(["navy", "crimson", "limegreen", "gold"])
    norm = mpl.colors.BoundaryNorm(np.arange(0.5,5), cmap.N)
    handles = [plt.plot([],[],color=cmap(i-1), ls="", marker=".", \
                    markersize=np.sqrt(10))[0] for i in range(1,5)]
    labels=["TOOL 1", "TOOL 2", "TOOL 3", "TOOL 4"]

    fig, ax = plt.subplots()
    results.plot.scatter(x='SITE', y='MinMax_linear_score',c='TOOL',marker='o', s=60,\
    alpha=.8,cmap=cmap, norm=norm, colorbar=False)
    plt.legend(handles, labels, loc=(1.02,0))
    plt.savefig(figFolder + 'MinMax_linear_score_scatter.png', bbox_inches='tight')

    fig, ax = plt.subplots()
    results.plot.scatter(x='SITE', y='Standard_linear_score',c='TOOL',marker='o', s=60,\
    alpha=.8,cmap=cmap, norm=norm, colorbar=False)
    plt.legend(handles, labels, loc=(1.02,0))
    plt.savefig(figFolder + 'Standard_linear_score_scatter.png', bbox_inches='tight')

    fig, ax = plt.subplots()
    results.plot.scatter(x='SITE', y='MinMax_rbf_score',c='TOOL',marker='o', s=60,\
    alpha=.8,cmap=cmap, norm=norm, colorbar=False)
    plt.legend(handles, labels, loc=(1.02,0))
    plt.savefig(figFolder + 'MinMax_rbf_score_scatter.png', bbox_inches='tight')

    fig, ax = plt.subplots()
    results.plot.scatter(x='SITE', y='Standard_rbf_score',c='TOOL',marker='o', s=60,\
    alpha=.8,cmap=cmap, norm=norm, colorbar=False)
    plt.legend(handles, labels, loc=(1.02,0))
    plt.savefig(figFolder + 'Standard_rbf_score_scatter.png', bbox_inches='tight')

    fig, ax = plt.subplots()
    results.plot.scatter(x='SITE', y='MinMax_sigmoid_score',c='TOOL',marker='o', s=60,\
    alpha=.8,cmap=cmap, norm=norm, colorbar=False)
    plt.legend(handles, labels, loc=(1.02,0))
    plt.savefig(figFolder + 'MinMax_sigmoid_score_scatter.png', bbox_inches='tight')

    fig, ax = plt.subplots()
    results.plot.scatter(x='SITE', y='Standard_sigmoid_score',c='TOOL',marker='o', s=60,\
    alpha=.8,cmap=cmap, norm=norm, colorbar=False)
    plt.legend(handles, labels, loc=(1.02,0))
    plt.savefig(figFolder + 'Standard_sigmoid_score_scatter.png', bbox_inches='tight')

    fig, ax = plt.subplots()
    results.plot.scatter(x='Standard_rbf_best_C', y='Standard_rbf_best_gamma',c='TOOL',marker='o', s=60,\
    alpha=.8,cmap=cmap, norm=norm, colorbar=False)
    plt.legend(handles, labels, loc=(1.02,0))
    plt.savefig(figFolder + 'Standard_rbf_bestparams_scatter.png', bbox_inches='tight')
