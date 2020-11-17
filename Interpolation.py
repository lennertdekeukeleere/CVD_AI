import numpy as np
from scipy.interpolate import interp2d
from scipy import interpolate
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import griddata
import matplotlib as m
import math as ma
from matplotlib import cm
import sys
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.spatial import Delaunay
import os
import pandas as pd
np.set_printoptions(threshold=50000)

if __name__ == '__main__':
    figFolder = './Figures/'

    if not os.path.exists(figFolder):
        os.makedirs(figFolder)
    # df = pd.read_csv('./Data/test_assignment_sim.csv',\
    # usecols= ['FLOWFACTOR','SPACING','DEP TIME','TOOL','SITE_0'])
    coordinates = pd.read_csv('./Data/site_coordinates.csv')
    # coordinates.plot.scatter('SITE_X','SITE_Y')
    # plt.show()

    X = coordinates['SITE_X']*10**(-3)
    Y = coordinates['SITE_Y']*10**(-3)
    min_x = X.min()
    max_x = X.max()
    min_y = Y.min()
    max_y = Y.max()

    df = pd.read_csv('./Data/test_assignment_sim.csv')
    test_thickness = df.loc[0,'SITE_0':]*10**(-4)
    min_thickness = test_thickness.min()
    max_thickness = test_thickness.max()

    grid_x, grid_y = np.mgrid[min_x:max_x:50j, min_y:max_y:50j]
    grid = griddata((X,Y), test_thickness, (grid_x, grid_y), method='cubic')


    print(grid.shape)
    x_reshape = np.reshape(grid_x,50*50)
    y_reshape = np.reshape(grid_y,50*50)
    grid_reshape = np.reshape(grid,50*50)
    print(x_reshape)
    print(y_reshape)
    # print(grid_reshape)
    idx = np.isnan(grid_reshape)
    grid_reshape[idx] = 0
    # print(grid_reshape)

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.set_xlim(min_x,max_x)
    ax.set_ylim(min_y,max_y)
    ax.set_xlabel('X [mm]',fontsize=16)
    ax.set_ylabel('Y [mm]',fontsize=16)
    # plt.imshow(grid.T,cmap = cmap, extent=(-LGSize/2.,LGSize/2.,-LGSize/2.,LGSize/2.), origin='lower',aspect='auto',clim=(20,27))

    plt.imshow(grid.T,cmap = plt.cm.inferno, extent=(min_x,max_x,min_y,max_y),
                origin='lower',aspect='auto',clim=(min_thickness,max_thickness))
    cb = plt.colorbar()
    cb.ax.set_ylabel('Thickness [um]', rotation=90,fontsize=16)
    plt.savefig(figFolder + 'test_interpolation.png', bbox_inches='tight')

    sys.exit()

    grid_x_2, grid_y_2 = np.mgrid[-x_range/2:x_range/2:(int(x_range/10)+1)*1j, -y_range/2:y_range/2:(int(y_range/10)+1)*1j]

    # Interpolation:
    nodes = np.vstack((x, y)).T
    tri = Delaunay(nodes)
    CT_interpolator = CloughTocher2DInterpolator(tri, gain)

    x_reshape = np.reshape(grid_x_2,(int(y_range/10)+1)*(int(y_range/10)+1))
    y_reshape = np.reshape(grid_y_2,(int(y_range/10)+1)*(int(y_range/10)+1))

    gain_interpolated = CT_interpolator(x_reshape, y_reshape)
