import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

import os
import json
from collections import defaultdict

from matplotlib import rc
rc('text', usetex=True)
plt.rc('font', family='serif')#,size=12)

fig, ax = plt.subplots(figsize=(4.47 * 1.78, 2 * 1.78))
# fig, ax = plt.subplots(figsize=(4.47, 7))

def dimensions_from_filename(filename):
    tokens = filename.split('_')
    inner_dim = int(tokens[2])
    outer_dim = int(tokens[3])
    iteration = int(tokens[4].split('.')[0])

    return inner_dim, outer_dim, iteration

def get_error_from_training_results(training_results):
    ae = training_results['autoencoder']
    stats = list(ae.values()).pop()
    return stats['max_distance_error']

def load_dataset(folder):
    filenames = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    
    all_iteration_counts = [dimensions_from_filename(f)[2] for f in filenames] 
    num_its_per_sample = max(all_iteration_counts)
    num_samples = len(filenames) / num_its_per_sample

    # Need to aggragate the errors if we did multiple runs per dim combination
    dims_to_errors = defaultdict(float)
    for filename in filenames:
        inner_dim, outer_dim, iteration = dimensions_from_filename(filename)

        with open(os.path.join(folder, filename), 'r') as f:
            training_results = json.load(f)
            error = get_error_from_training_results(training_results) 

        val = dims_to_errors[(inner_dim, outer_dim)]
        # dims_to_errors[(inner_dim, outer_dim)] = max(error, val)
        dims_to_errors[(inner_dim, outer_dim)] += error / num_its_per_sample

    inner_dims = []
    outer_dims = []
    errors = []
    tups = []
    for (inner_dim, outer_dim), error in dims_to_errors.items():
            inner_dims.append(inner_dim)
            outer_dims.append(outer_dim)
            errors.append(error)

            tups.append((inner_dim, outer_dim, error))

    max_inner_dim = max(inner_dims)
    max_outer_dim = max(outer_dims)
    tups.sort()
    # errs = np.array([t[2] for t in tups])
    # errs = errs.reshape((max_inner_dim, max_outer_dim))

    df = pd.DataFrame(tups,columns=['inner_dim', 'outer_dim', 'error'])
    # df = df.pivot('inner_dim', 'outer_dim', 'error')

    return  df #errs #np.array(inner_dims), np.array(outer_dims), np.array(errors)


def main():
    # dataset_path = 'X/'
    dataset_path = 'mounted_bar_3_its/'
    # dataset_path = 'mounted_bar_no_averaging/'
    # inner_dims, outer_dims, errors = load_dataset(dataset_path)
    errs_df = load_dataset(dataset_path)



    # errs_df = errs_df.pivot('inner_dim', 'outer_dim', 'error')
    # errs_df = errs_df.pivot('outer_dim', 'inner_dim', 'error')
    errs_df = errs_df.pivot('inner_dim', 'outer_dim', 'error')
    errs_df = np.log(errs_df)
    # errs_df = errs_df.filter(items=range(3,11), axis=1) # X
    # errs_df = errs_df.filter(items=range(6,31), axis=0) # X
    errs_df = errs_df.filter(items=range(3,16), axis=0)
    errs_df = errs_df.filter(items=range(6,31), axis=1)
    errs_df = errs_df.iloc[::-1]
    print(errs_df)
    # errs_df = errs_df.iloc[5:29,5:29]
    # errs_df = np.maximum(errs_df, -4)
    # errs_df = errs_df.applymap(lambda x: 1 if x < 0.0135 else x)
    sns.heatmap(errs_df, ax=ax,square=True,xticklabels = 4, yticklabels=4)#, linewidth=0.5)

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_trisurf(errs_df.inner_dim, errs_df.outer_dim, np.log(errs_df.error), cmap=cm.jet, linewidth=0.2)

    ax.set_xlabel(r'Outer Dim')
    ax.set_ylabel(r'Inner Dim')
    # ax.locator_params(axis='x', nbins=10)
    # ax.locator_params(axis='y', nbins=10)

    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    fig.savefig('bar.svg')
    plt.show()


    # print(inner_dims)
    # print(outer_dims)
    # print(errors)
    # y = outer_dims
    # x = inner_dims
    # z = errors
    # generate 2 2d grids for the x & y bounds
    # y, x = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    # z = (1 - x / 2. + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
    # # x and y are bounds, so z should be the value *inside* those bounds.
    # # Therefore, remove the last value from the z array.
    # print(x)
    # z = z[:-1, :-1]
    # z_min, z_max = -np.abs(z).max(), np.abs(z).max()
    # fig, ax = plt.subplots()
    # c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
    # ax.set_title('pcolormesh')
    # # set the limits of the plot to the limits of the data
    # ax.axis([x.min(), x.max(), y.min(), y.max()])
    # fig.colorbar(c, ax=ax)
    # plt.show()

if __name__ == '__main__':
    main()
