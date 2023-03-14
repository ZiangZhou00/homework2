import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.preprocessing import PolynomialFeatures


def generate_q1_data(N, pdf_params):
    n = 2

    # Get random samples from a uniform distribution to assign to each class
    u = np.random.rand(N)
    labels = u >= pdf_params['priors'][0]

    X = np.zeros((N, n))
    for i in range(N):
        if labels[i] == 0:
            # Choose randomly from the two Gaussians in Class 0
            if np.random.rand() < pdf_params['gmm_a'][0]:
                X[i, :] = np.random.multivariate_normal(pdf_params['mu'][0], pdf_params['Sigma'][0])
            else:
                X[i, :] = np.random.multivariate_normal(pdf_params['mu'][1], pdf_params['Sigma'][1])
        else:
            # Choose randomly from the two Gaussians in Class 1
            if np.random.rand() < pdf_params['gmm_a'][2]:
                X[i, :] = np.random.multivariate_normal(pdf_params['mu'][2], pdf_params['Sigma'][2])
            else:
                X[i, :] = np.random.multivariate_normal(pdf_params['mu'][3], pdf_params['Sigma'][3])

    return X, labels


def create_prediction_score_grid(bounds_X, bounds_Y, params, prediction_function, phi=None, num_coords=200):
    # Note that I am creating a 200x200 rectangular grid
    xx, yy = np.meshgrid(np.linspace(bounds_X[0], bounds_X[1], num_coords),
                         np.linspace(bounds_Y[0], bounds_Y[1], num_coords))

    # Flattening grid and feed into a fitted transformation function if provided
    grid = np.c_[xx.ravel(), yy.ravel()]
    if phi:
        grid = phi.transform(grid)

    # Z matrix are the predictions given the provided model parameters
    Z = prediction_function(grid, params).reshape(xx.shape)

    return xx, yy, Z


def generate_gmm_data(N, gmm_pdf):
    # Generates N vector samples from the specified mixture of Gaussians
    # Returns samples and their component labels
    # Data dimensionality is determined by the size of mu/Sigma parameters

    # Decide randomly which samples will come from each component
    u = np.random.random(N)
    thresholds = np.cumsum(gmm_pdf['priors'])
    thresholds = np.insert(thresholds, 0, 0)  # For intervals of classes

    n = gmm_pdf['mu'].shape[0]  # Data dimensionality

    X = np.zeros((N, n))
    C = len(gmm_pdf['priors'])  # Number of components
    for i in range(C + 1):
        # Get randomly sampled indices for this Gaussian, checking between thresholds based on class priors
        indices = np.argwhere((thresholds[i - 1] <= u) & (u <= thresholds[i]))[:, 0]
        # No. of samples in this Gaussian
        X[indices, :] = multivariate_normal.rvs(gmm_pdf['mu'][i - 1], gmm_pdf['Sigma'][i - 1], len(indices))

    return X[:, 0:2], X[:, 2]


def generate_q2_data(N, dataset_name):
    gmm_pdf = {}
    gmm_pdf['priors'] = np.array([.3, .4, .3])
    gmm_pdf['mu'] = np.array([[-10, 0, 10], [0, 0, 0], [10, 0, -10]])  # Gaussian distributions means
    gmm_pdf['Sigma'] = np.array([[[1, 0, -3], [0, 1, 0], [-3, 0, 15]], [[8, 0, 0], [0, .5, 0], [0, 0, .5]],
                                 [[1, 0, -3], [0, 1, 0], [-3, 0, 15]]])  # Gaussian distributions covariance matrices

    X, y = generate_gmm_data(N, gmm_pdf)

    # Plot the original data and their true labels
    fig = plt.figure(figsize=(10, 10))

    ax_raw = fig.add_subplot(111, projection='3d')

    ax_raw.scatter(X[:, 0], X[:, 1], y, marker='o', color='b')
    ax_raw.set_xlabel(r"$x_1$")
    ax_raw.set_ylabel(r"$x_2$")
    ax_raw.set_zlabel(r"$y$")
    # Set equal axes for 3D plots
    ax_raw.set_box_aspect((np.ptp(X[:, 0]), np.ptp(X[:, 1]), np.ptp(y)))

    plt.title("{} Dataset".format(dataset_name))
    plt.tight_layout()
    plt.show()

    return X, y
# Reference from MARK ZOLOTAS