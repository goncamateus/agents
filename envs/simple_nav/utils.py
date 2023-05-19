from scipy.stats import multivariate_normal


def gaussian_activation(x, y, xmean, ymean, x_var=1, xy_cov=0, yx_cov=0, y_var=1):
    """
    Return the value for a 2d gaussian distribution with mean at [xmean, ymean] and the covariance matrix based on
    [[x_var, xy_cov],[yx_cov, y_var]].
    """
    means = [xmean, ymean]
    cov_mat = [[x_var, xy_cov], [yx_cov, y_var]]

    rv = multivariate_normal(means, cov_mat)

    return rv.pdf([x, y])


def min_max_norm(val, min, max):
    return (val - min) / (max - min)
