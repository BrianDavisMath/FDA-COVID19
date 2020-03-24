import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
from pathos.multiprocessing import ProcessingPool as Pool


def mixed_metric(features_1_, features_2_, cat_weight_, bool_features_=None):
    """
    A distance function that uses Jaccard distance for binary features and Euclidean distance for continuous features.
    The mixed metric is the root of sum of squares of the distance on the binary and continuous features.
    :param features_1_: 1D numpy array of features
    :param features_2_: 1D numpy array of features
    :param cat_weight_: Scalar between 0 and 1 for the amount to weight the Jaccard distance on binary features
    :param bool_features_: 1D numpy boolean array indexing which features are bool type (fingerprints)
    :return: 1D numpy array returning distances from rows of features_1 to rows of features_2
    """
    if not np.any(bool_features_):
        "If no binary features are flagged, or if none explicitly flagged:"
        return pairwise_distances(features_1_.reshape(1, -1), features_2_.reshape(1, -1), metric="euclidean")
    elif np.all(bool_features_):
        "If all features explicitly flagged as binary:"
        return pairwise_distances(features_1_.reshape(1, -1), features_2_.reshape(1, -1), metric="jaccard")
    else:
        "If both binary / continuous features are flagged as present:"
        if (features_1_.ndim != 1) or (features_2_.ndim != 1):
            raise ValueError("Input to mixed_metric should be 1D array.")
        categorical_features_1 = features_1_[bool_features_].astype(bool).reshape(1, -1)
        continuous_features_1 = features_1_[~bool_features_].reshape(1, -1)
        categorical_features_2 = features_2_[bool_features_].astype(bool).reshape(1, -1)
        continuous_features_2 = features_2_[~bool_features_].reshape(1, -1)
        cat_distances = cat_weight_ * pairwise_distances(categorical_features_1, categorical_features_2,
                                                        metric="jaccard")
        cont_distances = (1 - cat_weight_) * pairwise_distances(continuous_features_1, continuous_features_2,
                                                               metric="euclidean")
        hybrid_dist = np.sqrt(np.square(cat_distances) + np.square(cont_distances))[0, 0]
        return hybrid_dist


def min_distance(features_1_list, features_2_array, cat_weight_, bool_features=None):
    features_1 = np.array(features_1_list)
    min_dist = np.inf
    for features_2 in features_2_array.tolist():
        dist = mixed_metric(features_1, np.array(features_2), cat_weight_, bool_features)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def min_distances(features_1, features_2, cat_weight_=None, bool_features=None):
    if bool_features is not None and cat_weight_ is None:
        cat_weight_ = np.mean(bool_features)
    elif bool_features is None and cat_weight_ is None:
        cat_weight_ = 0
    else:
        pass
    distances_ = [min_distance(row, features_2, cat_weight_, bool_features) for row in features_1.tolist()]
    return np.array(distances_).flatten()


def min_distances_parallel(features_1, features_2, cat_weight_=None, bool_features=None):
    if bool_features is not None and cat_weight_ is None:
        cat_weight_ = np.mean(bool_features)
    elif bool_features is None and cat_weight_ is None:
        cat_weight_ = 0
    else:
        pass

    def get_min_distance(features_list):
        return min_distance(features_list, features_2, bool_features)
    distances = np.array(Pool().map(get_min_distance, features_1.tolist()))
    return distances.flatten()


def min_dist_func_getter(parallel_, cat_weight_=None, is_bool_feature_=None):
    """Get correct distance function according to 'parallel' keyword, and pre-feed it with is_bool_feature_ argument."""
    if parallel_:
        return lambda X, Y: min_distances_parallel(X, Y, cat_weight_, is_bool_feature_)
    else:
        return lambda X, Y: min_distances(X, Y, cat_weight_, is_bool_feature_)


def get_weights_and_bias(features_, labels_, is_validation_, cat_weight=None, is_bool_feature=None, parallel=False):
    """
    Computes validation weights for later use in weighted model performance metrics. Also computes uky_score.
    :param features_: 2D numpy array of features
    :param cat_weight: Scalar between 0 and 1 for how much to weight the binary features in distance computation
    :param is_bool_feature: 1D boolean numpy array indicating which features are binary (fingerprints)
    :param labels_: 1D boolean numpy array of activity labels
    :param is_validation_: 1D boolean numpy array indicating which rows are in the validation set
    :param parallel: Optional keyword argument (Default: False) for parallel computation of distances
    :return: 1D numpy array of weights (length equal to number of validation rows), uky_score (float)
    """
    training_indices, *_ = np.where(~is_validation_)
    validation_indices, *_ = np.where(is_validation_)
    validation_size = validation_indices.size
    active_indices, *_ = np.where(labels_)
    decoy_indices, *_ = np.where(~labels_)
    training_active_indices = np.sort(np.array(list(set(training_indices).intersection(set(active_indices)))))
    training_decoy_indices = np.sort(np.array(list(set(training_indices).intersection(set(decoy_indices)))))
    validation_decoy_indices = np.sort(np.array(list(set(validation_indices).intersection(set(decoy_indices)))))
    validation_active_indices = np.sort(np.array(list(set(validation_indices).intersection(set(active_indices)))))
    if is_bool_feature is None:
        "If no binary features are flagged, assume all features are continuous."
        *_, dim = features_.shape
        is_bool_feature = np.zeros(dim).astype(bool)
    # compute min distances
    min_dist_func = min_dist_func_getter(parallel, is_bool_feature)
    active_active_dsts = min_dist_func(features_[validation_active_indices], features_[training_active_indices])
    active_decoy_dsts = min_dist_func(features_[validation_active_indices], features_[training_decoy_indices])
    decoy_active_dsts = min_dist_func(features_[validation_decoy_indices], features_[training_active_indices])
    decoy_decoy_dsts = min_dist_func(features_[validation_decoy_indices], features_[training_decoy_indices])
    # compute differences / ratios
    active_ratios = active_active_dsts / active_decoy_dsts
    active_spatial_biases = np.sum(active_decoy_dsts - active_active_dsts)
    decoy_ratios = decoy_decoy_dsts / decoy_active_dsts
    decoy_spatial_biases = np.sum(decoy_active_dsts - decoy_decoy_dsts)
    # compute spatial statistic uky_score
    uky_score = np.sqrt(active_spatial_biases**2 + decoy_spatial_biases**2)
    # compute validation_weights
    cols = np.hstack([validation_active_indices, validation_decoy_indices])
    data = np.hstack([active_ratios, decoy_ratios])
    data = (1 + np.argsort(data)) / validation_size
    validation_weights = csr_matrix((data, (np.zeros(validation_size).astype(int), np.argsort(cols)))
                                    ).toarray().flatten()
    return validation_weights, uky_score


