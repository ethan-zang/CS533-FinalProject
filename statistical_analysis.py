import numpy as np

from scipy import stats


def main():
    """
    Driver script for calculating Pearson's linear correlation coefficients between all of the feature vectors and the classifications achieved
    """
    classifications = np.load('y.npy')
    feature_vector = np.load('signal_features.npy')
    feature_classification_pearson_coefficients = []
    print(feature_vector.shape)

    for i in range(feature_vector.shape[1]):
        feature_classification_pearson_coefficients.append(stats.pearsonr(feature_vector[:, i], classifications)[0])

    print('Pearson coefficients: ', feature_classification_pearson_coefficients)


if __name__ == '__main__':
    main()
