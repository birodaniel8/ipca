import numpy as np


class PCA():
    """
    This class estimates the principal components of a given input matrix.
    """

    def __init__(self):
        pass

    def fit(self, y, standardize=True):
        """
        This method estimates the principal components via SVD

        Parameters
        ----------
        y : numpy array
            Input data with t x n matrix format where t is the number of observations and n is the number of variables
        standardize : bool, optional
            If true, the input data y is standardized, by default True
        """
        # Input check and setting the estimation inputs:
        self.input_check(y=y, standardize=standardize)
        self.t_, self.n_ = y.shape

        # Standardization:
        if standardize == True:
            self.y_ = (y - np.mean(y, axis=0))/np.std(y, axis=0)

        # Singular value decomposition (SVD):
        u, s, vt = np.linalg.svd(self.y_)

        # Saving esimated values:
        self.u_ = u
        self.s_ = s
        self.vt_ = vt
        self.score_ = u[:, :self.n_] @ np.diag(s)
        self.loading_ = vt
        self.lambda_ = s ** 2 / self.n_
        self.explained_ = self.lambda_ / np.sum(self.lambda_)
        self.fitted_ = True

    def get_components(self, k=None):
        """
        This method returns the first k principal components and their loadings.

        Parameters
        ----------
        k : int, optional
            Number of principal components to be returned, by default None

        Returns
        -------
        numpy array
            First k principal components
        numpy array
            Loadings on the first k principal components
        """
        if self.fitted_:
            if k is None:
                k = self.n_
            else:
                assert isinstance(k, int), "K must be an integer!"
                assert k <= self.n_, "K must be less or equal than the number of columns in Y (N)!"
            return self.score_[:, :k], self.loading_[:k, :]

    def get_eigen_vectors(self, k=None):
        """
        This method returns the eigenvectors corresponding to the first k largest eigenvalues (with descending order).

        Parameters
        ----------
        k : int, optional
            Number of eigenvectors to return, by default None

        Returns
        -------
        numpy array
            Array of the k eigenvectors.
        """
        if self.fitted_:
            if k is None:
                k = self.n_
            else:
                assert isinstance(k, int), "K must be an integer!"
                assert k <= self.n_, "K must be less or equal than the number of columns in Y (N)!"
            return self.vt_.T[:, :k]

    def input_check(self, y, standardize):
        """
        This method checks the input for the PCA fit.
        """
        assert isinstance(y, np.ndarray), "Y must be a numpy array!"
        assert len(y.shape) == 2, "Y must be a 2D array!"
        self.y = y
        self.y_ = y
        assert isinstance(standardize, bool), "You must specify the standardize parameter as True or False!"
