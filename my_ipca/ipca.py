import numpy as np
from numpy.linalg import pinv, svd, cholesky
from my_ipca.pca import PCA


class IPCA():
    """
    IPCA model class.

    Estimated values
    ----------------
    z_ : numpy array [N x L x T]
        Array of asset characteristics. If orthogonalization is set, this variable contains the orthogonalized values
    y_ : numpy array [T x N]
        Array of observed asset returns. If standardizaton is set, this variable contains the standardized values. If
        the model is estimated based on managed portfolio returns then this is not calculated
    x_ : numpy array [T x L]
        Array of the transformed data (managed portfolio returns). This can also be used as an input to estimate IPCA
    k_ : int
        Number of factors to be estimated. If not given, K is equal to the number of characteristics given in Z
    t_ : int
        Number of time periods (T)
    n_ : int
        Number of assets (N)
    l_ : int
        Number of characteristics (L)
    gamma_ : numpy array [L x K(+1)]
        Array of estimated mapping parameters. For the restricted model (without constant) it is the same as gamma_beta.
        For the unrestricted model it adds gamma_alpha as the first column.
    gamma_alpha_ : numpy array [L x 1]
        Array of estimated mapping parameters on the intercepts. For the restricted model it gives a None value.
    gamma_beta_ : numpy array [L x K]
        Array of estimated mapping parameters on the characteristics. 
    factor_ : numpy array [T x K]
        Array of estimated latent factors.
    factor_all_ : numpy array [T x K(+1)]
        Array of estimated latent factors with the intercept term added. For the restricted model it is the same as the
        factor array.
    beta_ : numpy array [N x K x T]
        Array of estimated loadings on factors.
    gamma_path_ : list of numpy arrays
        List of estimated mapping parameters for each iterations.
    """

    def __init__(self):
        pass

    def fit(self,
            z,
            y=None,
            x=None,
            k=None,
            add_constant=False,
            standardize=True,
            fit_method="als",
            svd_orthogonalize=True,
            max_iteration=1000,
            tolerance=1e-16):
        """
        This method estimates the IPCA for a given z asset characteristics with k factors. The latent factors can be
        estimated from observed asset or managed portfolio returns.

        Parameters
        ----------
        z : numpy array [N x L x T]
            Array of asset characteristics
        y : numpy array [T x N]
            Array of observed asset returns
        x : numpy array [T x L]
            Array of observed managed portfolio returns
        k : int, optional
            Number of factors to be estimated, by default None
        add_constant : bool, optional
            If true, a constant is added to the model, by default False
        standardize : bool, optional
            If true, the input array y is standardized, by default True
        fit_method : str, optional
            Method for fitting IPCA. Possible values: "als" - Alternating Least Squares, "svd" - SVD decomposition based estimation (only for orthogonalized characteristics), by default "als"
        svd_orthogonalize : bool, optional
            If true, the characteristics are orthogonalized for the SVD based estimation, by default True
        max_iteration : int, optional
            Maximum number of iterations, by default 1000
        tolerance : float, optional
            Applied tolerance on the gamma matrix, by default 1e-16
        """
        # Input check and setting the estimation inputs:
        self.input_check(z=z, y=y, x=x, k=k, add_constant=add_constant, standardize=standardize,
                         svd_orthogonalize=svd_orthogonalize, fit_method=fit_method, max_iteration=max_iteration,
                         tolerance=tolerance)
        self.n_, self.l_, self.t_ = z.shape
        self.k_ = z.shape[1] if k is None else k

        # Standardization:
        if (standardize) & (self.estimation_input == "y"):
            self.y_ = (y - np.mean(y, axis=0))/np.std(y, axis=0)

        if self.fit_method == "svd":
            # Estimate IPCA with SVD:
            self.fit_svd(svd_orthogonalize, standardize)
        else:
            # Initialization:
            gamma_init = IPCA()
            if self.estimation_input == "y":
                gamma_init.fit(self.z_, y=self.y_, k=self.k_, fit_method="svd", svd_orthogonalize=False,
                               standardize=False)
            else:
                gamma_init.fit(self.z_, x=self.x_, k=self.k_, fit_method="svd", svd_orthogonalize=False,
                               standardize=False)
            self.gamma_ = gamma_init.gamma_
            self.gamma_beta_ = self.gamma_
            self.gamma_alpha_ = None
            if self.add_constant:
                self.gamma_ = np.hstack([np.zeros(shape=(self.l_, 1)), self.gamma_beta_])
                self.gamma_alpha_ = self.gamma_[:, 0]

            # ALS fit:
            self.fit_als()

    def fit_svd(self, orthogonalize=True, standardize=True):
        """
        Fit IPCA with SVD based estimation.

        Parameters
        ----------
        orthogonalize : bool, optional
            If true, the characteristics are orthogonalized. SVD works only with orthogonalized data, by default True
        """
        # Ortogonalization of the instrumental variables:
        if orthogonalize:
            self.z_ = np.stack([np.linalg.qr(self.z[:, :, t])[0] * np.sqrt(self.n_) for t in range(self.t_)], axis=2)
            self.orthogonalized = True

        # Calculate the transformed data x (portfolio returns, if not given):
        # Z.T @ y / N
        if self.estimation_input == "y":
            self.x_ = np.vstack([self.z_[:, :, t].T @ self.y_[t] / self.n_ for t in range(self.t_)])

        # Run PCA on the transformed data:
        pca = PCA()
        pca.fit(self.x_, standardize=standardize)

        # Gamma is the first k eigen vector or x:
        self.gamma_ = pca.get_eigen_vectors(self.k_)

        # The estimated IPCA factors (components) and factor loadings:
        self.factor_ = np.vstack([self.gamma_.T @ self.x_[t, :] for t in range(self.t_)])
        self.beta_ = np.stack([self.z_[:, :, t] @ self.gamma_ for t in range(self.t_)], axis=2)

    def fit_als(self):
        """
        Fit IPCA with ALS.
        """
        # Calculate the transformed data x (portfolio returns, if not given):
        # Z.T @ y / N
        if self.estimation_input == "y":
            self.x_ = np.vstack([self.z_[:, :, t].T @ self.y_[t] / self.n_ for t in range(self.t_)])

        gamma_diff = np.Inf
        i = 0
        self.gamma_path_ = []

        while i < self.max_iteration and gamma_diff > self.tolerance:
            # 1. Update factors:
            if not self.add_constant:
                # inv(gamma.T @ Z.T @ Z @ gamma) @ gamma.T @ (N * X)
                self.factor_ = np.vstack([
                    pinv(self.gamma_.T @ self.z_[:, :, t].T @ self.z_[:, :, t] @ self.gamma_) @
                    (self.gamma_.T @ (self.n_ * self.x_[t, :])) for t in range(self.t_)
                ])
                self.factor_all_ = self.factor_
            else:
                # inv(gamma_beta.T @ Z.T @ Z @ gamma_beta) @ gamma_beta.T @ (N * X - Z.T @ Z @ gamma_alpha)
                self.factor_ = np.vstack([
                    pinv(self.gamma_beta_.T @ self.z_[:, :, t].T @ self.z_[:, :, t] @ self.gamma_beta_) @
                    (self.gamma_beta_.T @ (self.n_ * self.x_[t, :] - self.z_[:, :, t].T @
                                           self.z_[:, :, t] @ self.gamma_alpha_)) for t in range(self.t_)
                ])
                self.factor_all_ = np.hstack([np.zeros(shape=(self.t_, 1)), self.factor_])

            # 2. Update gamma:
            # inv(sum_T(Z.T @ Z (x) F @ F.T)) @ (sum_T(N * X (x) F))
            g_1 = sum([np.kron(self.z_[:, :, t].T @ self.z_[:, :, t],
                               (self.factor_all_[t, :].reshape(-1, 1) @
                                self.factor_all_[t, :].reshape(-1, 1).T)) for t in range(self.t_)])
            g_2 = sum([np.kron((self.n_ * self.x_[t, :].reshape(-1, 1)),
                               (self.factor_all_[t, :].reshape(-1, 1))) for t in range(self.t_)])
            self.gamma_ = (pinv(g_1) @ g_2).reshape(self.gamma_.shape)

            # 3. Decompose gamma:
            if self.add_constant:
                self.gamma_alpha_ = self.gamma_[:, 0]
                self.gamma_beta_ = self.gamma_[:, 1:]
            else:
                self.gamma_beta_ = self.gamma_

            # 4. Orthogonalize gamma:
            # gamma_beta = Q
            # factor = factor @ R.T, where Q and R from gamma_beta = Q@R
            self.gamma_beta_, r = np.linalg.qr(self.gamma_beta_)
            self.factor_ = self.factor_ @ r.T

            if self.add_constant:
                # Orthogonalization between gamma_alpha and gamma_beta:
                # gamma_alpha = gamma_alpha - gamma_beta @ temp
                # factor = temp + factor, where temp = inv(gamma_beta.T @ gamma_beta) @ gamma_beta.T @ gamma_alpha
                temp = pinv(self.gamma_beta_.T @ self.gamma_beta_) @ self.gamma_beta_.T @ self.gamma_alpha_
                self.gamma_alpha_ = self.gamma_alpha_ - self.gamma_beta_ @ temp
                self.factor_ = temp + self.factor_

            # 5. Identifying f:
            # gamma_beta = gamma_beta @ U
            # factor = factor @ inv(U).T, where U is from USV = f.T@f
            u = np.linalg.svd(self.factor_.T @ self.factor_)[0]
            self.gamma_beta_ = self.gamma_beta_ @ u
            self.factor_ = self.factor_ @ pinv(u).T

            sign = np.sign(np.sum(self.factor_, axis=0))
            self.factor_ = np.multiply(self.factor_, sign)
            self.gamma_beta_ = np.multiply(self.gamma_beta_, sign)

            # 6. Concatenating the estimated values:
            if self.add_constant:
                self.gamma_ = np.hstack([self.gamma_alpha_.reshape(-1, 1), self.gamma_beta_])
                self.factor_all_ = np.hstack([np.zeros(shape=(self.t_, 1)), self.factor_])

            # 7. Calculate implied factor loading:
            self.beta_ = np.stack([self.z_[:, :, t] @ self.gamma_ for t in range(self.t_)], axis=2)

            # Save ith estimated gamma:
            self.gamma_path_.append(self.gamma_)

            if i >= 2:
                gamma_diff = np.sum((self.gamma_ - self.gamma_path_[-2])**2)
            i += 1

    def input_check(self,
                    z,
                    y,
                    x,
                    k,
                    add_constant,
                    standardize,
                    fit_method,
                    svd_orthogonalize,
                    max_iteration,
                    tolerance):
        """
        This function checks the inputs for the IPCA fit.
        """

        assert isinstance(z, np.ndarray), "Z must be a numpy array!"
        assert len(z.shape) == 3, "Z must be a 3D array!"
        n, l, t = z.shape
        self.z = z
        self.z_ = z
        assert (y is not None) | (x is not None), "You must specify either Y or X!"
        assert (y is not None) ^ (x is not None), "You must specify either Y or X, but not both!"
        if y is not None:
            assert isinstance(y, np.ndarray), "Y must be a numpy array!"
            assert len(y.shape) == 2, "Y must be a 2D array!"
            assert (t == y.shape[0]) & (n == y.shape[1]), "Y must be a [T x N] array!"
            self.estimation_input = "y"  # set Y as the estimation input
            self.y = y
            self.y_ = y
        if x is not None:
            assert isinstance(x, np.ndarray), "X must be a numpy array!"
            assert len(x.shape) == 2, "X must be a 2D array!"
            assert (t == x.shape[0]) & (l == x.shape[1]), "X must be a [T x L] array!"
            self.estimation_input = "x"  # set X as the estimation input
            self.x = x
            self.x_ = x
            self.standardize = False
        if k is not None:
            assert isinstance(k, int), "K must be an integer!"
            assert k <= l, "K must be less or equal than the number of characteristics (L)!"
        assert isinstance(add_constant, bool), "You must specify the add_constant parameter as True or False!"
        self.add_constant = add_constant
        assert isinstance(standardize, bool), "You must specify the standardize parameter as True or False!"
        assert np.isin(fit_method, ["svd", "als"]), "You must specicy the estimation method as 'svd' or 'als'!"
        self.fit_method = fit_method
        assert isinstance(svd_orthogonalize, bool), "You must specify the svd_orthogonalize parameter as True or False!"
        assert isinstance(max_iteration, int), "The maximum number of iteration must be a positive integer number!"
        assert max_iteration > 0, "The maximum number of iteration must be a positive integer number!"
        self.max_iteration = max_iteration
        assert isinstance(tolerance, (int, float)), "The tolerance must be a positive number!"
        assert tolerance > 0, "The tolerance must be a positive number!"
        self.tolerance = tolerance
