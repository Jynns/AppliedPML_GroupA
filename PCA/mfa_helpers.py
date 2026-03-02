# Mixture of Factor analyzers for CelebA images
# from
# https://github.com/eitanrich/gans-n-gmms
# and 
# https://github.com/eitanrich/torch-mfa
#

import math
import time
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

"""## MFA"""


class MFA(torch.nn.Module):
    """
    A class representing a Mixture of Factor Analyzers [1] / Mixture of Probabilistic PCA [2] in pytorch.
    MFA/MPPCA are Gaussian Mixture Models with low-rank-plus-diagonal covariance, enabling efficient modeling
    of high-dimensional domains in which the data resides near lower-dimensional subspaces.
    The implementation is based on [3] (please quote this if you are using this package in your research).
    Attributes (old/new names) (model parameters):
    ------------------------------
    MU/means: Tensor shaped [n_components, n_features]
        The component means 
    A/common_subspace: Tensor shaped [n_components, n_features, n_factors]
        The component subspace directions / factor loadings. These should be orthogonal (but not orthonormal)
    log_D/log_component_variances: Tensor shaped [n_components, n_features]
        Log of the component diagonal variance values. Note that in MPPCA, all values along the diagonal are the same.
    PI_logits/PI_logits: Tensor shaped [n_components]
        Log of the component mixing-coefficients (probabilities). Apply softmax to get the actual PI values.
    Main Methods:
    -------------
    fit:
        Fit the MPPCA model parameters to pre-loaded training data using EM
    batch_fit:
        Fit the MPPCA model parameters to a (possibly large) pytorch dataset using EM in batches
    sample:
        Generate new samples from the trained model
    per_component_log_likelihood, log_prob, log_likelihood:
        Probability query methods
    responsibilities, log_responsibilities, map_component:
        Responsibility (which component the sample comes from) query methods
    reconstruct, conditional_reconstruct:
        Reconstruction and in-painting
    [1] Tipping, Michael E., and Christopher M. Bishop. "Mixtures of probabilistic principal component analyzers."
        Neural computation 11.2 (1999): 443-482.
    [2] Ghahramani, Zoubin, and Geoffrey E. Hinton. "The EM algorithm for mixtures of factor analyzers."
        Vol. 60. Technical Report CRG-TR-96-1, University of Toronto, 1996.
    [3] Richardson, Eitan, and Yair Weiss. "On gans and gmms."
        Advances in Neural Information Processing Systems. 2018.
    """

    def __init__(
        self,
        n_components: int,
        n_features: int,
        n_factors: int,
        init_method='rnd_samples'
    ):
        super(MFA, self).__init__()
        self.n_components = n_components  # also named K - number of mixture components
        self.n_features = n_features  # also named d - data dimensionality, e.g. number of pixels
        self.n_factors = n_factors  # also named l - number of factors / latent space dimensionality
        self.init_method = init_method

        # W is a n_features X n_factors matrix where n_features is the data dimensionality (num pixels)
        #  - > each feature is associated with n_factors factor loadings

        self.means = torch.nn.Parameter(torch.zeros(n_components, n_features), requires_grad=False) # MU
        self.common_subspace = torch.nn.Parameter(torch.zeros(n_components, n_features, n_factors), requires_grad=False) #A
        self.log_component_variances = torch.nn.Parameter(torch.zeros(n_components, n_features), requires_grad=False) #log_D
        self.PI_logits = torch.nn.Parameter(
            torch.log(torch.ones(n_components) / float(n_components)),
            requires_grad=False,
        ) # PI_logits

    def sample(self, n):
        """
        Generate random samples from the trained MFA / MPPCA
        :param n: How many samples
        :return: samples [n, n_features], c_nums - originating component numbers
        """
        if torch.all(self.common_subspace == 0.0):
            warnings.warn("SGD MFA training requires initialization. Please call batch_fit() first.")

        # selects n components randomly based on the mixing coefficients (PI_logits). These coefficients are converted
        # to probabilities using the softmax function
        c_nums = np.random.choice(
            self.n_components,
            n,
            p=torch.softmax(self.PI_logits, dim=0).detach().cpu().numpy(),
        )

        # low-dimensional latent variables
        z_l = torch.randn(n, self.n_factors, device=self.common_subspace.device)

        samples = torch.stack(
            [
                self.common_subspace[c_nums[i]] @ z_l[i]
                + self.means[c_nums[i]]
                for i in range(n)
            ]
        )
        return samples, c_nums

    @staticmethod
    def _component_log_likelihood(x, PI, MU, A, log_D):
        n_components, d, l = A.shape
        A_T = A.transpose(1, 2)
        inverse_diagonale = torch.exp(-log_D).view(n_components, d, 1)
        L = torch.eye(l, device=A.device).reshape(1, l, l) + A_T @ (inverse_diagonale * A)
        iL = torch.inverse(L)

        def per_component_mahalanobis_distance(i):
            x_c = (x - MU[i].reshape(1, d)).T  # shape = (d, n)
            m_d_1 = (inverse_diagonale[i] * x_c) - ((inverse_diagonale[i] * A[i]) @ iL[i]) @ (A_T[i] @ (inverse_diagonale[i] * x_c))
            return torch.sum(x_c * m_d_1, dim=0)

        mahalanobis_distances = torch.stack([per_component_mahalanobis_distance(i) for i in range(n_components)])
        det_L = torch.logdet(L)
        log_det_Sigma = det_L - torch.sum(torch.log(inverse_diagonale.reshape(n_components, d)), axis=1)
        log_prob_data_given_components = -0.5 * (
            (d * np.log(2.0 * math.pi) + log_det_Sigma).reshape(n_components, 1) + mahalanobis_distances
        )
        return PI.reshape(1, n_components) + log_prob_data_given_components.T

    def per_component_log_likelihood(self, x):
        """
        Calculate per-sample and per-component log-likelihood values
        :param x: samples [n, n_features]
        :param sampled_features: list of feature coordinates to use
        :return: log-probability values [n, n_components]
        """
        return MFA._component_log_likelihood(
            x, torch.softmax(self.PI_logits, dim=0), 
                             self.means, 
                             self.common_subspace, 
                             self.log_component_variances
        )

    def log_prob(self, x):
        """
        Calculate per-sample log-probability values
        :param x: samples [n, n_features]
        :param sampled_features: list of feature coordinates to use
        :return: log-probability values [n]
        """
        return torch.logsumexp(self.per_component_log_likelihood(x), dim=1)

    def log_responsibilities(self, x):
        """
        Calculate the log-responsibilities (log of the responsibility values - probability of each sample to originate
        from each of the component.
        :param x: samples [n, n_features]
        :param sampled_features: list of feature coordinates to use
        :return: log-responsibilities values [n, n_components]
        """
        comp_LLs = self.per_component_log_likelihood(x)
        return comp_LLs - torch.logsumexp(comp_LLs, dim=1).reshape(-1, 1)

    def responsibilities(self, x, sampled_features=None):
        """
        Calculate the responsibilities - probability of each sample to originate from each of the component.
        :param x: samples [n, n_features]
        :param sampled_features: list of feature coordinates to use
        :return: responsibility values [n, n_components]
        """
        return torch.exp(self.log_responsibilities(x))

    def map_component(self, x, sampled_features=None):
        """
        Get the Maximum a Posteriori component numbers
        :param x: samples [n, n_features]
        :param sampled_features: list of feature coordinates to use
        :return: component numbers [n]
        """
        return torch.argmax(self.log_responsibilities(x), dim=1)

    def conditional_reconstruct(self, full_x, observed_features):
        """
        Calculates the mean of the conditional probability P(x_hidden | x_observed) to reconstruct
        missing/hidden features from observed features using the Gaussian mixture model.

        The method uses the Woodbury matrix identity and conditional Gaussian formulas to efficiently
        compute the reconstruction without inverting the full covariance matrix.

        References:
        https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf#subsubsection.8.1.3
        https://en.wikipedia.org/wiki/Woodbury_matrix_identity

        Note: This is equivalent to calling reconstruct with sampled_features

        :param full_x: the full vectors (including the hidden coordinates, which can contain any values)
        :param observed_features: tensor containing a list of the observed coordinates of x
        :return: A cloned version of full_x with the hidden features reconstructed
        """
        assert observed_features is not None

        # ============================
        # Determine the most likely component for each sample
        # ============================
        component_indices = self.map_component(full_x, observed_features)

        # ============================
        # Create feature mask: True for observed features, False for hidden features
        # ============================
        observed_mask = torch.zeros(self.n_features, dtype=bool)
        observed_mask[observed_features] = True

        # ============================
        # Extract parameters for hidden and observed dimensions
        # ============================
        # Factor loadings for hidden features (dimensions we want to reconstruct)
        factor_loadings_hidden = self.common_subspace[component_indices][:, ~observed_mask, :]

        # Factor loadings for observed features (dimensions we know)
        factor_loadings_observed = self.common_subspace[component_indices][:, observed_mask, :]

        # Component means for hidden features
        mean_hidden = self.means[component_indices][:, ~observed_mask]

        # Component means for observed features
        mean_observed = self.means[component_indices][:, observed_mask]

        # Inverse diagonal variance matrix for observed features: D_observed^{-1} (shape: [n, n_observed, 1])
        inverse_variance_observed = torch.exp(-self.log_component_variances[component_indices][:, observed_mask]).unsqueeze(2)

        # ============================
        # Compute L matrix inverse using Woodbury identity
        # ============================
        # L_observed = I + A_observed^T * D_observed^{-1} * A_observed (shape: [n, n_factors, n_factors])
        identity_matrix = torch.eye(self.n_factors, device=mean_observed.device).reshape(1, self.n_factors, self.n_factors)
        L_observed = identity_matrix + factor_loadings_observed.transpose(1, 2) @ (inverse_variance_observed * factor_loadings_observed)

        # Compute L_observed^{-1} for the Woodbury matrix identity
        inverse_L_observed = torch.inverse(L_observed)

        # ============================
        # Project observed data deviations into latent space
        # ============================
        # Compute the latent representation: z = A_observed^T * D_observed^{-1} * (x_observed - μ_observed)
        # This projects the centered observed data through the weighted factor loadings
        observed_deviation = (full_x[:, observed_mask] - mean_observed).unsqueeze(2)  # [n, n_observed, 1]
        latent_projection = (factor_loadings_observed * inverse_variance_observed).transpose(1, 2) @ observed_deviation

        # ============================
        # Reconstruct hidden features using conditional Gaussian formula
        # ============================
        # E[x_hidden | x_observed] = μ_hidden + A_hidden * E[z | x_observed]
        # where E[z | x_observed] involves the Woodbury correction term

        reconstructed_samples = full_x.clone()

        # Woodbury correction term: A_observed^T * D_observed^{-1} * (A_observed * L_observed^{-1} * latent_projection)
        woodbury_correction = factor_loadings_observed.transpose(1, 2) @ (
            inverse_variance_observed * (factor_loadings_observed @ inverse_L_observed @ latent_projection)
        )

        # Final reconstruction: μ_hidden + A_hidden * (latent_projection - woodbury_correction)
        reconstructed_samples[:, ~observed_mask] = (
            mean_hidden.unsqueeze(2)
            + factor_loadings_hidden @ latent_projection
            - factor_loadings_hidden @ woodbury_correction
        ).squeeze(dim=2)

        return reconstructed_samples

    @staticmethod
    def _small_sample_ppca(x, n_factors):
        # See https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
        mu = torch.mean(x, dim=0)
        # U, S, V = torch.svd(x - mu.reshape(1, -1))    # torch svd is less memory-efficient
        U, S, V = np.linalg.svd((x - mu.reshape(1, -1)).cpu().numpy(), full_matrices=False)
        V = torch.from_numpy(V.T).to(x.device)
        S = torch.from_numpy(S).to(x.device)
        sigma_squared = torch.sum(torch.pow(S[n_factors:], 2.0))/((x.shape[0]-1) * (x.shape[1]-n_factors))
        A = V[:, :n_factors] * torch.sqrt((torch.pow(S[:n_factors], 2.0).reshape(1, n_factors)/(x.shape[0]-1) - sigma_squared))
        return mu, A, torch.log(sigma_squared) * torch.ones(x.shape[1], device=x.device)


    def _init_from_data(self, x, samples_per_component, feature_sampling=False):
        n = x.shape[0]
        K, d, l = self.common_subspace.shape

        if self.init_method == 'kmeans':
            # Import this only if 'kmeans' method was selected (not sure this is a good practice...)
            from sklearn.cluster import KMeans
            sampled_features = np.random.choice(d, int(d*feature_sampling)) if feature_sampling else np.arange(d)

            t = time.time()
            print('Performing K-means clustering of {} samples in dimension {} to {} clusters...'.format(
                x.shape[0], sampled_features.size, K))
            _x = x[:, sampled_features].cpu().numpy()
            clusters = KMeans(n_clusters=K, max_iter=300, n_jobs=-1).fit(_x)
            print('... took {} sec'.format(time.time() - t))
            component_samples = [clusters.labels_ == i for i in range(K)]
        elif self.init_method == 'rnd_samples':
            m = samples_per_component
            o = np.random.choice(n, m*K, replace=False) if m*K < n else np.arange(n)
            assert n >= m*K
            component_samples = [[o[i*m:(i+1)*m]] for i in range(K)]

        params = [torch.stack(t) for t in zip(
            *[MFA._small_sample_ppca(x[component_samples[i]], n_factors=l) for i in range(K)])]

        self.means.data = params[0]
        self.common_subspace.data = params[1]
        self.log_component_variances.data = params[2]

    def _parameters_sanity_check(self):
        # Verifies that no mixture component has an extremely small weight.
        assert torch.all(torch.softmax(self.PI_logits, dim=0) > 0.01 / self.n_components), self.PI_logits
        # Verifies that no mixture component has an extremely small or large variance.
        assert torch.all(torch.exp(self.log_component_variances) > 1e-5) and torch.all(
            torch.exp(self.log_component_variances) < 1.0
        ), f"{torch.min(self.log_component_variances).item()} - {torch.max(self.log_component_variances).item()}"
        # Checks that the subspace values are not exploding
        assert torch.all(torch.abs(self.common_subspace) < 10.0), torch.max(torch.abs(self.common_subspace))
        # Checks that the means values are not exploding
        assert torch.all(torch.abs(self.means) < 1.0), torch.max(torch.abs(self.means))

    def batch_fit(
        self,
        train_dataset,
        test_dataset=None,
        batch_size=1000,
        test_size=1000,
        max_iterations=20,
    ):
        """
        Estimate Maximum Likelihood MPPCA parameters for the provided data using EM per
        Tipping, and Bishop. Mixtures of probabilistic principal component analyzers.
        Memory-efficient batched implementation for large datasets that do not fit in memory:
        E step:
            For all mini-batches:
            - Calculate and store responsibilities
            - Accumulate sufficient statistics
        M step:
            Re-calculate all parameters
        :param train_dataset: pytorch Dataset object containing the training data (will be iterated over)
        :param test_dataset: optional pytorch Dataset object containing the test data (otherwise train_daset will be used)
        :param batch_size: the batch size
        :param test_size: number of samples to use when reporting likelihood
        :param max_iterations: number of iterations (=epochs)
        :param feature_sampling: allows faster responsibility calculation by sampling data coordinates
        """
        # ============================
        # Initialize model parameters
        # ============================
        # Use 2*(n_factors+1) samples per component for initialization (ensures sufficient data)
        init_samples_per_component = (self.n_factors + 1) * 2
        print(f"Random init with {init_samples_per_component} samples per component...")

        # Randomly sample initialization data from training set
        init_keys = [
            key
            for i, key in enumerate(RandomSampler(train_dataset))
            if i < init_samples_per_component * self.n_components
        ]
        init_samples, _ = zip(*[train_dataset[key] for key in init_keys])
        self._init_from_data(
            torch.stack(init_samples).to(self.means.device),
            samples_per_component=init_samples_per_component,
        )

        # ============================
        # Prepare test data for evaluation
        # ============================
        test_dataset = test_dataset or train_dataset
        all_test_keys = [key for key in SequentialSampler(test_dataset)]
        test_samples, _ = zip(*[test_dataset[key] for key in all_test_keys[:test_size]])
        test_samples = torch.stack(test_samples).to(self.means.device)

        # ============================
        # Main EM iteration loop
        # ============================
        likelihood_log = []
        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

        for iteration in range(max_iterations):
            iteration_start_time = time.time()

            # ============================
            # Initialize sufficient statistics for E-step
            # ============================
            # Sum of responsibilities for each component: Σ_n r_nk (shape: [K])
            sum_responsibilities = torch.zeros(
                size=[self.n_components],
                dtype=torch.float64,
                device=self.means.device
            )

            # Weighted sum of data points: Σ_n r_nk * x_n (shape: [K, D])
            sum_weighted_data = torch.zeros(
                size=[self.n_components, self.n_features],
                dtype=torch.float64,
                device=self.means.device,
            )

            # Weighted covariance term: Σ_n r_nk * x_n * x_n^T * A_k (shape: [K, D, L])
            sum_weighted_data_covariance = torch.zeros(
                size=[self.n_components, self.n_features, self.n_factors],
                dtype=torch.float64,
                device=self.means.device,
            )

            # Weighted sum of squared norms: Σ_n r_nk * ||x_n||^2 (shape: [K])
            sum_weighted_squared_norms = torch.zeros(
                self.n_components,
                dtype=torch.float64,
                device=self.means.device
            )

            # Evaluate and log current likelihood
            current_likelihood = torch.mean(self.log_prob(test_samples)).item()
            likelihood_log.append(current_likelihood)
            print(f"Iteration {iteration}/{max_iterations}, log-likelihood={current_likelihood}:")

            # ============================
            # E-step: Calculate responsibilities and accumulate sufficient statistics
            # ============================
            for batch_data, _ in tqdm(data_loader, desc="E-step: processing batches"):
                batch_data = batch_data.to(self.means.device)

                # Calculate responsibility matrix: p(component_k | data_point_n)
                batch_responsibilities = self.responsibilities(batch_data, sampled_features=None)

                # Accumulate sum of responsibilities across all data points
                sum_responsibilities += torch.sum(batch_responsibilities, dim=0).double()

                # Accumulate weighted sum of squared norms: r_nk * ||x_n||^2
                batch_squared_norms = torch.sum(torch.pow(batch_data, 2.0), dim=1, keepdim=True)
                sum_weighted_squared_norms += torch.sum(
                    batch_responsibilities * batch_squared_norms,
                    dim=0,
                ).double()

                # Accumulate statistics for each component
                for component_idx in range(self.n_components):
                    # Weight data points by their responsibility to this component
                    weighted_batch_data = batch_responsibilities[:, [component_idx]] * batch_data

                    # Accumulate weighted data sum: Σ r_nk * x_n
                    sum_weighted_data[component_idx] += torch.sum(weighted_batch_data, dim=0).double()

                    # Accumulate weighted covariance: Σ r_nk * x_n * (x_n^T * A_k)
                    sum_weighted_data_covariance[component_idx] += (
                        weighted_batch_data.T @ (batch_data @ self.common_subspace[component_idx])
                    ).double()

            # ============================
            # M-step: Update model parameters
            # ============================
            print("M-step: updating parameters", end="", flush=True)

            # Update mixing coefficients (component weights): π_k = Σ_n r_nk / N
            total_responsibilities = torch.sum(sum_responsibilities)
            self.PI_logits.data = torch.log(sum_responsibilities / total_responsibilities).float()

            # Update component means: μ_k = (Σ_n r_nk * x_n) / (Σ_n r_nk)
            self.means.data = (sum_weighted_data / sum_responsibilities.reshape(-1, 1)).float()

            # Compute centered covariance statistic: S*A = E[r_nk * x_n * x_n^T * A_k] - μ_k * μ_k^T * A_k
            centered_covariance_A = (
                sum_weighted_data_covariance / sum_responsibilities.reshape(-1, 1, 1)
                - (
                    self.means.reshape(self.n_components, self.n_features, 1)
                    @ (self.means.reshape(self.n_components, 1, self.n_features) @ self.common_subspace)
                ).double()
            )

            # Create diagonal noise matrix: σ² * I (shape: [K, L, L])
            variance_times_identity = torch.exp(self.log_component_variances[:, 0]).reshape(
                self.n_components, 1, 1
            ) * torch.eye(
                self.n_factors, device=self.means.device
            ).reshape(1, self.n_factors, self.n_factors)

            # Compute M matrix: M_k = A_k^T * A_k + σ²I (shape: [K, L, L])
            M_matrix = (self.common_subspace.transpose(1, 2) @ self.common_subspace + variance_times_identity).double()

            # Compute inverse of M for each component
            inverse_M = torch.stack([torch.inverse(M_matrix[i]) for i in range(self.n_components)])

            # Compute intermediate term: M^{-1} * A^T * S * A
            intermediate_term = inverse_M @ self.common_subspace.double().transpose(1, 2) @ centered_covariance_A

            # Update factor loadings/subspace: A_k = S*A * (σ²I + M^{-1}*A^T*S*A)^{-1}
            self.common_subspace.data = torch.stack([
                (centered_covariance_A[i] @ torch.inverse(
                    variance_times_identity[i].double() + intermediate_term[i]
                )).float()
                for i in range(self.n_components)
            ])

            # Compute trace term for variance update: tr(A_k^T * S*A * M^{-1})
            trace_term = torch.stack([
                torch.trace(self.common_subspace[i].double().T @ (centered_covariance_A[i] @ inverse_M[i]))
                for i in range(self.n_components)
            ])

            # Compute variance statistic: E[||x_n - μ_k||²]
            variance_statistic = (
                sum_weighted_squared_norms / sum_responsibilities
                - torch.sum(torch.pow(self.means, 2.0), dim=1).double()
            )

            # Update component variances: σ²_k = (1/D) * (variance_statistic - trace_term)
            self.log_component_variances.data = torch.log(
                (variance_statistic - trace_term) / self.n_features
            ).float().reshape(-1, 1) * torch.ones_like(self.log_component_variances)

            # Verify parameter validity
            self._parameters_sanity_check()

            iteration_time = time.time() - iteration_start_time
            print(f" ({iteration_time:.2f} sec)")

        # ============================
        # Final evaluation
        # ============================
        final_likelihood = torch.mean(self.log_prob(test_samples)).item()
        likelihood_log.append(final_likelihood)
        print(f"\nFinal train log-likelihood={final_likelihood}:")
        return likelihood_log


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class CropTransform:
    def __init__(self, bbox):
        self.bbox = bbox

    def __call__(self, img):
        return img.crop(self.bbox)


def samples_to_np_images(samples, image_shape=[64, 64, 3], clamp=True):
    assert len(samples.shape) == 2
    assert samples.shape[1] == np.prod(image_shape)
    assert len(image_shape) == 2 or (len(image_shape) == 3 and image_shape[2] > 1)
    samples_out = samples if not clamp else torch.clamp(samples, 0.0, 1.0)
    if len(image_shape) == 3:
        return samples_out.reshape(-1, image_shape[2], image_shape[0], image_shape[1]).permute(0, 2, 3, 1).cpu().numpy()
    else:
        return samples_out.reshape(-1, image_shape[0], image_shape[1]).cpu().numpy()


def sample_to_np_image(sample, image_shape=[64, 64, 3]):
    return samples_to_np_images(sample.unsqueeze(0), image_shape).squeeze()


def samples_to_mosaic(samples, image_shape=(64, 64, 3)):
    images = samples_to_np_images(samples, image_shape)
    num_images = images.shape[0]
    num_cols = int(np.ceil(np.sqrt(num_images)))
    rows = []
    for i in range(num_images // num_cols):
        rows.append(np.hstack([images[j] for j in range(i * num_cols, (i + 1) * num_cols)]))
    return (np.vstack(rows) * 255).astype(np.uint8)


def visualize_model(model: MFA, image_shape=(64, 64, 3), start_component=0, end_component=None):
    n_factors = model.n_factors
    assert len(image_shape) == 2 or (len(image_shape) == 3 and image_shape[2] > 1)

    height, width = image_shape[:2]
    spacer = min(8, width // 8)
    end_component = end_component or min(model.n_components, 2048 // (width * 3 + 2 + spacer))
    num_components = end_component - start_component
    # somewhat arbitrary scaling factor for the factor directions
    z = 1.5

    def to_im(x):
        return sample_to_np_image(x, image_shape=image_shape)

    if len(image_shape) == 3:
        canvas = np.ones(
            [
                (n_factors + 1) * (height + 1),
                num_components * (width * 3 + 2) + (num_components - 1) * spacer,
                image_shape[2],
            ]
        )
    else:
        canvas = np.ones(
            [
                (n_factors + 1) * (height + 1),
                num_components * (width * 3 + 2) + (num_components - 1) * spacer,
            ]
        )
    canvas = canvas * 0.5  # gray background  # TODO: not ideal but makes sure that the std image is somewhat visible

    for c_num in range(start_component, end_component):
        x_start = (c_num - start_component) * (width * 3 + 2 + spacer)

        # Creates the mean image of the component
        mu = model.means[c_num]
        canvas[:height, x_start + width // 2 : x_start + width // 2 + width] = to_im(mu)

        # Creates the std image of the component  # TODO: is this correct since it is always white?
        D = torch.exp(0.5 * model.log_component_variances[c_num])
        canvas[
            :height,
            x_start + width // 2 + width + 2 : x_start + width // 2 + 2 * width + 2,
        ] = to_im(D / torch.max(D))

        # For each [dimension of the latent space] shows the mean PLUS the factor, direction of the factor (which is given by
        # the common_subspace/factor loadings/A_i) and the mean MINUS the factor
        for i in range(n_factors):
            y_start = (i + 1) * (height + 1)

            A_i = model.common_subspace[c_num, :, i]
            canvas[y_start : y_start + height, x_start : x_start + width] = to_im(mu + z * A_i)

            canvas[
                y_start : y_start + height,
                x_start + width + 1 : x_start + 2 * width + 1,
            ] = to_im(0.5 + z * A_i)
            canvas[
                y_start : y_start + height,
                x_start + 2 * width + 2 : x_start + 3 * width + 2,
            ] = to_im(mu - z * A_i)
    scaled_canvas = (canvas * 255).astype(np.uint8)
    return scaled_canvas
