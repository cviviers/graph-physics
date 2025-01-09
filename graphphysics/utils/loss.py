import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from graphphysics.utils.nodetype import NodeType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class L2Loss(_Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def __name__(self):
        return "MSE"

    def forward(
        self,
        target: torch.Tensor,
        network_output: torch.Tensor,
        node_type: torch.Tensor,
        masks: list[NodeType],
        selected_indexes: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Computes L2 loss for nodes of specific types.

        Args:
            target (torch.Tensor): The target values.
            network_output (torch.Tensor): The predicted values from the network.
            node_type (torch.Tensor): Tensor containing the type of each node.
            masks (list[NodeType]): List of NodeTypes to include in the loss calculation.
            selected_indexes (torch.Tensor, optional): Indexes of nodes to exclude from the loss calculation.

        Returns:
            torch.Tensor: The mean squared error for the specified node types.

        Note:
            This method calculates the L2 loss only for nodes of the types specified in 'masks'.
            If 'selected_indexes' is provided, those nodes are excluded from the loss calculation.
        """
        mask = node_type == masks[0]
        for i in range(1, len(masks)):
            mask = torch.logical_or(mask, node_type == masks[i])

        if selected_indexes is not None:
            n, _ = network_output.shape
            nodes_mask = ~torch.isin(torch.arange(n), selected_indexes).to(device)
            mask = torch.logical_and(nodes_mask, mask)

        errors = ((network_output - target) ** 2)[mask]
        return torch.mean(errors)


class GaussianMixtureNLLLoss(_Loss):
    """
    Negative log-likelihood loss for a full-covariance GMM output:
    network_output shape: [N, K * per_component],
    where per_component = d + d(d+1)//2 + 1
    We parse each chunk to get alpha, mu, L.
    """

    def __init__(self, d: int, K: int, temperature: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.d = d
        self.K = K
        self.temperature = temperature

        # how many params per component
        self.per_comp = d + (d * (d + 1)) // 2 + 1  # mean + L + logit

    @property
    def __name__(self):
        return "GaussianMixtureNLLLoss"

    def forward(
        self,
        target: torch.Tensor,  # shape [N, d]
        network_output: torch.Tensor,  # shape [N, K * per_comp]
        node_type: torch.Tensor,
        masks: list,  # same usage as L2Loss for which nodes to compute
        selected_indexes: torch.Tensor = None,
    ) -> torch.Tensor:

        # 1) Mask out nodes we don't want
        mask = node_type == masks[0]
        for i in range(1, len(masks)):
            mask = torch.logical_or(mask, node_type == masks[i])
        if selected_indexes is not None:
            n = network_output.shape[0]
            nodes_mask = ~torch.isin(torch.arange(n).to(device), selected_indexes)
            mask = torch.logical_and(nodes_mask, mask)

        # keep only masked nodes
        target = target[mask]  # shape [M, d]
        network_output = network_output[mask]  # shape [M, K * per_comp]

        # Parse mixture parameters
        #    network_output[i] -> ( K x [mu, L, logit] )
        M = target.shape[0]
        # reshape => [M, K, per_comp]
        net_3d = network_output.view(M, self.K, self.per_comp)

        # separate out mixture logits, means, L factors
        # each row: [logit, mu1..mud, L1..L_{d(d+1)/2}]
        # We'll place the logit first for convenience
        logit = net_3d[..., 0]  # shape [M, K]

        # means
        means = []
        idx_start = 1
        idx_end = 1 + self.d
        means = net_3d[..., idx_start:idx_end]  # shape [M, K, d]

        # L factors
        L_len = (self.d * (self.d + 1)) // 2
        idx2_start = idx_end
        idx2_end = idx_end + L_len
        L_flat = net_3d[..., idx2_start:idx2_end]  # shape [M, K, L_len]

        # Convert L_flat to lower-triangular [M, K, d, d]
        # We'll fill a dxd matrix with zero above diagonal
        L_mat = torch.zeros(M, self.K, self.d, self.d, device=network_output.device)
        tril_indices = torch.tril_indices(row=self.d, col=self.d, offset=0)
        L_mat[..., tril_indices[0], tril_indices[1]] = L_flat

        # Apply temperature scaling
        #   scale L by self.temperature
        L_mat = L_mat * self.temperature

        # Compute mixture weights alpha => softmax over logit
        alpha = F.softmax(logit, dim=-1)  # shape [M, K]

        # For each node & each mixture, compute the PDF of target
        #    p(x|mu,L) = Normal( x | mu, Sigma ), Sigma = L L^T
        # We'll do it in a loop or vectorized:
        x = target.unsqueeze(1)  # shape [M, 1, d]
        mu = means  # shape [M, K, d]

        diff = x - mu  # shape [M, K, d]

        # Sigma = L L^T => we can solve L * y = diff => y
        # logdet(Sigma) = 2 * sum(log(diag(L)))
        # Then do the normal formula for log pdf:
        #  log N(x|mu, Sigma) = -0.5 * (d * log(2pi) + 2*sum(log(diag(L))) + y^T y)
        #   where y = L^-1 (diff).

        # diagonal of L
        diag_L = L_mat[..., torch.arange(self.d), torch.arange(self.d)]  # [M,K,d]

        # logdet(Sigma)
        logdet = 2.0 * torch.sum(torch.log(torch.abs(diag_L) + 1e-8), dim=-1)  # [M,K]

        # solve L y = diff (batch-lower-triangular solve)
        #   shape [M,K,d]
        # we can do an explicit loop or a fancy lower-triangular solve
        #  torch.linalg solves for each batch row in diff
        y = torch.linalg.solve_triangular(
            L_mat, diff.unsqueeze(-1), upper=False
        ).squeeze(-1)
        # shape [M,K,d]

        # mahalanobis = sum_{dim} y^2
        maha = torch.sum(y**2, dim=-1)  # shape [M,K]

        const = self.d * torch.log(
            torch.tensor(2.0 * 3.141592653589793, device=network_output.device)
        )
        logpdf = -0.5 * (const + logdet + maha)

        # Mixture weighting => log( alpha_k ) + logpdf
        log_alpha = torch.log(alpha + 1e-12)
        log_mixture = log_alpha + logpdf  # shape [M,K]

        # Log-sum-exp over K => log p(x)
        log_prob_x = torch.logsumexp(log_mixture, dim=-1)  # shape [M]

        # Negative log-likelihood
        nll = -torch.mean(log_prob_x)
        return nll
