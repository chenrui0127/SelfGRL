import math
import torch
import torch.nn as nn
import math

class BayesianThermodynamicsLoss(nn.Module):
    def __init__(self, embedding_dim: int, init_kappa: float = 0.1, init_kappa_align: float = None):
        """
        Bayesian thermodynamics adaptive loss.

        embedding_dim: embedding dimension d, e.g., hidden_dim=1024
        init_kappa: KDE bandwidth for global uniformity
        init_kappa_align: alignment concentration, default = d
        """
        super().__init__()

        self.d = float(embedding_dim)

        if init_kappa_align is None:
            init_kappa_align = self.d

        self.s_align = nn.Parameter(
            torch.tensor(math.log(init_kappa_align), dtype=torch.float32)
        )
        self.s_unif = nn.Parameter(
            torch.tensor(math.log(init_kappa), dtype=torch.float32)
        )

        self.reg_constant = -math.log(2.0 * self.d)

    def forward(self, loss_align: torch.Tensor, loss_unif: torch.Tensor) -> torch.Tensor:
        # kappa_align > 0
        kappa_align = torch.exp(self.s_align)

        # 1 / kappa^2 = exp(-2s)
        inv_kappa_sq = torch.exp(-2.0 * self.s_unif)

        weighted_align = kappa_align * loss_align
        weighted_unif = self.d * inv_kappa_sq * loss_unif

        reg_align = -self.s_align
        reg_unif = 2.0 * self.s_unif

        total_loss = weighted_align + weighted_unif + reg_align + reg_unif + self.reg_constant
        return total_loss

    @property
    def current_thermodynamics_status(self):
        with torch.no_grad():
            kappa_align = torch.exp(self.s_align).item()
            kappa = torch.exp(self.s_unif).item()
            effective_T = (kappa ** 2) / (2.0 * self.d)

            return {
                "kappa_align": kappa_align,
                "kappa": kappa,
                "effective_T": effective_T,
                "w_align": kappa_align,
                "w_unif": self.d / (kappa ** 2),
            }