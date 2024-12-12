from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

class DistillJS(nn.Module):
    """Distilling the Knowledge via Jensen-Shannon Divergence"""
    def __init__(self, T):
        super(DistillJS, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s_log = F.log_softmax(y_s / self.T, dim=1)
        p_s = p_s_log.exp()
        p_t_log = F.log_softmax(y_t / self.T, dim=1)
        p_t = p_t_log.exp()

        m = 0.5 * (p_s + p_t)
        kl_pm = F.kl_div(p_s_log, m, reduction='batchmean')
        kl_qm = F.kl_div(p_t_log, m, reduction='batchmean')
        js = 0.5 * (kl_pm + kl_qm) * (self.T**2)
        return js


class DistillChiSquare(nn.Module):
    """Distilling the Knowledge via Chi-square Divergence"""
    def __init__(self, T):
        super(DistillChiSquare, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        eps = 1e-9
        chi = (p_s * p_s) / (p_t + eps)
        chi = chi.sum(dim=1) - 1.0
        chi = chi.mean() * (self.T**2)
        return chi

class DistillRenyiAlpha(nn.Module):
    """Distilling the Knowledge via Rényi Divergence of order alpha"""
    def __init__(self, T, alpha):
        super(DistillRenyiAlpha, self).__init__()
        self.T = T
        self.alpha = alpha
        if self.alpha == 1.0:
            raise ValueError("Alpha should not be 1.0 for Rényi divergence.")

    def forward(self, y_s, y_t):
        p_s = F.softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        eps = 1e-9
        Z = torch.sum((p_s.clamp(eps, 1.0)**self.alpha) * (p_t.clamp(eps, 1.0)**(1.0 - self.alpha)), dim=1)
        D_alpha = (1.0/(self.alpha - 1.0)) * torch.log(Z)
        D_alpha = D_alpha.mean() * (self.T**2)
        return D_alpha
class DistillLeCam(nn.Module):
    """Distilling the Knowledge via Le Cam Distance (same as TV)"""
    def __init__(self, T):
        super(DistillLeCam, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        # Le Cam distance = Total Variation distance
        p_s = F.softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        # TV = 0.5 * sum |p_i - q_i|
        # Le Cam distance is equal to this value.
        diff = torch.abs(p_s - p_t)
        lecam_dist = 0.5 * diff.sum(dim=1).mean() * (self.T**2)
        return lecam_dist
class DistillTV(nn.Module):
    """Distilling the Knowledge via Total Variation Distance"""
    def __init__(self, T):
        super(DistillTV, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        diff = torch.abs(p_s - p_t)
        # TV distance = 0.5 * sum_i |p_i - q_i|
        tv = 0.5 * diff.sum(dim=1).mean() * (self.T**2)
        return tv
class DistillPower(nn.Module):
    """Distilling the Knowledge via Power Divergence (Cressie-Read family)"""
    def __init__(self, T, lam):
        super(DistillPower, self).__init__()
        self.T = T
        self.lam = lam
        if self.lam == 0 or self.lam == -1:
            raise ValueError("lambda must not be 0 or -1 for power divergence.")

    def forward(self, y_s, y_t):
        p_s = F.softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        eps = 1e-9
        ratio = (p_s + eps) / (p_t + eps)

        # D_lambda(P||Q) = (1/(lambda*(lambda+1))) * sum_i p(i)[(p(i)/q(i))^lambda - 1]
        power_div = torch.mean(torch.sum(p_s * (ratio.pow(self.lam) - 1.0), dim=1))
        factor = 1.0 / (self.lam * (self.lam + 1.0))
        power_div = factor * power_div * (self.T**2)
        return power_div