import torch
import torch.nn as nn
import torch.nn.functional as F

def _pairwise_cosine_distance(x: torch.Tensor):
    # x: [N, C] (already L2-normalized outside)
    S = x @ x.t()                    # cosine similarity
    D = (1.0 - S).clamp_min(0.0)     # cosine distance
    return S, D

def _sinkhorn(a, b, C, eps=5e-2, iters=100):
    # a: [Ns], b: [Nt], sum=1;  C: [Ns, Nt] cost
    K = torch.exp(-C / eps)          # Gibbs kernel
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(iters):
        u = a / (K @ v + 1e-12)
        v = b / (K.t() @ u + 1e-12)
    P = torch.diag(u) @ K @ torch.diag(v)  # transport plan π
    return P

class GraphDistillationLoss(nn.Module):
    """
    L_total = w_node * Lnode + w_edge * Ledge + w_gw * LGW
    - 支持学生/教师节点数不同 (Ns != Nt)
    - 节点和边在 Ns!=Nt 时通过 π 做软对齐
    """
    def __init__(self,
                 w_node=1.0, w_edge=1.0, w_gw=1.0,
                 gw_eps=5e-2, gw_iters=100,
                 use_gw=True):
        super().__init__()
        self.w_node = w_node
        self.w_edge = w_edge
        self.w_gw = w_gw
        self.gw_eps = gw_eps
        self.gw_iters = gw_iters
        self.use_gw = use_gw
        self.mse = nn.MSELoss()

    def forward(self, student_graph, teacher_graph):
        # safe zeros for autograd
        device = next(self.parameters()).device if len(list(self.parameters())) else 'cpu'
        zero = torch.tensor(0.0, device=device, requires_grad=True)

        if student_graph is None or teacher_graph is None:
            return zero

        # [Ns, C], [Nt, C]
        s_nodes = student_graph["node_feats"]     # float
        t_nodes = teacher_graph["node_feats"]

        # L2 normalize => focus on direction (matches paper)
        s_norm = F.normalize(s_nodes, p=2, dim=-1)
        t_norm = F.normalize(t_nodes, p=2, dim=-1)

        Ns, Nt = s_norm.size(0), t_norm.size(0)

        # uniform marginals for GW
        a = torch.full((Ns,), 1.0 / max(Ns, 1), device=s_norm.device)
        b = torch.full((Nt,), 1.0 / max(Nt, 1), device=t_norm.device)

        # similarity & distance matrices
        S_s, D_s = _pairwise_cosine_distance(s_norm)   # [Ns, Ns]
        S_t, D_t = _pairwise_cosine_distance(t_norm)   # [Nt, Nt]

        # ---------- GW transport plan ----------
        # cost for node match (feature dissimilarity)
        C_feat = 1.0 - (s_norm @ t_norm.t())          # in [0,2], smaller=better
        with torch.enable_grad():
            pi = _sinkhorn(a, b, C_feat, eps=self.gw_eps, iters=self.gw_iters)  # [Ns, Nt]

        # ---------- Lnode ----------
        # soft aligned teacher features: T' = π * t  (row-normalized transport)
        row_sum = pi.sum(dim=1, keepdim=True).clamp_min(1e-12)
        t_soft = (pi @ t_norm) / row_sum               # [Ns, C]
        L_node = F.mse_loss(s_norm, t_soft)

        # ---------- Ledge ----------
        # soft align similarity: S_t' = π * S_t * π^T  (project to student graph size)
        S_t_proj = pi @ S_t @ pi.t()                   # [Ns, Ns]
        # normalize diagonals to 1 for stability
        eye_s = torch.eye(Ns, device=S_s.device)
        S_t_proj = S_t_proj - torch.diag(torch.diag(S_t_proj)) + eye_s
        L_edge = F.mse_loss(S_s, S_t_proj)

        # ---------- LGW ----------
        if self.use_gw:
            # GW discrepancy = sum_{i,j,k,l} |D_s(i,j) - D_t(k,l)|^2 π_{ik} π_{jl}
            # implement as: || D_s - π D_t π^T ||_F^2  (standard relax)
            D_t_proj = pi @ D_t @ pi.t()              # [Ns, Ns]
            L_gw = F.mse_loss(D_s, D_t_proj)
        else:
            L_gw = zero

        return self.w_node * L_node + self.w_edge * L_edge + self.w_gw * L_gw
