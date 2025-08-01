import torch
import torch.nn as nn

class GraphDistillationLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.mse = nn.MSELoss()

    def forward(self, student_graph, teacher_graph):
        if student_graph is None or teacher_graph is None:
            return 0.0

        s_nodes = student_graph["node_feats"]
        t_nodes = teacher_graph["node_feats"]
        adj = teacher_graph["adj_matrix"]  # assume same adj

        if s_nodes.shape != t_nodes.shape:
            return 0.0  # skip if unmatched node count

        # Node-level loss
        node_loss = self.mse(s_nodes, t_nodes)

        # Edge-level: difference in relative spatial vectors
        edge_loss = 0.0
        N = adj.shape[0]
        for i in range(N):
            for j in range(N):
                if adj[i, j] > 0:
                    ts = t_nodes[i] - t_nodes[j]
                    ss = s_nodes[i] - s_nodes[j]
                    edge_loss += self.mse(ss, ts)
        edge_loss = edge_loss / (adj.sum() + 1e-6)

        return self.weight * (node_loss + edge_loss)
