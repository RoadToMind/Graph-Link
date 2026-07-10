"""
ABD-NET Actor implementation.
Implements Phi (link-wise encoder), M (dynamics-informed message passing via DGL),
and Psi (per-joint action decoder), plus critic and stochastic sampling for PPO.

Paper: "Articulated Body Dynamics-Informed Neural Network Policy"
Equations referenced: (6), (7), (8), (9), (10), Algorithm 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
from torch.distributions.normal import Normal


# ---------------------------------------------------------------------------
# Graph construction (call once at startup from SAPIEN articulation)
# ---------------------------------------------------------------------------

def build_kinematic_dgl_graph(articulation):
    """
    Build a DGLGraph from a SAPIEN Articulation object.
    Edges are directed child -> parent so that prop_nodes_topo with
    reverse=False processes leaves first and the root last, matching
    Algorithm 1's leaf-to-root traversal order.

    Args:
        articulation: sapien.physx.PhysxArticulation (or ManiSkill wrapper)

    Returns:
        g         : DGLGraph, edges child -> parent
        root_idx  : int, index of the root link (no parent)
        link_names: list[str], link names in node-index order
        parent_of : list[int], parent_of[i] = parent index, or -1 for root
    """
    links = articulation.get_links()
    link_names = [l.get_name() for l in links]
    name_to_idx = {name: i for i, name in enumerate(link_names)}

    child_ids, parent_ids = [], []
    parent_of = [-1] * len(links)

    for joint in articulation.get_joints():
        parent_link = joint.get_parent_link()
        child_link = joint.get_child_link()
        if parent_link is None:
            # root-mounting joint with no parent link — skip
            continue
        c_idx = name_to_idx[child_link.get_name()]
        p_idx = name_to_idx[parent_link.get_name()]
        child_ids.append(c_idx)
        parent_ids.append(p_idx)
        parent_of[c_idx] = p_idx

    g = dgl.graph((child_ids, parent_ids), num_nodes=len(links))
    root_idx = parent_of.index(-1)  # exactly one link has no parent

    return g, root_idx, link_names, parent_of


# ---------------------------------------------------------------------------
# Module M: Dynamics-Informed Message Passing   (Sec. IV-B, Eqs. 6-8)
# ---------------------------------------------------------------------------

class DynamicsMessagePassing(nn.Module):
    """
    Implements the bottom-up ABA-inspired message passing (Eqs. 6-8 /
    lines 4-11 of Algorithm 1) via dgl.prop_nodes_topo.

    Per-link learnable parameters:
        B_i in R^d  -- analogous to rigid-body inertia I_i  (Eq. 3)
        W_i in R^{d x d} -- analogous to motion subspace S_j (Eq. 4)

    Forward:
        Input : z  [K, d]  -- link embeddings from Phi
        Output: v  [K, d]  -- link representations after tree propagation
    """

    def __init__(self, K: int, d: int):
        super().__init__()
        self.K = K
        self.d = d

        # B_i: zero-init so initial prior is "no intrinsic inertia offset"
        self.B = nn.Parameter(torch.zeros(K, d))

        # W_i: scaled identity init — near-orthonormal from the start,
        # small scale (0.1) so initial child attenuation is mild rather
        # than zeroing out all child contributions from step 1.
        self.W = nn.Parameter(
            torch.eye(d).unsqueeze(0).repeat(K, 1, 1) * 0.1
        )

    # -- DGL message/reduce/apply functions ----------------------------------

    def message_func(self, edges):
        """
        Eq. (8): v^a_j = v_j - v_j * (W_j W_j^T v_j)
        Computes the child contribution before sending it to the parent.
        edges.src = child side (edges are child -> parent).
        """
        v_j = edges.src['v']        # [E, d]
        W_j = edges.src['W']        # [E, d, d]

        # W_j^T v_j : project v_j into the learned motion-subspace basis
        Wt_v = torch.bmm(
            W_j.transpose(1, 2),
            v_j.unsqueeze(-1)        # [E, d, 1]
        )                            # -> [E, d, 1]

        # W_j (W_j^T v_j) : project back -- this is the "absorbed" component
        proj = torch.bmm(W_j, Wt_v).squeeze(-1)   # [E, d]

        # subtract the joint-absorbed component (Eq. 8)
        v_a = v_j - v_j * proj      # element-wise: diag(v_j) applied
        return {'v_a': v_a}

    def reduce_func(self, nodes):
        """
        Eq. (6): m_i = sum_{j in CH(i)} v^a_j
        Aggregates all child contributions into the parent's message buffer.
        Only fires for nodes that received at least one message (non-leaves).
        """
        m_i = nodes.mailbox['v_a'].sum(dim=1)   # [N_active, d]
        return {'m': m_i}

    def apply_node_func(self, nodes):
        """
        Eq. (7): v_i = softplus(z_i + B_i) + m_i
        softplus enforces positivity, mirroring positive-definiteness of
        physical inertia. m_i is 0 for leaves (pre-initialized in forward).
        """
        z_i = nodes.data['z']
        B_i = nodes.data['B']
        m_i = nodes.data['m']
        v_i = F.softplus(z_i + B_i) + m_i
        return {'v': v_i}

    # -- forward -------------------------------------------------------------

    def forward(self, g: dgl.DGLGraph, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g : DGLGraph with N = B*K nodes (batched across B environments)
            z : [B*K, d] link embeddings from Phi, in node-index order

        Returns:
            v : [B*K, d] link representations after leaf-to-root sweep
        """
        N = z.shape[0]   # B*K when batched
        device = z.device

        # local_var() scopes ndata writes to this call — safe for reuse
        g = g.local_var()

        # Populate node features for message/apply functions to read
        g.ndata['z'] = z
        # Tile B and W across the batch dimension (B copies of K params)
        # z has shape [B*K, d]; B_tiled must match node count B*K
        B_tiled = self.B.repeat(N // self.K, 1)      # [B*K, d]
        W_tiled = self.W.repeat(N // self.K, 1, 1)   # [B*K, d, d]
        g.ndata['B'] = B_tiled
        g.ndata['W'] = W_tiled

        # Pre-initialize m=0 for all nodes — Algorithm 1 line 4.
        # Critical: leaves never receive messages so reduce_func never
        # fires for them; without this they'd read uninitialized 'm'.
        g.ndata['m'] = torch.zeros(N, self.d, device=device)
        g.ndata['v'] = torch.zeros(N, self.d, device=device)  # placeholder

        # Propagate leaf-to-root in topological order (one frontier at a time)
        dgl.prop_nodes_topo(
            g,
            message_func=self.message_func,
            reduce_func=self.reduce_func,
            apply_node_func=self.apply_node_func,
            reverse=False,
        )

        return g.ndata['v']   # [B*K, d]


# ---------------------------------------------------------------------------
# Orthogonality regularization loss   (Eq. 9)
# ---------------------------------------------------------------------------

def orthogonality_loss(M_module: DynamicsMessagePassing,
                       v: torch.Tensor) -> torch.Tensor:
    """
    Eq. (9): L_orth = (1/K) * sum_i || W_i^T diag(v_i) W_i - I ||_F^2

    Penalizes deviation of W_i^T diag(v_i) W_i from identity, which is
    the assumption made to avoid computing (W^T diag(v) W)^{-1} in Eq.(8).

    Args:
        M_module : DynamicsMessagePassing instance
        v        : [K, d] link representations (single-env, not batched)
                   Use the mean over batch if calling during batched training.
    """
    W = M_module.W                              # [K, d, d]
    diag_v = torch.diag_embed(v)               # [K, d, d]
    WtDW = torch.bmm(
        W.transpose(1, 2),
        torch.bmm(diag_v, W)
    )                                           # [K, d, d]
    I = torch.eye(W.shape[-1], device=W.device).unsqueeze(0)  # [1, d, d]
    return ((WtDW - I) ** 2).sum(dim=(1, 2)).mean()           # scalar


# ---------------------------------------------------------------------------
# Full ABD-NET Actor+Critic   (Phi + M + Psi  +  MLP critic)
# ---------------------------------------------------------------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ABDNetAgent(nn.Module):
    """
    Full ABD-NET policy + value function, API-compatible with the MLP Agent
    in ppo.py (get_value, get_action, get_action_and_value).

    Architecture:
        Phi  : K independent per-link MLPs  obs -> z_i          (Sec. IV-A)
        M    : DynamicsMessagePassing        {z_i} -> {v_i}      (Sec. IV-B)
        Psi  : K-1 per-joint linear heads    v_{PA(j)} -> a_j    (Sec. IV-C)
        Critic: shared MLP on flat obs -> scalar value

    Args:
        g          : DGLGraph (child -> parent), built by build_kinematic_dgl_graph
        root_idx   : int
        parent_of  : list[int]  parent_of[i] = parent link index, -1 for root
        obs_dim    : int        dimension of the flat observation vector
        action_dim : int        total action dimension (sum of all joint DoFs)
        d          : int        hidden feature dimension (default 256)
        phi_hidden : int        hidden size of per-link phi MLP (default 64)
    """

    def __init__(
        self,
        g: dgl.DGLGraph,
        root_idx: int,
        parent_of: list,
        obs_dim: int,
        action_dim: int,
        d: int = 256,
        phi_hidden: int = 64,
    ):
        super().__init__()

        self.g = g
        self.root_idx = root_idx
        self.parent_of = parent_of
        self.K = g.num_nodes()
        self.d = d
        self.action_dim = action_dim

        # non-root link indices in deterministic order (for action assembly)
        self.joint_indices = [i for i in range(self.K) if i != root_idx]
        # DoF per joint — assume uniform split; override if robot has mixed DoF
        dof_per_joint = action_dim // len(self.joint_indices)
        self.dof_per_joint = dof_per_joint

        # -- Phi: per-link encoders (Sec. IV-A) ------------------------------
        # Each phi_i is a small MLP: obs_dim -> phi_hidden -> d
        # NOT weight-shared: each link gets its own parameters
        self.phi = nn.ModuleList([
            nn.Sequential(
                layer_init(nn.Linear(obs_dim, phi_hidden)),
                nn.Tanh(),
                layer_init(nn.Linear(phi_hidden, d)),
                nn.Tanh(),
            )
            for _ in range(self.K)
        ])

        # -- M: dynamics-informed message passing (Sec. IV-B) ----------------
        self.M = DynamicsMessagePassing(self.K, d)

        # -- Psi: per-joint action heads (Sec. IV-C) -------------------------
        # One linear head per non-root link, reading v_{PA(j)}
        self.psi = nn.ModuleList([
            layer_init(nn.Linear(d, dof_per_joint), std=0.01 * np.sqrt(2))
            for _ in self.joint_indices
        ])

        # -- Stochastic policy: learned log-std --------------------------------
        self.actor_logstd = nn.Parameter(
            torch.ones(1, action_dim) * -0.5
        )

        # -- Critic: standard MLP on flat obs (independent of ABD structure) --
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

    # -- internal helpers ----------------------------------------------------

    def _encode_and_propagate(self, s: torch.Tensor):
        """
        Run Phi then M on a batch of observations.

        Args:
            s : [B, obs_dim]

        Returns:
            v       : [B, K, d]   link representations
            v_flat  : [B*K, d]    same, flattened for DGL batching
        """
        B = s.shape[0]

        # Phi: [B, obs_dim] -> [B, K, d]
        # Each phi[i] maps [B, obs_dim] -> [B, d]; stack along dim 1
        z = torch.stack([self.phi[i](s) for i in range(self.K)], dim=1)  # [B, K, d]

        # Flatten to [B*K, d] for DGL batched graph
        z_flat = z.reshape(B * self.K, self.d)

        # Build batched graph: B copies of the same topology
        batched_g = dgl.batch([self.g] * B).to(z_flat.device)

        # M: leaf-to-root propagation -> [B*K, d]
        v_flat = self.M(batched_g, z_flat)

        # Unflatten: [B, K, d]
        v = v_flat.reshape(B, self.K, self.d)

        return v, v_flat

    def _decode_actions(self, v: torch.Tensor) -> torch.Tensor:
        """
        Psi: for each non-root link j, read v_{PA(j)} and decode action.

        Args:
            v : [B, K, d]

        Returns:
            action_mean : [B, action_dim]
        """
        parts = []
        for idx, j in enumerate(self.joint_indices):
            pa = self.parent_of[j]              # PA(j) index
            v_pa = v[:, pa, :]                  # [B, d]  -- Eq. (10)
            a_j = self.psi[idx](v_pa)           # [B, dof_per_joint]
            parts.append(a_j)
        return torch.cat(parts, dim=-1)         # [B, action_dim]

    # -- PPO-compatible API --------------------------------------------------

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        v, _ = self._encode_and_propagate(x)
        action_mean = self._decode_actions(v)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        return Normal(action_mean, action_std).sample()

    def get_action_and_value(self, x: torch.Tensor, action=None):
        """
        Full forward pass for PPO update step.

        Returns:
            action    : [B, action_dim]
            log_prob  : [B]
            entropy   : [B]
            value     : [B, 1]
            v         : [B, K, d]   link representations (needed for L_orth)
        """
        v, _ = self._encode_and_propagate(x)
        action_mean = self._decode_actions(v)

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
            v,                  # returned so caller can compute L_orth
        )