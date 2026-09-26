"""Encoder-only ABD-NET components used by the diffusion policy.

This is intentionally separate from ``abdnet_actor.py``.  The diffusion policy
uses only Phi and dynamics-informed message passing M; it has no ABD action
decoder, Gaussian policy head, or critic.
"""

from __future__ import annotations

import math

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_kinematic_dgl_graph(articulation, action_joint_names=None):
    """Build the same controlled-link graph used by ``ppo.py``.

    Edges point from child to parent. The selected controlled joints define
    both the included links and their motor order.
    """

    if action_joint_names is None:
        action_joint_names = [
            joint.get_name()
            for joint in articulation.get_active_joints()
            if "root" not in joint.get_name()
        ]
    action_joint_names = list(action_joint_names)

    actuated_joints = [
        joint
        for joint in articulation.get_joints()
        if joint.get_name() in action_joint_names
    ]
    found_joint_names = {joint.get_name() for joint in actuated_joints}
    missing_joint_names = set(action_joint_names) - found_joint_names
    if missing_joint_names:
        raise ValueError(
            f"Action joints missing from articulation: {missing_joint_names}"
        )

    link_names = []
    for joint in actuated_joints:
        parent_name = joint.get_parent_link().get_name()
        child_name = joint.get_child_link().get_name()
        if parent_name not in link_names:
            link_names.append(parent_name)
        if child_name not in link_names:
            link_names.append(child_name)

    link_index = {link_name: index for index, link_name in enumerate(link_names)}
    child_indices = []
    parent_indices = []
    parent_of = [-1] * len(link_names)
    edge_is_actuated = []
    node_is_actuated = [False] * len(link_names)
    motor_order = [-1] * len(link_names)

    for joint in actuated_joints:
        child_index = link_index[joint.get_child_link().get_name()]
        parent_index = link_index[joint.get_parent_link().get_name()]
        child_indices.append(child_index)
        parent_indices.append(parent_index)
        parent_of[child_index] = parent_index
        edge_is_actuated.append(True)
        node_is_actuated[child_index] = True
        motor_order[child_index] = action_joint_names.index(joint.get_name())

    root_candidates = [
        index for index, parent_index in enumerate(parent_of) if parent_index == -1
    ]
    if len(root_candidates) != 1:
        raise ValueError(
            "Expected one root in the robot kinematic tree, "
            f"found indices {root_candidates}"
        )

    graph = dgl.graph(
        (child_indices, parent_indices),
        num_nodes=len(link_names),
    )
    graph.edata["is_actuated"] = torch.tensor(
        edge_is_actuated,
        dtype=torch.bool,
    )
    graph.ndata["is_actuated"] = torch.tensor(
        node_is_actuated,
        dtype=torch.bool,
    )
    graph.ndata["motor_order"] = torch.tensor(motor_order, dtype=torch.long)

    if graph.num_nodes() != len(action_joint_names) + 1:
        raise ValueError(
            "Expected one controlled-tree link per action joint plus the root: "
            f"links={graph.num_nodes()}, action_joints={len(action_joint_names)}"
        )
    if graph.num_edges() != len(action_joint_names):
        raise ValueError(
            "Expected one controlled-tree edge per action joint: "
            f"edges={graph.num_edges()}, action_joints={len(action_joint_names)}"
        )

    return graph, root_candidates[0], link_names, parent_of


def _initialize_linear(
    layer: nn.Linear,
    standard_deviation: float = math.sqrt(2.0),
) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, standard_deviation)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class DynamicsMessagePassing(nn.Module):
    """Leaf-to-root ABD message passing over a batched DGL graph."""

    def __init__(self, num_links: int, feature_dim: int):
        super().__init__()
        self.num_links = num_links
        self.feature_dim = feature_dim

        self.inertia_offset = nn.Parameter(torch.zeros(num_links, feature_dim))

        initial_feature_value = np.log(2.0)
        initial_projection = 0.9
        motion_basis_scale = np.sqrt(
            initial_projection / initial_feature_value
        )
        self.motion_basis = nn.Parameter(
            torch.eye(feature_dim).unsqueeze(0).repeat(num_links, 1, 1)
            * motion_basis_scale
        )

    @staticmethod
    def _reduce(nodes):
        return {"message": nodes.mailbox["articulated_value"].sum(dim=1)}

    @staticmethod
    def _apply_node_update(nodes):
        return {"value": nodes.data["value"] + nodes.data["message"]}

    @staticmethod
    def _message(edges):
        child_value = edges.src["value"]
        motion_basis = edges.src["motion_basis"]

        projected_value = torch.bmm(
            motion_basis,
            torch.bmm(
                motion_basis.transpose(1, 2),
                child_value.unsqueeze(-1),
            ),
        ).squeeze(-1)

        positive_projection = F.softplus(projected_value, beta=5.0)
        bounded_projection = 1.0 - torch.reciprocal(
            1.0 + positive_projection
        )
        return {
            "articulated_value": child_value * (1.0 - bounded_projection)
        }

    def forward(self, graph: dgl.DGLGraph, encoded_links: torch.Tensor):
        num_nodes = encoded_links.shape[0]
        if num_nodes % self.num_links != 0:
            raise ValueError(
                f"Expected a multiple of {self.num_links} graph nodes, "
                f"received {num_nodes}"
            )

        graph = graph.local_var()
        batch_size = num_nodes // self.num_links
        inertia_offset = self.inertia_offset.repeat(batch_size, 1)

        graph.ndata["message"] = torch.zeros_like(encoded_links)
        graph.ndata["motion_basis"] = self.motion_basis.repeat(
            batch_size, 1, 1
        )
        graph.ndata["value"] = F.softplus(encoded_links + inertia_offset)

        dgl.prop_nodes_topo(
            graph,
            message_func=self._message,
            reduce_func=self._reduce,
            apply_node_func=self._apply_node_update,
            reverse=False,
        )
        return graph.ndata["value"]


class ABDConditionEncoder(nn.Module):
    """Encode flat state observations into post-M per-link features."""

    def __init__(
        self,
        graph: dgl.DGLGraph,
        observation_dim: int,
        feature_dim: int = 64,
        phi_hidden_dim: int = 64,
    ):
        super().__init__()
        self.graph = graph
        self.num_links = graph.num_nodes()
        self.feature_dim = feature_dim
        self.phi = nn.ModuleList(
            [
                nn.Sequential(
                    _initialize_linear(
                        nn.Linear(observation_dim, phi_hidden_dim)
                    ),
                    nn.Tanh(),
                    _initialize_linear(
                        nn.Linear(phi_hidden_dim, feature_dim)
                    ),
                    nn.Tanh(),
                )
                for _ in range(self.num_links)
            ]
        )
        self.message_passing = DynamicsMessagePassing(
            self.num_links,
            feature_dim,
        )
        self._batched_graphs = {}

    def _get_batched_graph(self, batch_size: int, device: torch.device):
        cache_key = (batch_size, str(device))
        if cache_key not in self._batched_graphs:
            self._batched_graphs[cache_key] = dgl.batch(
                [self.graph] * batch_size
            ).to(device)
        return self._batched_graphs[cache_key]

    def forward(self, observations: torch.Tensor):
        if observations.ndim != 2:
            raise ValueError(
                "ABDConditionEncoder expects [batch, observation_dim], "
                f"received {tuple(observations.shape)}"
            )

        batch_size = observations.shape[0]
        encoded_links = torch.stack(
            [link_encoder(observations) for link_encoder in self.phi],
            dim=1,
        )
        encoded_links_flat = encoded_links.reshape(
            batch_size * self.num_links,
            self.feature_dim,
        )
        batched_graph = self._get_batched_graph(
            batch_size,
            encoded_links_flat.device,
        )
        propagated_links = self.message_passing(
            batched_graph,
            encoded_links_flat,
        )
        return propagated_links.reshape(
            batch_size,
            self.num_links,
            self.feature_dim,
        )


def orthogonality_loss(
    message_passing: DynamicsMessagePassing,
    link_features: torch.Tensor,
) -> torch.Tensor:
    """Compute ABD weighted orthogonality for a batch without Kxdxd copies.

    Args:
        message_passing: The encoder's message-passing module.
        link_features: Tensor shaped ``[batch, num_links, feature_dim]``.
    """

    motion_basis = message_passing.motion_basis
    expected_shape = motion_basis.shape[:2]
    if link_features.ndim != 3 or link_features.shape[1:] != expected_shape:
        raise ValueError(
            f"Expected link features [batch, {expected_shape[0]}, "
            f"{expected_shape[1]}], received {tuple(link_features.shape)}"
        )

    row_gram_squared = torch.bmm(
        motion_basis,
        motion_basis.transpose(1, 2),
    ).square()
    quadratic_term = torch.einsum(
        "bki,kij,bkj->bk",
        link_features,
        row_gram_squared,
        link_features,
    )
    row_norm_squared = motion_basis.square().sum(dim=-1)
    trace_term = torch.einsum(
        "bki,ki->bk",
        link_features,
        row_norm_squared,
    )
    loss_per_link = (
        quadratic_term - 2.0 * trace_term + motion_basis.shape[-1]
    )
    return loss_per_link.clamp_min(0.0).mean()
