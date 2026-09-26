import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import dgl
import gymnasium as gym
import torch


BASELINE_DIR = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "baselines"
    / "diffusion_policy"
)
sys.path.insert(0, str(BASELINE_DIR))

import train_abd as train_abd_module
import train_abd_all as train_abd_all_module
from diffusion_policy.abd_encoder import (
    DynamicsMessagePassing,
    build_kinematic_dgl_graph,
    orthogonality_loss,
)


class DiffusionPolicyABDTest(unittest.TestCase):
    def setUp(self):
        train_abd_module.device = torch.device("cpu")
        train_abd_all_module.device = torch.device("cpu")
        self.observation_horizon = 2
        self.observation_dim = 7
        self.action_dim = 3
        self.feature_dim = 8
        self.graph = dgl.graph(([1, 2], [0, 1]), num_nodes=3)
        self.environment = SimpleNamespace(
            single_observation_space=gym.spaces.Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(self.observation_horizon, self.observation_dim),
            ),
            single_action_space=gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.action_dim,),
            ),
        )
        self.args = train_abd_module.Args(
            obs_horizon=self.observation_horizon,
            act_horizon=4,
            pred_horizon=16,
            abd_feature_dim=self.feature_dim,
            abd_phi_hidden_dim=8,
            unet_dims=[8, 16, 32],
            n_groups=8,
        )

    def make_agent(self):
        return train_abd_module.Agent(
            self.environment,
            self.args,
            self.graph,
            root_index=0,
        )

    def test_panda_abd_graph_selects_only_arm_joints(self):
        class Joint:
            def __init__(self, name, parent_name, child_name):
                self.name = name
                self.parent_link = SimpleNamespace(get_name=lambda: parent_name)
                self.child_link = SimpleNamespace(get_name=lambda: child_name)

            def get_name(self):
                return self.name

            def get_parent_link(self):
                return self.parent_link

            def get_child_link(self):
                return self.child_link

        arm_joint_names = [f"panda_joint{index}" for index in range(1, 8)]
        finger_joint_names = [
            "panda_finger_joint1",
            "panda_finger_joint2",
        ]
        arm_joints = [
            Joint(
                name,
                f"panda_link{index - 1}",
                f"panda_link{index}",
            )
            for index, name in enumerate(arm_joint_names, start=1)
        ]
        finger_joints = [
            Joint(finger_joint_names[0], "panda_hand", "panda_leftfinger"),
            Joint(finger_joint_names[1], "panda_hand", "panda_rightfinger"),
        ]
        arm_controller = SimpleNamespace(
            single_action_space=gym.spaces.Box(-1.0, 1.0, shape=(7,)),
            joints=arm_joints,
        )
        gripper_controller = SimpleNamespace(
            single_action_space=gym.spaces.Box(-1.0, 1.0, shape=(1,)),
            joints=finger_joints,
        )
        environment = SimpleNamespace(
            agent=SimpleNamespace(
                arm_joint_names=arm_joint_names,
                controller=SimpleNamespace(
                    controllers={
                        "arm": arm_controller,
                        "gripper": gripper_controller,
                    }
                ),
            )
        )

        self.assertEqual(
            train_abd_module.get_controller_joint_names(environment),
            arm_joint_names + finger_joint_names,
        )
        self.assertEqual(
            train_abd_module.get_abd_joint_names(environment),
            arm_joint_names,
        )

        articulation = SimpleNamespace(
            get_joints=lambda: arm_joints + finger_joints,
        )
        graph, root_index, link_names, _ = build_kinematic_dgl_graph(
            articulation,
            train_abd_module.get_abd_joint_names(environment),
        )
        self.assertEqual(graph.num_nodes(), 8)
        self.assertEqual(graph.num_edges(), 7)
        self.assertEqual(link_names[root_index], "panda_link0")
        self.assertNotIn("panda_hand", link_names)
        self.assertNotIn("panda_leftfinger", link_names)
        self.assertNotIn("panda_rightfinger", link_names)

    def test_condition_preserves_raw_state_and_uses_only_root(self):
        agent = self.make_agent()
        observations = torch.randn(
            4,
            self.observation_horizon,
            self.observation_dim,
        )

        condition, link_features = agent.build_obs_cond(observations)
        enriched_observations = condition.reshape(
            4,
            self.observation_horizon,
            self.observation_dim + self.feature_dim,
        )

        self.assertTrue(
            torch.equal(
                enriched_observations[..., : self.observation_dim],
                observations,
            )
        )
        expected_root_features = link_features[:, 0].reshape(
            4,
            self.observation_horizon,
            self.feature_dim,
        )
        self.assertTrue(
            torch.equal(
                enriched_observations[..., self.observation_dim :],
                expected_root_features,
            )
        )

    def test_all_node_condition_preserves_raw_state_and_every_link(self):
        all_node_args = train_abd_all_module.Args(
            obs_horizon=self.observation_horizon,
            pred_horizon=16,
            abd_phi_hidden_dim=8,
            unet_dims=[8, 16, 32],
            n_groups=8,
        )
        agent = train_abd_all_module.Agent(
            self.environment,
            all_node_args,
            self.graph,
            root_index=0,
        )
        observations = torch.randn(
            4,
            self.observation_horizon,
            self.observation_dim,
        )

        condition, link_features = agent.build_obs_cond(observations)
        all_node_feature_dim = (
            self.graph.num_nodes() * all_node_args.abd_feature_dim
        )
        enriched_observations = condition.reshape(
            4,
            self.observation_horizon,
            self.observation_dim + all_node_feature_dim,
        )

        self.assertEqual(all_node_args.abd_feature_dim, 256)
        self.assertEqual(agent.act_horizon, 12)
        self.assertTrue(
            torch.equal(
                enriched_observations[..., : self.observation_dim],
                observations,
            )
        )
        expected_link_features = link_features.reshape(
            4,
            self.observation_horizon,
            all_node_feature_dim,
        )
        self.assertTrue(
            torch.equal(
                enriched_observations[..., self.observation_dim :],
                expected_link_features,
            )
        )

    def test_agent_can_move_to_a_torch_device(self):
        agent = self.make_agent()
        moved_agent = agent.to(torch.device("cpu"))

        self.assertIs(moved_agent, agent)
        self.assertTrue(
            all(parameter.device.type == "cpu" for parameter in agent.parameters())
        )

    def test_diffusion_loss_backpropagates_through_abd(self):
        agent = self.make_agent()
        observations = torch.randn(
            2,
            self.observation_horizon,
            self.observation_dim,
        )
        actions = torch.randn(2, self.args.pred_horizon, self.action_dim)

        total_loss, diffusion_loss, abd_orthogonality_loss = agent.compute_loss(
            observations,
            actions,
        )
        total_loss.backward()

        self.assertTrue(
            torch.allclose(
                total_loss,
                diffusion_loss + self.args.orth_coef * abd_orthogonality_loss,
            )
        )
        phi_has_gradient = any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in agent.abd_encoder.phi.parameters()
        )
        message_passing_has_gradient = any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in agent.abd_encoder.message_passing.parameters()
        )
        self.assertTrue(phi_has_gradient)
        self.assertTrue(message_passing_has_gradient)

    def test_fast_orthogonality_loss_matches_dense_equation(self):
        num_links = 3
        feature_dim = 5
        message_passing = DynamicsMessagePassing(num_links, feature_dim)
        link_features = torch.rand(4, num_links, feature_dim)

        fast_loss = orthogonality_loss(message_passing, link_features)
        identity = torch.eye(feature_dim)
        dense_losses = []
        for sample_features in link_features:
            diagonal_features = torch.diag_embed(sample_features)
            weighted_gram = torch.bmm(
                message_passing.motion_basis.transpose(1, 2),
                torch.bmm(
                    diagonal_features,
                    message_passing.motion_basis,
                ),
            )
            dense_losses.append(
                ((weighted_gram - identity) ** 2).sum(dim=(1, 2)).mean()
            )

        dense_loss = torch.stack(dense_losses).mean()
        self.assertTrue(torch.allclose(fast_loss, dense_loss, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
