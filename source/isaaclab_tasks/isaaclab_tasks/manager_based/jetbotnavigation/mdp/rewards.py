# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def position_command_error_tanh_navigation(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with negative distance for navigation tasks."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return -distance  # Negative distance: reward gets higher as robot gets closer to target


def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Reward for facing towards the target (negative heading error)."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return -heading_b.abs()  # Negative absolute heading error: reward gets higher when facing target


def forward_velocity_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward forward velocity in the robot's body frame (prevents backward driving)."""
    asset: Articulation = env.scene[asset_cfg.name]
    # Only reward positive (forward) velocities, penalize backward motion
    forward_vel = asset.data.root_com_lin_vel_b[:, 0]  # x-component in body frame
    return torch.clamp(forward_vel, min=0.0)  # Only positive forward velocities get reward


def alignment_reward_exp(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward alignment between robot's forward direction and target direction using exponential kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    target_pos = command[:, :3]  # Target position relative to robot
    
    # Get robot's forward direction vector in world frame
    forward_vec = math_utils.quat_apply(asset.data.root_quat_w, asset.data.FORWARD_VEC_B)
    
    # Normalize target direction (from robot to target)
    target_direction = target_pos / (torch.norm(target_pos, dim=1, keepdim=True) + 1e-8)
    
    # Compute alignment (dot product)
    alignment = torch.sum(forward_vec * target_direction, dim=1)
    
    # Use exponential to map [-1, 1] to [e^-1, e^1] ≈ [0.37, 2.72]
    # This prevents negative rewards when misaligned and amplifies positive alignment
    return torch.exp(alignment)


def forward_velocity_alignment_combined(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Combined reward: forward velocity multiplied by alignment to encourage forward driving toward target.
    
    This implements the approach from the IsaacLab tutorial where the robot gets rewarded for:
    1. Driving forward (positive forward velocity)
    2. Being aligned with the target direction
    
    The multiplication ensures the robot only gets high rewards when BOTH conditions are met.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    target_pos = command[:, :3]  # Target position relative to robot
    
    # Forward velocity in body frame (only positive values)
    forward_vel = torch.clamp(asset.data.root_com_lin_vel_b[:, 0], min=0.0)
    
    # Get robot's forward direction vector in world frame
    forward_vec = math_utils.quat_apply(asset.data.root_quat_w, asset.data.FORWARD_VEC_B)
    
    # Normalize target direction (from robot to target)
    target_direction = target_pos / (torch.norm(target_pos, dim=1, keepdim=True) + 1e-8)
    
    # Compute alignment (dot product)
    alignment = torch.sum(forward_vec * target_direction, dim=1)
    
    # Use exponential to ensure positive scaling
    alignment_scaled = torch.exp(alignment)
    
    # Multiply forward velocity by alignment: high reward only when moving forward AND aligned
    return forward_vel * alignment_scaled


def distance_to_target_reward(env: ManagerBasedRLEnv, command_name: str, std: float = 1.0) -> torch.Tensor:
    """Reward decreasing distance to target using exponential kernel."""
    command = env.command_manager.get_command(command_name)
    target_pos = command[:, :3]  # Target position relative to robot
    distance = torch.norm(target_pos, dim=1)
    
    # Use exponential decay: closer to target = higher reward
    return torch.exp(-distance / std)


def backward_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize backward motion to ensure robot only drives forward."""
    asset: Articulation = env.scene[asset_cfg.name]
    forward_vel = asset.data.root_com_lin_vel_b[:, 0]  # x-component in body frame
    
    # Return negative penalty for backward motion (when forward_vel < 0)
    return torch.clamp(forward_vel, max=0.0)  # Only negative (backward) velocities contribute
