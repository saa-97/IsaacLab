# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul


from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)

def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)

def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()

def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)

def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)

def position_command_error_tanh_navigation(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)

def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)

def compute_terminal_reward(env, Tr=1.0):

    time_left = env.command_manager.get_term("pose_command").time_left  # e.g., [N, M]
    active = (time_left < Tr).float()  # still [N, M]
    
    xb = env.scene["robot"].data.root_link_pos_w[:, :3]  # [N, 3]
    target = env.command_manager.get_term("pose_command").pos_command_w[:, :3]  # [N, 3] or [N, M, 3]
    
    if target.dim() == 2:
        target = target.unsqueeze(1)  # becomes [N, 1, 3]
    xb = xb.unsqueeze(1)  # becomes [N, 1, 3]
    
    dist2 = torch.sum((xb - target)**2, dim=-1)
    
    reward = active * (1.0 / Tr) * (1.0 / (1.0 + dist2))
    return reward.mean(dim=1)

def compute_stalling_penalty(env):
    """
    Computes a penalty for stalling far from the target, matching paper's r_stall.
    """
    v = torch.norm(env.scene["robot"].data.root_com_lin_vel_w[:, :3], dim=1, keepdim=True)
    
    xb = env.scene["robot"].data.root_link_pos_w[:, :3].unsqueeze(1)  # [N, 1, 3]
    target = env.command_manager.get_term("pose_command").pos_command_w[:, :3]
    if target.dim() == 2:
        target = target.unsqueeze(1)  # [N, 1, 3]
    dist = torch.norm(xb - target, dim=-1)
    
    condition = ((v < 0.1) & (dist > 0.5)).float()  # [N, 1]
    penalty = -1.0 * condition
    return penalty.mean(dim=1)

def compute_exploration_reward(env, task_reward_tracker=0.0, is_terminal=False, remove_threshold=0.5):

    v = env.scene["robot"].data.root_com_lin_vel_w[:, :3]    # shape [N, 3]
    x = env.scene["robot"].data.root_link_pos_w[:, :3]       # shape [N, 3]
    x_star = env.command_manager.get_term("pose_command").pos_command_w[:, :3]  # shape [N, 3]
    
    if x_star.dim() == 3:  
        x_star = x_star.squeeze(1)  # now shape [N, 3]
    
    dot_prod = torch.sum(v * (x_star - x), dim=1)  # shape [N]
    norm_v = torch.norm(v, dim=1) + 1e-6           # shape [N]
    norm_diff = torch.norm(x_star - x, dim=1) + 1e-6  # shape [N]
    r_bias = dot_prod / (norm_v * norm_diff)        # shape [N]
    

    if isinstance(task_reward_tracker, (int, float)):
        task_reward_tracker = torch.tensor(task_reward_tracker, device=v.device, dtype=torch.float32).repeat(v.shape[0])
    mask = (task_reward_tracker < remove_threshold).float()  # shape [N]
    
    return r_bias * mask  # shape [N]


def compute_stop_reward(env, dist_threshold=0.2, vel_threshold=0.05, stop_penalty=-10.0):

    x = env.scene["robot"].data.root_link_pos_w[:, :3]  # [N, 3]
    target = env.command_manager.get_term("pose_command").pos_command_w[:, :3]  # [N, 3]
    dist = torch.norm(x - target, dim=1)  # [N]
    
    vel = torch.norm(env.scene["robot"].data.root_com_lin_vel_w[:, :3], dim=1)  # [N]
    
    penalty = stop_penalty * ((dist < dist_threshold).float() * (vel > vel_threshold).float())
    return penalty  # [N]

def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def position_command_error_tanh_combined(
    env: ManagerBasedRLEnv,
    command_name: str,
    coarse_std: float,
    fine_std: float,
    fine_thresh: float,
) -> torch.Tensor:
    """Reward position tracking with a combined coarse and fine-grained tanh kernel.

    This function provides a coarse reward for getting into the general vicinity of the target
    and a more fine-grained reward for precise positioning once close.
    """
    command = env.command_manager.get_command(command_name)
    # command is [x, y, heading] in the robot's base frame
    des_pos_b = command[:, :2]
    distance = torch.norm(des_pos_b, dim=1)

    # Coarse reward (always active)
    coarse_reward = 1 - torch.tanh(distance / coarse_std)

    # Fine-grained reward (active only when close to the target)
    # Create a mask for environments where the robot is within the fine-grained threshold
    fine_mask = (distance < fine_thresh).float()
    fine_reward = 1 - torch.tanh(distance / fine_std)

    # Combine the rewards: use fine-grained reward when close, otherwise just coarse
    # The mask ensures the fine-grained component is only added when the condition is met.
    # We add the coarse reward to ensure there's always a signal.
    return coarse_reward + fine_mask * fine_reward

def reward_stand_still_at_target(
    env: ManagerBasedRLEnv, command_name: str, dist_thresh: float, lin_vel_std: float, ang_vel_std: float
) -> torch.Tensor:
    """Rewards the agent for standing still when close to the target position."""
    command = env.command_manager.get_command(command_name)
    # Target position in the robot's base frame
    des_pos_b = command[:, :2]
    distance = torch.norm(des_pos_b, dim=1)

    # Current velocities
    base_lin_vel = torch.norm(env.scene["robot"].data.root_lin_vel_w[:, :2], dim=1)
    base_ang_vel = torch.abs(env.scene["robot"].data.root_ang_vel_w[:, 2])

    # Exponentially shaped reward for low velocity
    lin_vel_reward = torch.exp(-base_lin_vel**2 / lin_vel_std**2)
    ang_vel_reward = torch.exp(-base_ang_vel**2 / ang_vel_std**2)

    # Create a mask to apply this reward only when the robot is close to the target
    close_to_target_mask = (distance < dist_thresh).float()

    # The final reward is the sum of velocity rewards, activated only when near the target
    return (lin_vel_reward + ang_vel_reward) * close_to_target_mask