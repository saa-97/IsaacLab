# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg

from isaaclab.utils import configclass

from . import mdp

##
# Pre-defined configs
##

from isaaclab_assets.robots.jetbot import JETBOT_CONFIG  # isort:skip


##
# Scene definition
##


@configclass
class JetbotnavigationSceneCfg(InteractiveSceneCfg):
    """Configuration for a JetBot navigation scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = JETBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/base")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_pose = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,
        resampling_time_range=(15.0, 15.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-5.0, 5.0), 
            pos_y=(-5.0, 5.0), 
            heading=(-math.pi, math.pi)
        ),
    )

    # Add velocity commands for visualization arrows
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(15.0, 15.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,  # This enables the green/blue velocity arrows
        ranges=mdp.UniformVelocityCommandCfg.Ranges(lin_vel_x=(1.0, 3.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-2.0, 2.0)),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_velocity = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=[".*"], scale=10.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_pose"})
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task: main navigation rewards
    # Combined forward velocity and alignment reward (primary reward)
    forward_velocity_alignment = RewTerm(
        func=mdp.forward_velocity_alignment_combined, 
        weight=3.0, 
        params={"command_name": "base_pose", "asset_cfg": SceneEntityCfg("robot")}
    )
    
    # Distance to target reward (secondary reward)
    distance_to_target = RewTerm(
        func=mdp.distance_to_target_reward, 
        weight=1.0, 
        params={"command_name": "base_pose", "std": 2.0}
    )
    
    # -- penalties: prevent unwanted behaviors
    # Strong penalty for backward motion
    backward_penalty = RewTerm(
        func=mdp.backward_velocity_penalty, 
        weight=-5.0, 
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    
    # Stability penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.001)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # # (2) Base contact
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces"), "threshold": 1.0}
    # )

##
# Environment configuration
##


@configclass
class JetbotnavigationEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: JetbotnavigationSceneCfg = JetbotnavigationSceneCfg(num_envs=1024, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4  # Increased for more stable training
        self.episode_length_s = 10  # Longer episodes for better learning
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation

class JetbotnavigationEnvCfg_PLAY(JetbotnavigationEnvCfg):
    def __post_init__(self) -> None:
        """Post initialization for play mode."""
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 4.0
        # commands - use larger range for play
        self.commands.base_pose.ranges.pos_x = (-3.0, 3.0)
        self.commands.base_pose.ranges.pos_y = (-3.0, 3.0)
        # velocity commands for visualization
        self.commands.base_velocity.ranges.lin_vel_x = (1.5, 4.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-3.0, 3.0)
        # actions - keep the same scale for consistency
        self.episode_length_s = 8