#Rough_env_cfg from Manager Based RL, Config H1.

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg as CurrTerm
import torch
from isaaclab.managers import CommandTermCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
# source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/mdp/rewards.py

import isaaclab_tasks.manager_based.navigation.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
# from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.h1_navigation_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

# from omni.isaac.lab.managers import TimedataCfg
##
# Pre-defined configs
##
from isaaclab_assets import H1_MINIMAL_CFG  # isort: skip

import math



# @configclass
# class ObservationsCfg:
#     """Observation specifications for the MDP."""

#     @configclass
#     class PolicyCfg(ObsGroup):
#         """Observations for policy group."""
#         pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})

#         # observation terms (order preserved)
#         base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
#         base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
#         projected_gravity = ObsTerm(
#             func=mdp.projected_gravity,
#             noise=Unoise(n_min=-0.05, n_max=0.05),
#         )
#         velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
#         joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
#         joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
#         actions = ObsTerm(func=mdp.last_action)
#         height_scan = ObsTerm(
#             func=mdp.height_scan,
#             params={"sensor_cfg": SceneEntityCfg("height_scanner")},
#             noise=Unoise(n_min=-0.1, n_max=0.1),
#             clip=(-1.0, 1.0),
#         )

#added code
        # time_left = ObsTerm(
        #     func=lambda env: (
        #         env.command_manager.get_term("pose_command").time_left / 
        #         env.command_manager.get_term("pose_command").cfg.resampling_time_range[1]
        #     ).unsqueeze(-1),
        #     scale=1.0,
        #     clip=(0.0, 1.0)
        # )

        # target_position = ObsTerm(
        #     func=lambda env: (
        #         (env.command_manager.get_term("pose_command").pos_command_w[:, :2] - 
        #         env.scene["robot"].data.root_link_pos_w[:, :2]).view(-1, 2)  # Ensure shape (num_envs, 2)
        #     ),
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     scale=0.2,
        #     clip=(-1.0, 1.0)
        # )

    #     def __post_init__(self):
    #         self.enable_corruption = True
    #         self.concatenate_terms = True

    # # observation groups
    # policy: PolicyCfg = PolicyCfg()


# @configclass
# class CommandsCfg:
#     """Command specifications for the MDP."""



#     pose_command = mdp.UniformPose2dCommandCfg(
#         asset_name="robot",
#         simple_heading=False,
#         resampling_time_range=(8, 8),
#         debug_vis=True,
#         # Generate a tensor with [pos_x, pos_y, heading]
#         ranges=mdp.UniformPose2dCommandCfg.Ranges(
#             pos_x=(-3.0, 3.0),
#             pos_y=(-3.0, 3.0),
#             heading=(-math.pi, math.pi)
#         ),
#     )


@configclass
class H1Rewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)


#code not being used now

    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 4, "command_name": "pose_command", "asset_cfg": SceneEntityCfg("robot", body_names=".*torso_link")},
    )
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 0.4, "command_name": "pose_command", "asset_cfg": SceneEntityCfg("robot", body_names=".*torso_link")},
    )
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.2,
        params={"command_name": "pose_command"},
    )


#navigation related code from blind_3 paper

    terminal_reward = RewTerm(
        func=mdp.compute_terminal_reward,
        weight=1.0,
        params={"Tr": 1.0}
    )

    stalling_penalty = RewTerm(
        func=mdp.compute_stalling_penalty,
        weight=1.0,
        params={}
    )


    exploration_reward = RewTerm(
        func=mdp.compute_exploration_reward,
        weight=1.0,  # Adjust the weight as needed.
        params={"remove_threshold": 0.5}  # 0.5 corresponds to 50% of maximum terminal reward.
    )


    stop_reward = RewTerm(
        func=mdp.compute_stop_reward,
        weight=-2.0,  # adjust weight as needed
        params={
            "dist_threshold": 0.1,  # robot is considered "at the target" when within 0.2 m
            "vel_threshold": 0.01,  # robot should be nearly stationary below 0.05 m/s
            "stop_penalty": -20.0  # penalty applied when the robot is close but still moving
        }
    )

    lin_vel_z_l2 = None
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=0.00001,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=0.00001, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.00001,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.00001,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_link"),
        },
    )
    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_ankle")}
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw", ".*_hip_roll"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow"])},
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso")}
    )


@configclass
class H1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: H1Rewards = H1Rewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.scene.height_scanner:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"


        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [".*torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # Terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [".*torso_link"]

        # Rewards
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ".*torso_link"


@configclass
class H1RoughEnvCfg_PLAY(H1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 4
        self.episode_length_s = 8


        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

