import argparse
import os
import pickle
import shutil

from legged_env import LeggedEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from datetime import datetime
import re
def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        "robot_urdf": "urdf/go1/urdf/go1.urdf",
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RL_hip_joint": 0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "dof_names": [
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
        ],
        "joint_limits": {
            "FL_hip_joint": {"lower": -0.863, "upper": 0.863, "effort": 23.7, "velocity": 30.1},
            "FL_thigh_joint": {"lower": -0.686, "upper": 4.501, "effort": 23.7, "velocity": 30.1},
            "FL_calf_joint": {"lower": -2.818, "upper": -0.888, "effort": 35.55, "velocity": 20.06},

            "FR_hip_joint": {"lower": -0.863, "upper": 0.863, "effort": 23.7, "velocity": 30.1},
            "FR_thigh_joint": {"lower": -0.686, "upper": 4.501, "effort": 23.7, "velocity": 30.1},
            "FR_calf_joint": {"lower": -2.818, "upper": -0.888, "effort": 35.55, "velocity": 20.06},

            "RL_hip_joint": {"lower": -0.863, "upper": 0.863, "effort": 23.7, "velocity": 30.1},
            "RL_thigh_joint": {"lower": -0.686, "upper": 4.501, "effort": 23.7, "velocity": 30.1},
            "RL_calf_joint": {"lower": -2.818, "upper": -0.888, "effort": 35.55, "velocity": 20.06},
            
            "RR_hip_joint": {"lower": -0.863, "upper": 0.863, "effort": 23.7, "velocity": 30.1},
            "RR_thigh_joint": {"lower": -0.686, "upper": 4.501, "effort": 23.7, "velocity": 30.1},
            "RR_calf_joint": {"lower": -2.818, "upper": -0.888, "effort": 35.55, "velocity": 20.06},
        },
        # PD
        "kp": 20.0,
        "kd": 0.5,
        "soft_dof_pos_limit": 0.9,
        "soft_dof_vel_limit": 1.0,
        "soft_torque_limit": 1.0,
        # termination
        "termination_contact_names": [
            "base"
        ],
        "penalised_contact_names": [
            "base",
            "FL_thigh",
            "FR_thigh",
            "RL_thigh",
            "RR_thigh",
            "FL_calf",
            "FR_calf",
            "RL_calf",
            "RR_calf",
        ],  
        "feet_names": [
            "FL_foot",
            "FR_foot",
            "RL_foot",
            "RR_foot",            
        ],
        "hip_names": [
            "FL_hip",
            "FR_hip",
            "RL_hip",
            "RR_hip",            
        ],
        "termination_if_roll_greater_than": 0,  # degree. 0 menas no use
        "termination_if_pitch_greater_than": 0,
        "termination_duration": 3, #seconds
        # base pose
        "base_init_pos": [0.0, 0.0, 0.5],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 53,
        "num_privileged_obs": 56,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
        "clip_observations":100,
    }

    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.25,
        "step_period": 0.8,
        "step_offset": 0.5,
        "feet_height_target": 0.1,
        "reward_scales": {
            "tracking_lin_vel": 1.5,
            "tracking_ang_vel": 0.75,
            "lin_vel_z": -0.001,
            "ang_vel_xy": -0.05,
            "collision": -5.0,
            "action_rate": -0.01,
            "contact_no_vel": -0.3,
            "dof_acc": -2.5e-7,
            "hip_pos": -0.2,
            "contact": 0.15,
            "dof_pos_limits": -3.0,
            "dof_vel_limits": 0.0,
            "torques": -0.00001,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-1.0, 1.0],
        "lin_vel_y_range": [-1.0, 1.0],
        "ang_vel_range": [-1.0, 1.0],
    }
    noise_cfg = {
        "add_noise": True,
        "noise_level": 1.0,
        "noise_scales":{
            "dof_pos": 0.01,
            "dof_vel": 1.5,
            "lin_vel": 0.1,
            "ang_vel": 0.2,
            "gravity": 0.05,
        }

    }
    terrain_cfg = {
        "subterrain_size": 12.0,
        "horizontal_scale": 0.25,
        "vertical_scale": 0.005,
        "cols": 10,
        "rows":10,
        "selected_terrains":{
            "flat_terrain" : {"probability": .001},
            "random_uniform_terrain" : {"probability": .1},
            "pyramid_sloped_terrain" : {"probability": 0.5},
            "discrete_obstacles_terrain" : {"probability": 0.5},
            "pyramid_stairs_terrain" : {"probability": .9},
            "wave_terrain": {"probability": .1},
        }
    }

    

    return env_cfg, obs_cfg, noise_cfg, reward_cfg, command_cfg, terrain_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go1_rough_walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=100)
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint if this flag is set")
    parser.add_argument("--ckpt", type=int, default=0)
    parser.add_argument("--view", type=bool, default=False)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir_ = f"logs/{args.exp_name}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_dir_, timestamp)
    env_cfg, obs_cfg, noise_cfg, reward_cfg, command_cfg, terrain_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    
    env = LeggedEnv(
        num_envs=args.num_envs, 
        env_cfg=env_cfg, 
        obs_cfg=obs_cfg, 
        noise_cfg=noise_cfg, 
        reward_cfg=reward_cfg, 
        command_cfg=command_cfg,        
        terrain_cfg=terrain_cfg,        
        show_viewer=args.view,

    )

    if args.resume:
        # Get all subdirectories in the base log directory
        subdirs = [d for d in os.listdir(log_dir_) if os.path.isdir(os.path.join(log_dir_, d))]

        # Sort subdirectories by their names (assuming they are timestamped in lexicographical order)
        most_recent_subdir = sorted(subdirs)[-1] if subdirs else None
        most_recent_path = os.path.join(log_dir_, most_recent_subdir)

        if args.ckpt == 0:
            # List all files in the most recent subdirectory
            files = os.listdir(most_recent_path)

            # Regex to match filenames like 'model_100.pt' and extract the number
            model_files = [(f, int(re.search(r'model_(\d+)\.pt', f).group(1)))
                        for f in files if re.search(r'model_(\d+)\.pt', f)]
            model_file = max(model_files, key=lambda x: x[1])[0]
        else:
            model_file = f"model_{args.ckpt}.pt"
        resume_path = os.path.join(most_recent_path,  model_file)

    os.makedirs(log_dir, exist_ok=True)        
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    if args.resume:
        runner.load(resume_path)

    pickle.dump(
        [env_cfg, obs_cfg, noise_cfg, reward_cfg, command_cfg, train_cfg, terrain_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

