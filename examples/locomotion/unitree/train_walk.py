import argparse
import copy
import os
import pickle
import shutil

import numpy as np
import torch
import wandb
from reward_wrapper_for_walk import Walk
from locomotion_env import LocoEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from datetime import datetime
import re

def get_train_cfg(args):

    train_cfg_dict = {
        'algorithm': {
            'clip_param': 0.2,
            'desired_kl': 0.01,
            'entropy_coef': 0.01,
            'gamma': 0.99,
            'lam': 0.95,
            'learning_rate': 0.001,
            'max_grad_norm': 1.0,
            'num_learning_epochs': 5,
            'num_mini_batches': 4,
            'schedule': 'adaptive',
            'use_clipped_value_loss': True,
            'value_loss_coef': 1.0,
        },
        'init_member_classes': {},
        'policy': {
            'activation': 'elu',
            'actor_hidden_dims': [512, 256, 128],
            'critic_hidden_dims': [512, 256, 128],
            'init_noise_std': 1.0,
        },
        'runner': {
            'algorithm_class_name': 'PPO',
            'checkpoint': -1,
            'experiment_name': args.exp_name,
            'load_run': -1,
            'log_interval': 1,
            'max_iterations': args.max_iterations,
            'num_steps_per_env': 24,
            'policy_class_name': 'ActorCritic',
            'record_interval': 50,
            'resume': False,
            'resume_path': None,
            'run_name': '',
            'runner_class_name': 'runner_class_name',
            'save_interval': 100,
        },
        'runner_class_name': 'OnPolicyRunner',
        'seed': 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        'urdf_path': 'urdf/go2/urdf/go2.urdf',
        'links_to_keep': ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot',],
        'num_actions': 12,
        'num_dofs': 12,
        # joint/link names
        'default_joint_angles': {  # [rad]
            'FL_hip_joint': 0.0,
            'FR_hip_joint': 0.0,
            'RL_hip_joint': 0.0,
            'RR_hip_joint': 0.0,
            'FL_thigh_joint': 0.8,
            'FR_thigh_joint': 0.8,
            'RL_thigh_joint': 1.0,
            'RR_thigh_joint': 1.0,
            'FL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'RR_calf_joint': -1.5,
        },
        'dof_names': [
            'FR_hip_joint',
            'FR_thigh_joint',
            'FR_calf_joint',
            'FL_hip_joint',
            'FL_thigh_joint',
            'FL_calf_joint',
            'RR_hip_joint',
            'RR_thigh_joint',
            'RR_calf_joint',
            'RL_hip_joint',
            'RL_thigh_joint',
            'RL_calf_joint',
        ],
        'termination_contact_link_names': ['base'],
        'penalized_contact_link_names': ['base', 'thigh', 'calf'],
        'feet_link_names': ['foot'],
        'base_link_name': ['base'],
        # PD
        'PD_stiffness': {'joint': 30.0},
        'PD_damping': {'joint': 1.5},
        'use_implicit_controller': False,
        # termination
        'termination_if_roll_greater_than': 0.4,
        'termination_if_pitch_greater_than': 0.4,
        'termination_if_height_lower_than': -10.0,
        # base pose
        'base_init_pos': [0.0, 0.0, 0.42],
        'base_init_quat': [1.0, 0.0, 0.0, 0.0],
        # random push
        'push_interval_s': 5,
        'max_push_vel_xy': 1.0,
        # time (second)
        'episode_length_s': 20.0,
        'resampling_time_s': 4.0,
        'command_type': 'ang_vel_yaw',  # 'ang_vel_yaw' or 'heading'
        'action_scale': 0.25,
        'action_latency': 0.02,
        'clip_actions': 100.0,
        'send_timeouts': True,
        'control_freq': 50,
        'decimation': 4,
        'feet_geom_offset': 1,
        # domain randomization
        'randomize_friction': True,
        'friction_range': [0.2, 1.5],
        'randomize_base_mass': True,
        'added_mass_range': [-1., 3.],
        'randomize_com_displacement': True,
        'com_displacement_range': [-0.01, 0.01],
        'randomize_motor_strength': False,
        'motor_strength_range': [0.9, 1.1],
        'randomize_motor_offset': True,
        'motor_offset_range': [-0.02, 0.02],
        'randomize_kp_scale': True,
        'kp_scale_range': [0.8, 1.2],
        'randomize_kd_scale': True,
        'kd_scale_range': [0.8, 1.2],
        # coupling
        'coupling': False,
    }
    obs_cfg = {
        'num_obs': 9 + 3 * env_cfg['num_dofs'],
        'num_history_obs': 1,
        'obs_noise': {
            'ang_vel': 0.1,
            'gravity': 0.02,
            'dof_pos': 0.01,
            'dof_vel': 0.5,
        },
        'obs_scales': {
            'lin_vel': 2.0,
            'ang_vel': 0.25,
            'dof_pos': 1.0,
            'dof_vel': 0.05,
        },
        'num_priv_obs': 12 + 4 * env_cfg['num_dofs'],
    }
    reward_cfg = {
        'tracking_sigma': 0.25,
        'soft_dof_pos_limit': 0.9,
        'base_height_target': 0.3,
        'reward_scales': {
            'tracking_lin_vel': 1.0,
            'tracking_ang_vel': 0.5,
            'lin_vel_z': -2.0,
            'ang_vel_xy': -0.05,
            'orientation': -10.,
            'base_height': -50.,
            'torques': -0.0002,
            'collision': -1.,
            'dof_vel': -0.,
            'dof_acc': -2.5e-7,
            'feet_air_time': 1.0,
            'collision': -1.,
            'action_rate': -0.01,
        },
    }
    command_cfg = {
        'num_commands': 4,
        'lin_vel_x_range': [-1.0, 1.0],
        'lin_vel_y_range': [-1.0, 1.0],
        'ang_vel_range': [-1.0, 1.0],
    }
    terrain_cfg = {
        "terrain_type": "rough",
        "subterrain_size": 12.0,
        "horizontal_scale": 0.25,
        "vertical_scale": 0.005,
        "cols": 10,
        "rows": 10,
        "selected_terrains":{
            "flat_terrain" : {"probability": .001},
            "random_uniform_terrain" : {"probability": .1},
            "pyramid_sloped_terrain" : {"probability": .1},
            "discrete_obstacles_terrain" : {"probability": .1},
            "pyramid_stairs_terrain" : {"probability": .9},
            "wave_terrain": {"probability": .1},
        }
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_name', type=str, default='Go2')
    parser.add_argument('-v', '--vis', action='store_true', default=False)
    parser.add_argument('-c', '--cpu', action='store_true', default=False)
    parser.add_argument('-B', '--num_envs', type=int, default=10000)
    parser.add_argument('--max_iterations', type=int, default=1000)
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint if this flag is set")
    parser.add_argument('-o', '--offline', action='store_true', default=False)

    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--ckpt', type=int, default=1000)
    args = parser.parse_args()

    if args.debug:
        args.vis = True
        args.offline = True
        args.num_envs = 1

    gs.init(
        backend=gs.cpu if args.cpu else gs.gpu,
        logging_level='warning',
    )
    log_dir_ = f"logs/{args.exp_name}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_dir_, timestamp)
    env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg = get_cfgs()
    train_cfg = get_train_cfg(args)
    env = Walk(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        terrain_cfg=terrain_cfg,
        show_viewer=args.vis,
        eval=args.eval,
        debug=args.debug,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device='cuda:0')

    # if args.resume is not None:
    #     resume_dir = f'logs/{args.resume}'
    #     resume_path = os.path.join(resume_dir, f'model_{args.ckpt}.pt')
    #     print('==> resume training from', resume_path)
    #     runner.load(resume_path)
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
        print('==> resume training from', resume_path)
        runner.load(resume_path)


    # wandb.init(project='genesis', name=args.exp_name, dir=log_dir, mode='offline' if args.offline else 'online')
    os.makedirs(log_dir, exist_ok=True)  
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg, train_cfg],
        open(f'{log_dir}/cfgs.pkl', 'wb'),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    main()


'''
# training
python train_backflip.py -e EXP_NAME

# evaluation
python eval_backflip.py -e EXP_NAME --ckpt NUM_CKPT
'''