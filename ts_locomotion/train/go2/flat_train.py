"""
go2_rough_train.py
──────────────────
Overrides only the terrain settings (and, if you like, any other dicts).
"""

from go2_base import train_main




# ---- patch only the bits that change --------------------------------------


env_cfg_patch = {
    "self_collision": False,
    "randomize_rot": True,
    "dof_lower_limit": [ #order matters!
        -0.8, 0.4, -2.5,
        -0.5, 0.4, -2.5,
        -0.8, -0.4,  -2.5,
        -0.5, -0.4,  -2.5,
    ],
    "dof_upper_limit": [ #order matters!
        0.5, 2.3, -0.8,
        0.8, 2.3, -0.8,
        0.5, 2.3, -0.8,
        0.8, 2.3, -0.8,
    ],
    # "penalized_contact_link_names": ["base", "thigh", "calf"],
    "base_init_pos": [0.0, 0.0, 0.45],
    # 'max_push_vel_xy': 0.0,
    "termination_if_roll_greater_than": 100,
}

reward_cfg_patch = {
    "soft_dof_pos_limit": 1.0,
    "reward_scales": {
        "tracking_lin_vel_in_world": 1.5,
        "tracking_ang_vel_in_world": 0.75,
        "lin_vel_z": -2.0,
        "relative_base_height": -30.0,
        "orientation": -30.0,
        "ang_vel_xy": -0.05,
        "collision": -5.0,
        "front_feet_clearance": 30.0,
        "rear_feet_clearance": 30.0,
        "foot_clearance": -0.5,
        "action_rate": -0.01,
        "dof_acc": -2.5e-7,
        "dof_pos_limits": -50.0,
        "dof_vel": 0.0,
        "torques": 0.0,
        "powers": -2e-5,
        "termination": -30.0,
        "similar_to_default": -0.01,
        "feet_contact_forces": -0.001,
        "stand_still": -0.5,
    },
}



terrain_cfg_patch = {
    "terrain_type": "plane",
}

command_cfg_patch = {
    "num_commands": 3,
    "curriculum": False,
    "curriculum_duration": 2000, #1 calculated 1 iteration is 1 seocnd 2000 = 
    "mean_reward_threshold": 15,
    "lin_vel_x_range": [-1.0, 1.0],
    "lin_vel_y_range": [-0.5, 0.5],
    "ang_vel_range": [-1.0, 1.0],
}



# leave other five cfgs untouched
CFG_PATCHES = (
    env_cfg_patch,  # env_cfg
    {},  # obs_cfg
    {},  # noise_cfg
    reward_cfg_patch,  # reward_cfg
    command_cfg_patch,  # command_cfg
    terrain_cfg_patch,
)

if __name__ == "__main__":
    train_main(cfg_patches=CFG_PATCHES, default_exp_name="go2_walking")
