import torch
import math
import genesis as gs
# from genesis.utils.terrain import parse_terrain

from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
import numpy as np
import random
import copy
def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

# Helper function to get quaternion from Euler angles
def quaternion_from_euler_tensor(roll_deg, pitch_deg, yaw_deg):
    """
    roll_deg, pitch_deg, yaw_deg: (N,) PyTorch tensors of angles in degrees.
    Returns a (N, 4) PyTorch tensor of quaternions in [x, y, z, w] format.
    """
    # Convert to radians
    roll_rad = torch.deg2rad(roll_deg)
    pitch_rad = torch.deg2rad(pitch_deg)
    yaw_rad = torch.deg2rad(yaw_deg)

    # Half angles
    half_r = roll_rad * 0.5
    half_p = pitch_rad * 0.5
    half_y = yaw_rad * 0.5

    # Precompute sines/cosines
    cr = half_r.cos()
    sr = half_r.sin()
    cp = half_p.cos()
    sp = half_p.sin()
    cy = half_y.cos()
    sy = half_y.sin()

    # Quaternion formula (XYZW)
    # Note: This is the standard euler->quat for 'xyz' rotation convention.
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    # Stack into (N,4)
    return torch.stack([qx, qy, qz, qw], dim=-1)


def get_height_at_xy(height_field, x, y, horizontal_scale, vertical_scale, center_x, center_y):
    # Convert world coordinates to heightfield indices
    mat = np.array([[0, 1/horizontal_scale],
                    [1/horizontal_scale, 0]])
    vec = np.array([x+center_x, y+center_y])
    result = mat @ vec
    i = int(result[1])
    j = int(result[0])
    if 0 <= i < height_field.shape[0] and 0 <= j < height_field.shape[1]:
        return height_field[i, j] * vertical_scale
    else:
        raise ValueError(f"Requested (x={x}, y={y}) is outside the terrain bounds.")


class LeggedEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, noise_cfg, reward_cfg, command_cfg, terrain_cfg, show_viewer=False, eval_=False, device="cuda"):
        self.cfg = {
            "env_cfg": env_cfg,
            "obs_cfg": obs_cfg,
            "noise_cfg": noise_cfg,
            "reward_cfg": reward_cfg,
            "command_cfg": command_cfg,
            "terrain_cfg": terrain_cfg,
        }
        self.eval = eval_
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = obs_cfg["num_privileged_obs"]
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.command_curriculum = command_cfg["curriculum"]
        self.curriculum_complete_flag = False
        self.curriculum_duration = command_cfg["curriculum_duration"]
        self.curriculum_step = 0
        # self.joint_limits = env_cfg["joint_limits"]
        self.simulate_action_latency = env_cfg["simulate_action_latency"]  # there is a 1 step latency on real robot
        self.dt = 1 / env_cfg['control_freq']
        sim_dt = self.dt / env_cfg['decimation']
        sim_substeps = 1
        self.max_episode_length_s = env_cfg['episode_length_s']
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.noise_cfg = noise_cfg
        self.terrain_cfg = terrain_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.clip_obs = obs_cfg["clip_observations"]
        self.reward_scales = reward_cfg["reward_scales"]
        self.noise_scales = noise_cfg["noise_scales"]
        self.selected_terrains = terrain_cfg["selected_terrains"]

        
        if self.env_cfg["randomize_delay"]:
            # 1️⃣ Define Delay Parameters
            self.min_delay, self.max_delay = self.env_cfg["delay_range"]  # Delay range in seconds
            self.max_delay_steps = int(self.max_delay / self.dt)  # Convert max delay to steps

            # 2️⃣ Initialize Delay Buffers
            self.action_delay_buffer = torch.zeros(
                (self.num_envs, self.num_actions, self.max_delay_steps + 1), device=self.device
            )
            self.motor_delay_steps = torch.randint(
                int(self.min_delay / self.dt), self.max_delay_steps + 1,
                (self.num_envs, self.num_actions), device=self.device
            )
        # create scene
        self.mean_reward_threshold = self.command_cfg["mean_reward_threshold"]
        self.terrain_type = terrain_cfg["terrain_type"]
        visualized_number = min(num_envs, 100)        # capped at 100
        if self.terrain_type == "plane" and visualized_number > 3:
            visualized_number = 3
        elif self.terrain_type == "custom_plane" and visualized_number > 3:
            visualized_number = 20

        self.rendered_envs_idx = list(range(visualized_number))    # 👈 save it

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=sim_dt,
                substeps=sim_substeps,
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1 / self.dt * self.env_cfg['decimation']),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=self.rendered_envs_idx),
            rigid_options=gs.options.RigidOptions(
                dt=sim_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_self_collision=env_cfg['self_collision'],
                enable_joint_limit=True,
            ),
            show_viewer=False,
        )


        self.show_vis = show_viewer
        self.selected_robot = 0
        if show_viewer:
            self.cam_0 = self.scene.add_camera(
                res=(640, 480),
                pos=(5.0, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=show_viewer        
            )
        else:
            self.cam_0 = self.scene.add_camera(
                # res=(640, 480),
                pos=(5.0, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=show_viewer        
            )
        self._recording = False
        self.mean_reward_flag = False
        self._recorded_frames = []


        subterrain_size = terrain_cfg["subterrain_size"]
        horizontal_scale = terrain_cfg["horizontal_scale"]
        vertical_scale = terrain_cfg["vertical_scale"]        
        if self.terrain_type != "plane" and self.terrain_type != "custom_plane":
            # # add plain

            ########################## entities ##########################
            self.cols = terrain_cfg["cols"]
            self.rows = terrain_cfg["rows"]
            n_subterrains=(self.cols, self.rows)
            terrain_types = list(self.selected_terrains.keys())
            probs = [terrain["probability"] for terrain in self.selected_terrains.values()]
            total = sum(probs)
            normalized_probs = [p / total for p in probs]
            subterrain_grid  = self.generate_subterrain_grid(self.rows, self.cols, terrain_types, normalized_probs)


            # Calculate the total width and height of the terrain
            total_width = (self.cols)* subterrain_size
            total_height =(self.rows)* subterrain_size

            # Calculate the center coordinates
            self.center_x = total_width / 2
            self.center_y = total_height / 2

            self.terrain  = gs.morphs.Terrain(
                pos=(-self.center_x,-self.center_y,0),
                subterrain_size=(subterrain_size, subterrain_size),
                n_subterrains=n_subterrains,
                horizontal_scale=horizontal_scale,
                vertical_scale=vertical_scale,
                subterrain_types=subterrain_grid
            )        


            self.terrain_min_x = - (total_width  / 2.0)
            self.terrain_max_x =   (total_width  / 2.0)
            self.terrain_min_y = - (total_height / 2.0)
            self.terrain_max_y =   (total_height / 2.0)
            # Calculate the center of each subterrain in world coordinates

            self.global_terrain = self.scene.add_entity(self.terrain)
            
        else:
            if self.terrain_type == "custom_plane":
                self.scene.add_entity(
                    gs.morphs.URDF(file="urdf/plane/custom_plane.urdf", fixed=True),
                ) 
            else:
                self.scene.add_entity(
                    gs.morphs.Plane(),
                )
            self.random_pos = self.generate_positions()
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        if self.env_cfg["use_mjcf"]:
            self.robot  = self.scene.add_entity(
                gs.morphs.MJCF(
                file=self.env_cfg["robot_description"],
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
                ),
            )
        else:
            self.robot = self.scene.add_entity(
                gs.morphs.URDF(
                    file=self.env_cfg["robot_description"],
                    merge_fixed_links=True,
                    links_to_keep=self.env_cfg['links_to_keep'],
                    pos=self.base_init_pos.cpu().numpy(),
                    quat=self.base_init_quat.cpu().numpy(),
                ),
            )
        self.envs_origins = torch.zeros((self.num_envs, 7), device=self.device)

        # build
        self.scene.build(n_envs=num_envs)
        if self.terrain_type != "plane" and self.terrain_type != "custom_plane":
            self.subterrain_centers = []
            # Get the terrain's origin position in world coordinates
            terrain_origin_x, terrain_origin_y, terrain_origin_z = self.terrain.pos
            self.height_field = self.global_terrain.geoms[0].metadata["height_field"]
            for row in range(self.rows):
                for col in range(self.cols):
                    subterrain_center_x = terrain_origin_x + (col + 0.5) * subterrain_size
                    subterrain_center_y = terrain_origin_y + (row + 0.5) * subterrain_size
                    # subterrain_center_z = (self.height_field[int(subterrain_center_x), int(subterrain_center_y)] ) * vertical_scale 
                    subterrain_center_z = get_height_at_xy(
                        self.height_field,
                        subterrain_center_x,
                        subterrain_center_y,
                        horizontal_scale,
                        vertical_scale,
                        self.center_x,
                        self.center_y
                    )
                    
                    print(f"Height at ({subterrain_center_x},{subterrain_center_y}): {subterrain_center_z}")
                    self.subterrain_centers.append((subterrain_center_x, subterrain_center_y, subterrain_center_z))

            # Print the centers
            self.spawn_counter = 0
            self.max_num_centers = len(self.subterrain_centers)
            self.random_pos = self.generate_random_positions()
            self.height_field_tensor = torch.tensor(
                self.height_field, device=self.device, dtype=gs.tc_float
            )
        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_start for name in self.env_cfg["dof_names"]]
        self.hip_dofs = [self.robot.get_joint(name).dof_start for name in self.env_cfg["hip_joint_names"]]
        self.thigh_dofs = [self.robot.get_joint(name).dof_start for name in self.env_cfg["thigh_joint_names"]]
        def find_link_indices(names):
            link_indices = list()
            for link in self.robot.links:
                flag = False
                for name in names:
                    if name in link.name:
                        flag = True
                if flag:
                    link_indices.append(link.idx - self.robot.link_start)
            return link_indices


        self.termination_contact_indices = find_link_indices(
            self.env_cfg['termination_contact_link_names']
        )
        self.penalised_contact_indices = find_link_indices(
            self.env_cfg['penalized_contact_link_names']
        )
        self.calf_indices = find_link_indices(
            self.env_cfg['calf_link_name']
        )

        self.feet_indices = find_link_indices(
            self.env_cfg['feet_link_name']
        )
        self.base_link_index = find_link_indices(
            self.env_cfg['base_link_name']
        )
        self.thigh_indices = find_link_indices(
            self.env_cfg['thigh_link_name']
        )
        def _map_calf_to_foot_indices():
            calf_to_foot = {}
            valid_prefixes = ["FL", "FR", "RL", "RR"]

            calf_names = self.env_cfg['calf_link_name']          # list like ["calf"]
            foot_suffix = self.env_cfg['feet_link_name'][0]      # e.g., "foot"

            for calf_link in self.robot.links:
                if any(name in calf_link.name for name in calf_names):
                    prefix = calf_link.name.split("_")[0]  # e.g., "FL"
                    if prefix not in valid_prefixes:
                        continue

                    target_foot_name = f"{prefix}_{foot_suffix}"
                    for foot_link in self.robot.links:
                        if foot_link.name == target_foot_name:
                            calf_idx = calf_link.idx - self.robot.link_start
                            foot_idx = self.feet_indices[valid_prefixes.index(prefix)]
                            calf_to_foot[calf_idx] = foot_idx

            return calf_to_foot

        self.calf_to_foot_map = _map_calf_to_foot_indices()
        print(f"calf to foot mao {self.calf_to_foot_map}")
        print(f"motor dofs {self.motor_dofs}")
        print(f"feet indicies {self.feet_indices}")
        # PD control
        stiffness = self.env_cfg['PD_stiffness']
        damping = self.env_cfg['PD_damping']
        force_limit = self.env_cfg['force_limit']

        self.p_gains, self.d_gains, self.force_limits = [], [], []
        for dof_name in self.env_cfg['dof_names']:
            for key in stiffness.keys():
                if key in dof_name:
                    self.p_gains.append(stiffness[key])
                    self.d_gains.append(damping[key])
        for dof_name in self.env_cfg['dof_names']:
            for key in force_limit.keys():
                if key in dof_name:
                    self.force_limits.append(force_limit[key])
        self.p_gains = torch.tensor(self.p_gains, device=self.device)
        self.d_gains = torch.tensor(self.d_gains, device=self.device)
        self.batched_p_gains = self.p_gains[None, :].repeat(self.num_envs, 1)
        self.batched_d_gains = self.d_gains[None, :].repeat(self.num_envs, 1)
        self.robot.set_dofs_kp(self.p_gains, self.motor_dofs)
        self.robot.set_dofs_kv(self.d_gains, self.motor_dofs)
        # Set the force range using the calculated force limits
        self.robot.set_dofs_force_range(
            lower=-np.array(self.force_limits),  # Negative lower limit
            upper=np.array(self.force_limits),   # Positive upper limit
            dofs_idx_local=self.motor_dofs
        )
        # Store link indices that trigger termination or penalty
        self.feet_front_indices = self.feet_indices[:2]
        self.feet_rear_indices = self.feet_indices[2:]

        self.termination_exceed_degree_ignored = False
        self.termination_if_roll_greater_than_value = self.env_cfg["termination_if_roll_greater_than"]
        self.termination_if_pitch_greater_than_value = self.env_cfg["termination_if_pitch_greater_than"]
        if self.termination_if_roll_greater_than_value <= 1e-6 or self.termination_if_pitch_greater_than_value <= 1e-6:
            self.termination_exceed_degree_ignored = True

        print(f"termination exceed degree ignored is {self.termination_exceed_degree_ignored}")

        print("Robot links")
        for link in self.robot._links:
            print(link.name)
        
        print(f"termination_contact_indicies {self.termination_contact_indices}")
        print(f"penalised_contact_indices {self.penalised_contact_indices}")
        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
            if name=="termination":
                continue
            self.reward_functions[name] = getattr(self, "_reward_" + name)

        # initialize buffers
        self.init_buffers()
        print("Done initializing")



    def init_buffers(self):
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.zero_obs = torch.zeros(self.num_obs, device=self.device, dtype=gs.tc_float)
        self.zero_privileged_obs = torch.zeros(self.num_privileged_obs, device=self.device, dtype=gs.tc_float)
        self.privileged_obs_buf = torch.zeros((self.num_envs, self.num_privileged_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.time_out_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.out_of_bounds_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.hip_actions = torch.zeros((self.num_envs, len(self.hip_dofs)), device=self.device, dtype=gs.tc_float)
        self.thigh_actions = torch.zeros((self.num_envs, len(self.thigh_dofs)), device=self.device, dtype=gs.tc_float)

        self.feet_air_time = torch.zeros(
            (self.num_envs, len(self.feet_indices)),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.feet_max_height = torch.zeros(
            (self.num_envs, len(self.feet_indices)),
            device=self.device,
            dtype=gs.tc_float,
        )

        self.last_contacts = torch.zeros(
            (self.num_envs, len(self.feet_indices)),
            device=self.device,
            dtype=gs.tc_int,
        )

        self.episode_returns = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.hip_pos = torch.zeros_like(self.hip_actions)
        self.hip_vel = torch.zeros_like(self.hip_actions)
        self.thigh_dof_pos = torch.zeros_like(self.thigh_actions)
        self.thigh_dof_vel = torch.zeros_like(self.thigh_actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.contact_forces = torch.zeros(
            (self.num_envs, self.robot.n_links, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.num_dof = len(self.default_dof_pos )
        self.default_hip_pos = torch.tensor(
            [
                self.env_cfg["default_joint_angles"][name]
                for name in self.env_cfg["dof_names"]
                if name in self.env_cfg["hip_joint_names"]
            ],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.default_thigh_pos = torch.tensor(
            [
                self.env_cfg["default_joint_angles"][name]
                for name in self.env_cfg["dof_names"]
                if name in self.env_cfg["thigh_joint_names"]
            ],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.contact_duration_buf = torch.zeros(
            self.num_envs, 
            dtype=torch.float, 
            device=self.device, 
            requires_grad=False
        )
        self.pitch_exceed_duration_buf = torch.zeros(
            self.num_envs, 
            dtype=torch.float, 
            device=self.device
        )
        self.roll_exceed_duration_buf = torch.zeros(
            self.num_envs, 
            dtype=torch.float, 
            device=self.device
        )
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        # Iterate over the motor DOFs
        # self.soft_dof_vel_limit = self.env_cfg["soft_dof_vel_limit"]
        # ❶  Prefer user-supplied arrays if they exist
        if "dof_lower_limit" in self.env_cfg and "dof_upper_limit" in self.env_cfg:
            lower = torch.tensor(
                self.env_cfg["dof_lower_limit"], device=self.device, dtype=gs.tc_float
            )
            upper = torch.tensor(
                self.env_cfg["dof_upper_limit"], device=self.device, dtype=gs.tc_float
            )

            if lower.shape[0] != len(self.motor_dofs) or upper.shape[0] != len(self.motor_dofs):
                raise ValueError(
                    f"dof_lower/upper_limit lengths ({lower.shape[0]}/{upper.shape[0]}) "
                    f"must match #motor_dofs ({len(self.motor_dofs)})."
                )

            # stack → shape (n_dof, 2)  column-0 = lower, column-1 = upper
            self.dof_pos_limits = torch.stack([lower, upper], dim=1)

        else:
            # ❷  Fallback: read limits from the robot model
            #     (unchanged behaviour)
            self.dof_pos_limits = torch.stack(
                self.robot.get_dofs_limit(self.motor_dofs), dim=1
            )
        # ❸  Torque limits are still read from the model (unchanged)
        self.torque_limits = self.robot.get_dofs_force_range(self.motor_dofs)[1]
        soft_factor = self.reward_cfg["soft_dof_pos_limit"]  # e.g. 0.9  → 90 % range
        centres = self.dof_pos_limits.mean(dim=1)            # (n_dof,)
        ranges  = self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]

        self.dof_pos_limits[:, 0] = centres - 0.5 * ranges * soft_factor
        self.dof_pos_limits[:, 1] = centres + 0.5 * ranges * soft_factor

        self.motor_strengths = gs.ones((self.num_envs, self.num_dof), dtype=float)
        self.motor_offsets = gs.zeros((self.num_envs, self.num_dof), dtype=float)

        self.init_foot()
        self._randomize_controls()
        self._randomize_rigids()
        print(f"Dof_pos_limits{self.dof_pos_limits}")
        print(f"Default dof pos {self.default_dof_pos}")
        print(f"Default hip pos {self.default_hip_pos}")
        self.common_step_counter = 0
        # extras
        self.continuous_push = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.env_identities = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=gs.tc_int, 
        )
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

        self.prev_rel_h = torch.zeros((self.num_envs,), device=self.device)
        # Mapping from actual link index (e.g., 13) → index in feet_pos[:, :, :]
        self.link_idx_to_feet_tensor_idx = {
            link_idx: i for i, link_idx in enumerate(self.feet_indices)
        }

    def get_terrain_height_at(self, x_world, y_world):
        if self.terrain_cfg["terrain_type"] == "plane" or self.terrain_cfg["terrain_type"] == "custom_plane":
            return torch.zeros_like(x_world, device=self.device)
    
        s = self.terrain_cfg["horizontal_scale"]
    
        # Convert scalar inputs to tensors
        x_world = torch.as_tensor(x_world, device=self.device)
        y_world = torch.as_tensor(y_world, device=self.device)
    
        x_shifted = x_world + self.center_x
        y_shifted = y_world + self.center_y
    
        i = (y_shifted / s).long().clamp(0, self.height_field_tensor.shape[0] - 1)
        j = (x_shifted / s).long().clamp(0, self.height_field_tensor.shape[1] - 1)
    
        return self.height_field_tensor[i, j] * self.terrain_cfg["vertical_scale"]

    def get_terrain_height_at_for_base(self, x_world, y_world):
        if self.terrain_cfg["terrain_type"] == "plane" or self.terrain_cfg["terrain_type"] == "custom_plane":
          return torch.tensor(0.0, device=self.device)
        # Create the transform matrix (same as your NumPy one)
        s = self.terrain_cfg["horizontal_scale"]
        mat = torch.tensor([[0.0, 1.0 / s],
                            [1.0 / s, 0.0]], device=self.device)

        # Shift world position by center
        vec = torch.stack([x_world + self.center_x,
                        y_world + self.center_y])

        # Apply transformation
        result = mat @ vec

        i = result[1].long().clamp(0, self.height_field_tensor.shape[0] - 1)
        j = result[0].long().clamp(0, self.height_field_tensor.shape[1] - 1)

        return self.height_field_tensor[i, j] * self.terrain_cfg["vertical_scale"]


    def assign_fixed_commands(self, envs_idx):
        """
        Assign one of four randomly-sampled fixed-direction commands to each
        environment index in `envs_idx`, based on (env_id % 4):
            0 → forward  (+x, random between [0.5, max])
            1 → backward (−x, random between [min, -0.5])
            2 → right    (+y,  random between [0.5, max])
            3 → left     (−y,  random between [min, -0.5])
        """
        envs_idx = torch.as_tensor(envs_idx, device=self.device, dtype=torch.long)
        n_envs = len(envs_idx)
        n_cmds = 4

        cmd_types = (envs_idx % n_cmds).long()
        cmds = torch.zeros((n_envs, 3), device=self.device)

        # Masks for command types
        fwd_mask   = cmd_types == 0
        bwd_mask   = cmd_types == 1
        right_mask = cmd_types == 2
        left_mask  = cmd_types == 3

        # Forward (+x)
        cmds[fwd_mask, 0] = gs_rand_float(0.5, self.command_cfg["lin_vel_x_range"][1], (fwd_mask.sum(),), self.device)

        # Backward (-x)
        cmds[bwd_mask, 0] = gs_rand_float(self.command_cfg["lin_vel_x_range"][0], -0.5, (bwd_mask.sum(),), self.device)

        # Right (+y)
        cmds[right_mask, 1] = gs_rand_float(0.5, self.command_cfg["lin_vel_y_range"][1], (right_mask.sum(),), self.device)

        # Left (-y)
        cmds[left_mask, 1] = gs_rand_float(self.command_cfg["lin_vel_y_range"][0], -0.5, (left_mask.sum(),), self.device)

        # Apply to global buffer
        self.commands[envs_idx] = cmds
        # self._current_command_types[envs_idx] = cmd_types  # Track types only for updated envs

        


    def assign_fixed_commands_max(self, envs_idx):
        """
        Give each env in `envs_idx` one of four max-speed commands, chosen by
        (env_id % 4):

            0 → forward  (+x)
            1 → backward (−x)
            2 → right    (+y)
            3 → left     (−y)
        """
        envs_idx = torch.as_tensor(envs_idx, device=self.device, dtype=torch.long)
        n_cmds   = 4

        # Now the mapping is stable even if we pass a single env idx at a time
        cmd_types = (envs_idx % n_cmds).long()

        # Build the command tensor row-by-row
        cmds = torch.zeros((len(envs_idx), 3), device=self.device)

        forward_mask   = cmd_types == 0
        backward_mask  = cmd_types == 1
        right_mask     = cmd_types == 2
        left_mask      = cmd_types == 3

        # +x  (forward)
        cmds[forward_mask, 0]  = self.command_cfg["lin_vel_x_range"][1]
        # –x  (backward)
        cmds[backward_mask, 0] = self.command_cfg["lin_vel_x_range"][0]
        # +y  (right)
        cmds[right_mask, 1]    = self.command_cfg["lin_vel_y_range"][1]
        # –y  (left)
        cmds[left_mask, 1]     = self.command_cfg["lin_vel_y_range"][0]

        # Write into the global buffer
        self.commands[envs_idx] = cmds


    def _resample_commands_max(self, envs_idx):
        # Sample linear and angular velocities
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)

        # Randomly multiply by -1 or 1 (50/50 chance)
        random_signs = torch.randint(0, 2, self.commands[envs_idx].shape, device=self.device) * 2 - 1
        self.commands[envs_idx] *= random_signs

    def biased_sample(self, min_val, max_val, size, device, bias=2.0):
        """
        Sample values with bias towards positive range.
        The bias parameter skews values towards the upper end.
        """
        uniform_samples = torch.rand(size, device=device)  # [0, 1] uniform
        skewed_samples = uniform_samples ** (1.0 / bias)  # Biasing towards 1
        return min_val + (max_val - min_val) * skewed_samples

    def _resample_commands_without_omega(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)

    def _resample_commands(self, envs_idx):
        if True:
            self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        else:
            self.commands[envs_idx, 0] = self.biased_sample(
                *self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device, bias=2.0
            )
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)

    def _zero_commands(self, envs_idx):
        self.commands[envs_idx, 0] = 0.0
        self.commands[envs_idx, 1] = 0.0
        self.commands[envs_idx, 2] = 0.0


    def generate_subterrain_grid(self, rows, cols, terrain_types, weights):
        """
        Generate a 2D grid (rows x cols) of terrain strings chosen randomly
        based on 'weights', but do NOT place 'pyramid_sloped_terrain' adjacent 
        to another 'pyramid_sloped_terrain'.
        """
        grid = [[None for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                terrain_choice = random.choices(terrain_types, weights=weights, k=1)[0]
                if terrain_choice == "pyramid_sloped_terrain":
                    terrain_choice = random.choice(["pyramid_sloped_terrain", "pyramid_down_sloped_terrain"])
                elif terrain_choice == "pyramid_stairs_terrain":
                    # Define terrain options and their corresponding probabilities
                    terrain_options = ["pyramid_stairs_terrain", "pyramid_down_stairs_terrain"]
                    terrain_weights = [0.0, 1.0]  # climb up priority
                    # Choose terrain based on the weights
                    terrain_choice = random.choices(terrain_options, weights=terrain_weights, k=1)[0]

                grid[i][j] = terrain_choice
        return grid

    def init_foot(self):
        self.feet_num = len(self.feet_indices)
       
        self.step_period = self.reward_cfg["step_period"]
        self.step_offset = self.reward_cfg["step_offset"]
        self.step_height_for_front = self.reward_cfg["front_feet_relative_height"]
        self.step_height_for_rear = self.reward_cfg["rear_feet_relative_height"]
        #todo get he first feet_pos here
        # Get positions for all links and slice using indices
        all_links_pos = self.robot.get_links_pos()
        all_links_vel = self.robot.get_links_vel()

        self.feet_pos = all_links_pos[:, self.feet_indices, :]
        self.thigh_pos =  all_links_pos[:, self.thigh_indices, :]
        self.feet_front_pos = all_links_pos[:, self.feet_front_indices, :]
        self.feet_rear_pos = all_links_pos[:, self.feet_rear_indices, :]
        self.feet_vel = all_links_vel[:, self.feet_indices, :]
        self.calf_pos = all_links_pos[:, self.calf_indices, :]
        self.calf_vel = all_links_vel[:, self.calf_indices, :]
        self.front_feet_pos_base = self._world_to_base_transform(self.feet_front_pos, self.base_pos, self.base_quat)
        self.rear_feet_pos_base = self._world_to_base_transform(self.feet_rear_pos, self.base_pos, self.base_quat)
        self.thigh_pos_base =  self._world_to_base_transform(self.thigh_pos, self.base_pos, self.base_quat)

    def update_feet_state(self):
        # Get positions for all links and slice using indices
        all_links_pos = self.robot.get_links_pos()
        all_links_vel = self.robot.get_links_vel()

        self.feet_pos = all_links_pos[:, self.feet_indices, :]
        self.thigh_pos =  all_links_pos[:, self.thigh_indices, :]
        self.calf_pos = all_links_pos[:, self.calf_indices, :]
        self.calf_vel = all_links_vel[:, self.calf_indices, :]
        self.feet_front_pos = all_links_pos[:, self.feet_front_indices, :]
        self.feet_rear_pos = all_links_pos[:, self.feet_rear_indices, :]
        self.feet_vel = all_links_vel[:, self.feet_indices, :]
        self.front_feet_pos_base = self._world_to_base_transform(self.feet_front_pos, self.base_pos, self.base_quat)
        self.rear_feet_pos_base = self._world_to_base_transform(self.feet_rear_pos, self.base_pos, self.base_quat)
        self.thigh_pos_base =  self._world_to_base_transform(self.thigh_pos, self.base_pos, self.base_quat)


    def _quaternion_to_matrix(self, quat):
        w, x, y, z = quat.unbind(dim=-1)
        R = torch.stack([
            1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)
        ], dim=-1).reshape(-1, 3, 3)
        return R

    def _world_to_base_transform(self, points_world, base_pos, base_quat):
        # Convert quaternion to rotation matrix
        R = self._quaternion_to_matrix(base_quat)

        # Subtract base position to get relative position
        points_relative = points_world - base_pos.unsqueeze(1)

        # Apply rotation to transform to base frame
        points_base = torch.einsum('bij,bkj->bki', R.transpose(1, 2), points_relative)
        return points_base


    def post_physics_step_callback(self):
        self.update_feet_state()
        self.phase = (self.episode_length_buf * self.dt) % self.step_period / self.step_period
        # Assign phases for quadruped legs
        """
        small_offset = 0.05  # tweak as needed, 0 < small_offset < step_offset typically
        self.phase_FL_RR = self.phase
        self.phase_FR_RL = (self.phase + self.step_offset) % 1

        # Now offset one leg in each diagonal pair slightly
        phase_FL = self.phase_FL_RR
        phase_RR = (self.phase_FL_RR + small_offset) % 1     # shifted by small_offset

        phase_FR = self.phase_FR_RL
        phase_RL = (self.phase_FR_RL + small_offset) % 1     # shifted by small_offset

        # Concatenate in the order (FL, FR, RL, RR)
        self.leg_phase = torch.cat([
            phase_FL.unsqueeze(1),
            phase_FR.unsqueeze(1),
            phase_RL.unsqueeze(1),
            phase_RR.unsqueeze(1)
        ], dim=-1)
        """
        if self.show_vis:
            self._draw_debug_vis()
        # Assign phases for quadruped legs
        self.phase_FL_RR = self.phase  # Front-left (FL) and Rear-right (RR) in sync
        self.phase_FR_RL = (self.phase + self.step_offset) % 1  # Front-right (FR) and Rear-left (RL) offset

        # Assign phases to legs based on their indices (FL, FR, RL, RR) order matters
        self.leg_phase = torch.cat([
            self.phase_FL_RR.unsqueeze(1),  # FL
            self.phase_FR_RL.unsqueeze(1),  # FR
            self.phase_FR_RL.unsqueeze(1),  # RL
            self.phase_FL_RR.unsqueeze(1)   # RR
        ], dim=-1)
        


    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        if self.env_cfg["randomize_delay"]:
            # 3️⃣ Store new actions in delay buffer (Shift the buffer)
            self.action_delay_buffer[:, :, :-1] = self.action_delay_buffer[:, :, 1:].clone()
            self.action_delay_buffer[:, :, -1] = self.actions  # Insert latest action

            # 3) Vectorized gather for delayed actions

            T = self.action_delay_buffer.shape[-1]  # T = max_delay_steps + 1
            # (num_envs, num_actions)
            delayed_indices = (T - 1) - self.motor_delay_steps
            # Expand to (num_envs, num_actions, 1)
            gather_indices = delayed_indices.unsqueeze(-1)

            # Gather from last dimension
            delayed_actions = self.action_delay_buffer.gather(dim=2, index=gather_indices).squeeze(-1)

            exec_actions = delayed_actions
        else:
            exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        dof_pos_list = []
        dof_vel_list = []
        for i in range(self.env_cfg['decimation']):
            self.torques = self._compute_torques(exec_actions)
            if self.num_envs == 0:
                torques = self.torques.squeeze()
                self.robot.control_dofs_force(torques, self.motor_dofs)
            else:
                self.robot.control_dofs_force(self.torques, self.motor_dofs)
            self.scene.step()
            self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
            self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)




        

        pos_after_step = self.robot.get_pos()
        quat_after_step = self.robot.get_quat()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.hip_pos[:] = self.robot.get_dofs_position(self.hip_dofs)
        self.hip_vel[:] = self.robot.get_dofs_velocity(self.hip_dofs)
        self.thigh_dof_pos[:] = self.robot.get_dofs_position(self.thigh_dofs)
        self.thigh_dof_vel[:] = self.robot.get_dofs_velocity(self.thigh_dofs)
        self.contact_forces[:] = torch.tensor(
            self.robot.get_links_net_contact_force(),
            device=self.device,
            dtype=gs.tc_float,
        )        
        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )

        #resample commands here
        self.curriculum_step += 1
        if self.command_curriculum and not self.curriculum_complete_flag:
            if self.mean_reward_flag:
                self.curriculum_complete_flag = True
                print("Curriculum is finished")
            else:
                self.assign_fixed_commands(envs_idx)
        # elif self.command_curriculum:
        #     self._resample_commands(envs_idx)
        else:
            self._resample_commands(envs_idx)


        self.post_physics_step_callback()
        
        # random push
        self.common_step_counter += 1
        push_interval_s = self.env_cfg['push_interval_s']
        if push_interval_s > 0:
            max_push_vel_xy = self.env_cfg['max_push_vel_xy']
            dofs_vel = self.robot.get_dofs_velocity() # (num_envs, num_dof) [0:3] ~ base_link_vel
            push_vel = gs_rand_float(-max_push_vel_xy, max_push_vel_xy, (self.num_envs, 2), self.device)
            push_vel[((self.common_step_counter + self.env_identities) % int(push_interval_s / self.dt) != 0)] = 0
            dofs_vel[:, :2] += push_vel
            self.robot.set_dofs_velocity(dofs_vel)



        self.check_termination()
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.compute_rewards()

        self.compute_observations()

        self._render_headless()
        self.extras["observations"]["critic"] = self.privileged_obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras


    def compute_observations(self):
        sin_phase = torch.sin(2 * np.pi * self.leg_phase)  # Shape: (batch_size, 4)
        cos_phase = torch.cos(2 * np.pi * self.leg_phase)  # Shape: (batch_size, 4)

        # Prepare all components
        base_lin_vel = self.base_lin_vel * self.obs_scales["lin_vel"]
        base_ang_vel = self.base_ang_vel * self.obs_scales["ang_vel"]
        projected_gravity = self.projected_gravity
        commands = self.commands * self.commands_scale
        dof_pos_scaled = (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"]
        dof_vel_scaled = self.dof_vel * self.obs_scales["dof_vel"]
        torques_scaled = self.torques * self.obs_scales["torques"]
        actions = self.actions
        # ─── post_physics_step_callback() ────────────────────────────────
        swing_mask = (self.leg_phase > 0.55)                 # (N,4)
        foot_force = torch.norm(self.contact_forces[:,       # (N,4)
                                self.feet_indices, :3], dim=2)

        collision  = (swing_mask & (foot_force > 1.0)).float()   # (N,4)
        # Debug checks
        debug_items = {
            "base_ang_vel": base_ang_vel,
            "projected_gravity": projected_gravity,
            "commands": commands,
            "dof_pos_scaled": dof_pos_scaled,
            "dof_vel_scaled": dof_vel_scaled,
            "actions": actions,
        }

        # for name, tensor in debug_items.items():
        #     if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        #         print(f">>> WARNING: NaN or Inf in {name} <<<")
        #         print(tensor)

        # compute observations
        self.obs_buf = torch.cat(
            [
                base_ang_vel,        # 3
                projected_gravity,   # 3
                commands,            # 3
                dof_pos_scaled,      # 12
                dof_vel_scaled,      # 12
                actions              # 12
            ],
            axis=-1,
        )

        self.privileged_obs_buf = torch.cat(
            [
                base_lin_vel,        # 3
                base_ang_vel,        # 3
                projected_gravity,   # 3
                commands,            # 3
                dof_pos_scaled,      # 12
                dof_vel_scaled,      # 12
                torques_scaled,      # 12
                actions,              # 12
                # sin_phase,        # 4
                # cos_phase,        # 4
                # collision         # 4
            ],
            axis=-1,
        )

        self.obs_buf = torch.clip(self.obs_buf, -self.clip_obs, self.clip_obs)
        self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -self.clip_obs, self.clip_obs)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        # Final check
        if torch.isnan(self.obs_buf).any() or torch.isinf(self.obs_buf).any():
            print(">>> WARNING: NaN or Inf in final obs_buf <<<")
            print(self.obs_buf)



    def compute_rewards(self):
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.reward_cfg["only_positive_rewards"]:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
        self.episode_returns += self.rew_buf     # add current step reward


    def get_observations(self):
        self.extras["observations"]["critic"] = self.privileged_obs_buf

        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return self.privileged_obs_buf



    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.noise_cfg["add_noise"]
        noise_level =self.noise_cfg["noise_level"]
        # noise_vec[:3] = self.noise_scales["lin_vel"] * noise_level * self.obs_scales["lin_vel"]
        # noise_vec[3:6] = self.noise_scales["ang_vel"] * noise_level * self.obs_scales["ang_vel"]
        # noise_vec[6:9] = self.noise_scales["gravity"] * noise_level
        # noise_vec[9:12] = 0. # commands
        # noise_vec[12:12+self.num_actions] = self.noise_scales["dof_pos"] * noise_level * self.obs_scales["dof_pos"]
        # noise_vec[12+self.num_actions:12+2*self.num_actions] = self.noise_scales["dof_vel"] * noise_level * self.obs_scales["dof_vel"]
        # noise_vec[12+3*self.num_actions:12+4*self.num_actions] = 0. # previous actions
        # noise_vec[12+4*self.num_actions:12+4*self.num_actions+8] = 0. # sin/cos phase

        noise_vec[:3] = self.noise_scales["ang_vel"] * noise_level * self.obs_scales["ang_vel"]
        noise_vec[3:6] = self.noise_scales["gravity"] * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = self.noise_scales["dof_pos"] * noise_level * self.obs_scales["dof_pos"]
        noise_vec[9+self.num_actions:9+2*self.num_actions] = self.noise_scales["dof_vel"] * noise_level * self.obs_scales["dof_vel"]
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions

        return noise_vec


    def check_termination(self):
        """Check if environments need to be reset."""
        # (n_envs, n_links, 3) tensor of net contact forces
        contact_threshold_exceeded = (torch.norm(
            self.contact_forces[:, self.termination_contact_indices, :], dim=-1
        ) > 1.0)
        # For each environment, if ANY contact index exceeds force threshold, treat it as contact
        in_contact = torch.any(contact_threshold_exceeded, dim=1)
        self.contact_duration_buf[in_contact] += self.dt
        self.reset_buf = self.contact_duration_buf > self.env_cfg["termination_duration"]
        #pitch and roll degree exceed termination
        if not self.termination_exceed_degree_ignored:
            # Check where pitch and roll exceed thresholds
            pitch_exceeded = torch.abs(self.base_euler[:, 1]) > math.radians(self.termination_if_pitch_greater_than_value)
            roll_exceeded = torch.abs(self.base_euler[:, 0]) > math.radians(self.termination_if_roll_greater_than_value)

            # Increment duration where exceeded
            self.pitch_exceed_duration_buf[pitch_exceeded] += self.dt
            self.roll_exceed_duration_buf[roll_exceeded] += self.dt

            # Reset duration where NOT exceeded
            self.pitch_exceed_duration_buf[~pitch_exceeded] = 0.0
            self.roll_exceed_duration_buf[~roll_exceeded] = 0.0

            # Trigger reset if exceed duration > threshold (e.g., 3 seconds)
            pitch_timeout = self.pitch_exceed_duration_buf > self.env_cfg["angle_termination_duration"]
            roll_timeout = self.roll_exceed_duration_buf > self.env_cfg["angle_termination_duration"]

            self.reset_buf |= pitch_timeout
            self.reset_buf |= roll_timeout
        # Timeout termination
        if self.command_curriculum and not self.curriculum_complete_flag:
            yaw_limit = 60
            exceed_yaw = torch.abs(self.base_euler[:, 2]) > math.radians(yaw_limit)
            self.reset_buf |= exceed_yaw


        # # shape (num_envs, num_dof) → Bool where True = violation
        # out_of_limits = (self.dof_pos < self.dof_pos_limits[:, 0]) | \
        #                 (self.dof_pos > self.dof_pos_limits[:, 1])

        # # any() along the dof dimension ⇒ shape (num_envs,)
        # joint_violation = out_of_limits.any(dim=1)

        # self.reset_buf |= joint_violation        # mark those envs for reset

        self.reset_buf |= self.base_pos[:, 2] < self.env_cfg['termination_if_height_lower_than']
        # -------------------------------------------------------
        #  Add out-of-bounds check using terrain_min_x, etc.
        # -------------------------------------------------------
        # min_x, max_x, min_y, max_y = self.terrain_bounds  # or however you store them
        
        # We assume base_pos[:, 0] is x, base_pos[:, 1] is y
        if self.terrain_type != "plane" and self.terrain_type != "custom_plane":
            self.out_of_bounds_buf = (
                (self.base_pos[:, 0] < self.terrain_min_x) |
                (self.base_pos[:, 0] > self.terrain_max_x) |
                (self.base_pos[:, 1] < self.terrain_min_y) |
                (self.base_pos[:, 1] > self.terrain_max_y)
            )
            self.reset_buf |= self.out_of_bounds_buf
        # For those that are out of bounds, penalize by marking episode_length_buf = max
        # self.episode_length_buf[out_of_bounds] = self.max_episode_length

        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        # indices that are **not** being reset
        all_idx     = torch.arange(self.num_envs, device=self.device)
        keep_mask   = ~torch.isin(all_idx, torch.as_tensor(envs_idx, device=self.device))
        active_idx  = all_idx[keep_mask]

        if active_idx.numel() > 0:                                 # at least one env left
            mean_reward = torch.mean(self.episode_returns[active_idx]).item()
        else:                                                      # everything is being reset
            mean_reward = 0.0
        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        # self.hip_pos[envs_idx] = self.default_hip_pos
        # self.hip_vel[envs_idx] = 0.0
        # self.thigh_dof_pos[envs_idx] = self.default_thigh_pos
        # self.thigh_dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        # reset base
        # Check if the new_base_pos contains any NaNs
        # Randomly choose positions from pre-generated random_pos for each environment
        random_indices = torch.randint(0, self.num_envs, (len(envs_idx),), device=self.device)
        self.base_pos[envs_idx] = self.random_pos[random_indices] + self.base_init_pos
            

        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        # if 0 in env_idx:
        if not self.mean_reward_flag:
            self.mean_reward_flag = mean_reward > self.mean_reward_threshold
        if self.command_curriculum and not self.curriculum_complete_flag:
            self.curriculum_complete_flag =  self.curriculum_step*self.dt > self.curriculum_duration
        else:
            self.curriculum_complete_flag = True

        if self.env_cfg["randomize_rot"] and ((self.mean_reward_flag and self.curriculum_complete_flag) or self.eval) :
            # 1) Get random roll, pitch, yaw (in degrees) for each environment.
            
            roll = gs_rand_float(*self.env_cfg["roll_range"],  (len(envs_idx),), self.device)
            pitch = gs_rand_float(*self.env_cfg["pitch_range"], (len(envs_idx),), self.device)
            yaw = gs_rand_float(*self.env_cfg["yaw_range"],    (len(envs_idx),), self.device)

            # 2) Convert them all at once into a (N,4) quaternion tensor [x, y, z, w].
            quats_torch = quaternion_from_euler_tensor(roll, pitch, yaw)  # (N, 4)

            # 3) Move to CPU if needed and assign into self.base_quat in one shot
            #    (assuming self.base_quat is a numpy array of shape [num_envs, 4]).
            self.base_quat[envs_idx] = quats_torch
        else:
            self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)

        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)

        if self.env_cfg["randomize_delay"]:
            self.motor_delay_steps[envs_idx] = torch.randint(
                int(self.min_delay / self.dt),
                self.max_delay_steps + 1,
                (len(envs_idx), self.num_actions),
                device=self.device
            )

        # 1b. Check DOFs
        dof_pos = self.robot.get_dofs_position(self.motor_dofs)

        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = -(2/ self.dt)
        self.feet_air_time[envs_idx] = 0.0
        self.feet_max_height[envs_idx] = 0.0
        self.reset_buf[envs_idx] = True
        self.contact_duration_buf[envs_idx] = 0.0
        self.episode_returns[envs_idx]  = 0.0
        # fill extras
        self._randomize_rigids(envs_idx)
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._zero_commands(envs_idx)
        if self.env_cfg['send_timeouts']:
            self.extras['time_outs'] = self.time_out_buf

    def generate_random_positions(self):
        """
        Use the _random_robot_position() method to generate unique random positions
        for each environment.
        """
        positions = torch.zeros((self.num_envs, 3), device=self.device)
        for i in range(self.num_envs):
            x, y, z = self._random_robot_position()
            # positions[i] = torch.tensor([0, 0, z], device=self.device)
            positions[i] = torch.tensor([x, y, z], device=self.device)
        return positions

    def generate_positions(self):
        """
        Use the _random_robot_position() method to generate unique random positions
        for each environment.
        """
        positions = torch.zeros((self.num_envs, 3), device=self.device)
        for i in range(self.num_envs):
            positions[i] = torch.tensor([0, 0, 0], device=self.device)
        return positions

    def _random_robot_position(self):
        # 1. Sample random row, col(a subterrain)
        # 0.775 ~ l2_norm(0.7, 0.31)
        # go2_size_xy = 0.775
        # row = np.random.randint(int((self.rows * self.terrain.subterrain_size[0]-go2_size_xy)/self.terrain.horizontal_scale))
        # col = np.random.randint(int((self.cols * self.terrain.subterrain_size[1]-go2_size_xy)/self.terrain.horizontal_scale))
        center = self.subterrain_centers[self.spawn_counter]
        x, y, z = center[0], center[1], center[2]
        self.spawn_counter+= 1
        if self.spawn_counter == len(self.subterrain_centers):
            self.spawn_counter = 0
       
        return x, y, z


    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        
        actions_scaled = actions * self.env_cfg['action_scale']
        torques = (
            self.batched_p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos + self.motor_offsets)
            - self.batched_d_gains * self.dof_vel
        )
        torques =  torques * self.motor_strengths

            # torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel

        return torch.clip(torques, -self.torque_limits, self.torque_limits)


    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, self.privileged_obs_buf


    def _render_headless(self):
        if self._recording and len(self._recorded_frames) < 150:
            x, y, z = self.base_pos[0].cpu().numpy()  # Convert the tensor to NumPy
            self.cam_0.set_pose(pos=(x+5.0, y, z+5.5), lookat=(x, y, z+0.5))
            if self.show_vis:
                self.cam_0.render(
                    rgb=True,
                )
            frame, _, _, _ = self.cam_0.render()
            self._recorded_frames.append(frame)
        elif self.show_vis:
            x, y, z = self.base_pos[0].cpu().numpy()  # Convert the tensor to NumPy
            self.cam_0.set_pose(pos=(x+5.0, y, z+5.5), lookat=(x, y, z+0.5))
            self.cam_0.render(
                rgb=True,
            )
    def get_recorded_frames(self):
        if len(self._recorded_frames) >=10:
            frames = self._recorded_frames
            self._recorded_frames = []
            self._recording = False
            return frames
        else:
            return None

    def start_recording(self, record_internal=True):
        self._recorded_frames = []
        self._recording = True
        if record_internal:
            self._record_frames = True
        else:
            self.cam_0.start_recording()

    def stop_recording(self, save_path=None):
        self._recorded_frames = []
        self._recording = False
        if save_path is not None:
            print("fps", int(1 / self.dt))
            self.cam_0.stop_recording(save_path, fps = int(1 / self.dt))

    # ------------ domain randomization----------------

    def _randomize_rigids(self, env_ids=None):


        if env_ids == None:
            env_ids = torch.arange(0, self.num_envs)
        elif len(env_ids) == 0:
            return

        if self.env_cfg['randomize_friction']:
            self._randomize_link_friction(env_ids)
        if self.env_cfg['randomize_base_mass']:
            self._randomize_base_mass(env_ids)
        if self.env_cfg['randomize_com_displacement']:
            self._randomize_com_displacement(env_ids)

    def _randomize_controls(self, env_ids=None):

        if env_ids == None:
            env_ids = torch.arange(0, self.num_envs)
        elif len(env_ids) == 0:
            return

        if self.env_cfg['randomize_motor_strength']:
            self._randomize_motor_strength(env_ids)
        if self.env_cfg['randomize_motor_offset']:
            self._randomize_motor_offset(env_ids)
        if self.env_cfg['randomize_kp_scale']:
            self._randomize_kp(env_ids)
        if self.env_cfg['randomize_kd_scale']:
            self._randomize_kd(env_ids)

    def _randomize_link_friction(self, env_ids):
        min_friction, max_friction = self.env_cfg['friction_range']

        # Generate random friction values only for the selected environments
        random_friction = min_friction + (max_friction - min_friction) * torch.rand(len(env_ids), self.robot.n_links)

        # Apply friction to the specified environments
        self.robot.set_friction_ratio(
            friction_ratio=random_friction,
            links_idx_local=np.arange(0, self.robot.n_links),
            envs_idx=env_ids,  # Apply only to selected environments
        )
        

    def _randomize_base_mass(self, env_ids):

        min_mass, max_mass = self.env_cfg['added_mass_range']
        mass_shift = min_mass + (max_mass - min_mass) * torch.rand(len(env_ids), 1, device=self.device)
        ls_idx_local = torch.tensor(self.base_link_index, device=self.device, dtype=torch.int32)

        # Apply only to base_link in the selected environments
        self.robot.set_mass_shift(
            mass_shift=mass_shift,
            links_idx_local=ls_idx_local,  
            envs_idx=env_ids  
        )


    def _randomize_com_displacement(self, env_ids):

        min_displacement, max_displacement = self.env_cfg['com_displacement_range']
        com_shift = min_displacement + (max_displacement - min_displacement) * torch.rand(len(env_ids), 1, 3, device=self.device)
        ls_idx_local = torch.tensor(self.base_link_index, device=self.device, dtype=torch.int32)


        # Apply only to base_link in the selected environments
        self.robot.set_COM_shift(
            com_shift=com_shift,
            links_idx_local=ls_idx_local,  # Wrap it in a list
            envs_idx=env_ids  # Apply only to specific environments
        )


    def _randomize_motor_strength(self, env_ids):

        min_strength, max_strength = self.env_cfg['motor_strength_range']
        self.motor_strengths[env_ids, :] = gs.rand((len(env_ids), 1), dtype=float) \
                                           * (max_strength - min_strength) + min_strength

    def _randomize_motor_offset(self, env_ids):

        min_offset, max_offset = self.env_cfg['motor_offset_range']
        self.motor_offsets[env_ids, :] = gs.rand((len(env_ids), self.num_dof), dtype=float) \
                                         * (max_offset - min_offset) + min_offset

    def _randomize_kp(self, env_ids):

        min_scale, max_scale = self.env_cfg['kp_scale_range']
        kp_scales = gs.rand((len(env_ids), self.num_dof), dtype=float) \
                    * (max_scale - min_scale) + min_scale
        self.batched_p_gains[env_ids, :] = kp_scales * self.p_gains[None, :]

    def _randomize_kd(self, env_ids):

        min_scale, max_scale = self.env_cfg['kd_scale_range']
        kd_scales = gs.rand((len(env_ids), self.num_dof), dtype=float) \
                    * (max_scale - min_scale) + min_scale
        self.batched_d_gains[env_ids, :] = kd_scales * self.d_gains[None, :]


    def _draw_debug_vis(self):
        self.scene.clear_debug_objects()

        VEL_LENGTH_SCALE = 0.3
        VEL_RADIUS       = 0.05

        # Draw for every environment that is being shown in the viewer
        for env_idx in self.rendered_envs_idx:
            # ─── origin slightly above the robot base ─────────────────────
            origin = self.base_pos[env_idx].clone().cpu()
            origin[2] += 0.2

            # ─── BLUE arrow: current base-frame linear velocity ───────────
            vel_body  = self.base_lin_vel[env_idx].unsqueeze(0)                   # (1,3)
            vel_world = transform_by_quat(
                            vel_body,
                            self.base_quat[env_idx].unsqueeze(0)
                        )[0].cpu()
            self.scene.draw_debug_arrow(
                pos   = origin,
                vec   = vel_world * VEL_LENGTH_SCALE,
                radius= VEL_RADIUS,
                color = (0.0, 0.0, 1.0, 0.8)
            )

            # ─── GREEN arrow: commanded velocity (rotated to world frame) ─
            cmd_body  = torch.tensor(
                            [*self.commands[env_idx, :2], 0.0],
                            device=self.device, dtype=gs.tc_float
                        ).unsqueeze(0)
            cmd_world = transform_by_quat(
                            cmd_body,
                            self.base_quat[env_idx].unsqueeze(0)
                        )[0].cpu()
            self.scene.draw_debug_arrow(
                pos   = origin,
                vec   = cmd_world * VEL_LENGTH_SCALE,
                radius= VEL_RADIUS,
                color = (0.0, 1.0, 0.0, 0.8)
            )



    # ------------ reward functions----------------

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])


    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_roll_penalty(self):
        # Penalize large roll (base_euler[:, 0] is roll in radians)
        return torch.square(self.base_euler[:, 0])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_relative_base_height(self):
        # --- 現在の相対高さ --------------------------------------------------
        terrain_h = self.get_terrain_height_at_for_base(self.base_pos[:, 0], self.base_pos[:, 1])
        rel_h     = self.base_pos[:, 2] - terrain_h          # shape = (N,)

        # --- 目標との差分 (低すぎるときだけ) ---------------------------------
        target  = self.reward_cfg["relative_base_height_target"]
        penalty = torch.square(torch.relu(target - rel_h))   # shape = (N,)

        # --- ピッチ角が ±10° 以内の環境だけ有効にする -------------------------
        # base_euler[:, 1] は pitch (deg) で格納されている前提
        pitch_ok = (torch.abs(self.base_euler[:, 1]) < 5.0) # Bool tensor shape = (N,)
        penalty  = penalty * pitch_ok.float()                # True →そのまま, False→0
        return penalty

    def _reward_collision(self):
        """
        Penalize collisions on selected bodies.
        Returns the per-env penalty value as a 1D tensor of shape (n_envs,).
        """
        undesired_forces = torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1)
        collisions = (undesired_forces > 0.1).float()  # shape (n_envs, len(...))
        
        # Sum over those links to get # of collisions per environment
        return collisions.sum(dim=1)

    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))

    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # Iterate over legs (order: FL, FR, RL, RR)
        for i in range(self.feet_num):
            # Determine if the current phase indicates a stance phase (< 0.55)
            is_stance = self.leg_phase[:, i] < 0.55

            # Check if the foot is in contact with the ground
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1

            # Reward correct contact behavior (stance matches contact)
            res += ~(contact ^ is_stance)  # XOR for mismatch, negate for correct match

        return res

    def _reward_front_thigh_forward_limit(self):
        """
        Penalise FL and FR thigh joints if they swing too far forward.
        • Angles < THRESH (default −0.4 rad) are considered “excess”.
        • Penalty = (excess)^2  summed over the two joints.
        """
        THRESH = -0.3                                               # radians

        # ‣ self.thigh_dof_pos   shape: (N_envs, 4)  order = FR, FL, RL, RR
        front_angles = self.thigh_dof_pos[:, :2]                    # FR & FL

        # positive value only when angle < THRESH
        excess = torch.relu(THRESH - front_angles)                  # (N, 2)
        penalty = torch.sum(excess**2, dim=1)                       # (N,)

        return penalty                                              # higher ⇒ worse

    def _reward_hip_pos(self):
        return torch.sum(torch.abs(self.hip_pos- self.default_hip_pos), dim=(1))



    def _reward_front_feet_swing_height(self):
        # 地面との接触判定（front feet）
        contact = torch.norm(self.contact_forces[:, self.feet_front_indices, :3], dim=2) > 1.0
        
        # 前足の位置 (z成分)
        z = self.feet_front_pos[:, :, 2]

        # 地形の高さ
        terrain_h = self.get_terrain_height_at(
            self.feet_front_pos[:, :, 0],
            self.feet_front_pos[:, :, 1]
        )

        # 高さ差（地面との相対高度）
        rel_h = z - terrain_h  # [num_envs, 2]

        # 前足の速度（x方向）を feet_vel から抽出
        front_feet_vel = self.feet_vel[:, :2, :]          # shape (N, 2, 3)
        foot_vel_x     = front_feet_vel[:, :, 0]          # x‑velocity
        # 条件：接地していないかつ前進中
        swing_forward = (~contact) & (foot_vel_x > 0.05)

        # 高さが足りない場合にペナルティ
        height_error = torch.relu(self.step_height_for_front - rel_h)
        pos_error = height_error * swing_forward
        # 報酬（低すぎるスイング足をペナルティ）
        reward = torch.sum(pos_error, dim=1)
        return reward


    def _reward_front_feet_clearance(self):
      """
      Encourage front feet to swing backward and lift off the terrain.
      - Reward grows quadratically with height (above 5 cm).
      - Reward is capped at 0.0225 once the height exceeds 20 cm.
      """
      # 1) Contact state
      contact = torch.norm(
          self.contact_forces[:, self.feet_front_indices, :3], dim=2
      ) > 1.0  # shape: (N, 2)
  
      # 2) Clearance above terrain
      z = self.feet_front_pos[:, :, 2]
      terrain_h = self.get_terrain_height_at(
          self.feet_front_pos[:, :, 0],
          self.feet_front_pos[:, :, 1]
      )
      rel_h = z - terrain_h  # shape: (N, 2)
  
      #   # 3) Check if rear feet are swinging (not in contact and vx nonzero)
      #   foot_vel_x = self.feet_vel[:, :2, 0]  # front feet vx (feet 2 and 3)
      #   swing_fwd = (~contact) & (foot_vel_x > 0.05)  # (N, 2)
      # 3) Check if rear feet are swinging (not in contact and vx nonzero)
      foot_vel_x = self.feet_vel[:, :2, 0]  # front feet vx (feet 2 and 3)
      swing_fwd = (~contact) & (foot_vel_x > 0.05)  # (N, 2)
  


      # 4) Quadratic reward logic
      clearance_start = 0.01         # reward begins above 5 cm
      max_reward_height = 0.10       # cap reward at 20 cm
      max_bonus = (max_reward_height - clearance_start) ** 2  # = 0.0225
  
      above_target = torch.clamp(rel_h - clearance_start, min=0.0)  # (N, 2)
      bonus = above_target.pow(2)
      bonus = torch.clamp(bonus, max=max_bonus)  # limit growth
  
      # Apply only during swing
      bonus = bonus * swing_fwd
  
      # Sum across both rear feet
      reward = torch.sum(bonus, dim=1)  # (N,)
  
      # Debug for env 1
      # idx = 1
      # if idx < reward.shape[0]:
      #     print("\n[DEBUG] Rear Foot Clearance (env 1)")
      #     print("Contact:", contact[idx].cpu().numpy())
      #     print("rel_h:", rel_h[idx].cpu().numpy())
      #     print("vx:", foot_vel_x[idx].cpu().numpy())
      #     print("swing_fwd:", swing_fwd[idx].cpu().numpy())
      #     print("bonus (per foot):", bonus[idx].cpu().numpy())
      #     print("Final reward:", reward[idx].item())
  
      return reward

    def _reward_rear_feet_clearance(self):
      """
      Encourage rear feet to swing backward and lift off the terrain.
      - Reward grows quadratically with height (above 5 cm).
      - Reward is capped at 0.0225 once the height exceeds 20 cm.
      """
      # 1) Contact state
      contact = torch.norm(
          self.contact_forces[:, self.feet_rear_indices, :3], dim=2
      ) > 1.0  # shape: (N, 2)
  
      # 2) Clearance above terrain
      z = self.feet_rear_pos[:, :, 2]
      terrain_h = self.get_terrain_height_at(
          self.feet_rear_pos[:, :, 0],
          self.feet_rear_pos[:, :, 1]
      )
      rel_h = z - terrain_h  # shape: (N, 2)
  
      
      # 3) Check if rear feet are swinging (not in contact and vx nonzero)
      foot_vel_x = self.feet_vel[:, 2:, 0]  # front feet vx (feet 2 and 3)
      swing_bwd = (~contact) & (foot_vel_x > 0.05)  # (N, 2)
  
      # 4) Quadratic reward logic
      clearance_start = 0.01         # reward begins above 5 cm
      max_reward_height = 0.10       # cap reward at 15 cm
      max_bonus = (max_reward_height - clearance_start) ** 2  # = 0.0225
  
      above_target = torch.clamp(rel_h - clearance_start, min=0.0)  # (N, 2)
      bonus = above_target.pow(2)
      bonus = torch.clamp(bonus, max=max_bonus)  # limit growth
  
      # Apply only during swing
      bonus = bonus * swing_bwd
  
      # Sum across both rear feet
      reward = torch.sum(bonus, dim=1)  # (N,)
  
      # Debug for env 1
      # idx = 1
      # if idx < reward.shape[0]:
      #     print("\n[DEBUG] Rear Foot Clearance (env 1)")
      #     print("Contact:", contact[idx].cpu().numpy())
      #     print("rel_h:", rel_h[idx].cpu().numpy())
      #     print("vx:", foot_vel_x[idx].cpu().numpy())
      #     print("swing_bwd:", swing_bwd[idx].cpu().numpy())
      #     print("bonus (per foot):", bonus[idx].cpu().numpy())
      #     print("Final reward:", reward[idx].item())
  
      return reward

    def _reward_calf_clearance(self):
        """
        Foot-style clearance reward for all calves (4×):
            • target height is measured in the body frame
            • reward = (height-error)^2  ×  lateral speed of the calf tip
        """
        # --- World-frame position & velocity of each calf ----------------
        calf_world_pos = self.calf_pos                # (N, 4, 3)
        calf_world_vel = self.calf_vel                # (N, 4, 3)

        # ---  Convert to body frame --------------------------------------
        pos_body = self._world_to_base_transform(
            calf_world_pos, self.base_pos, self.base_quat)      # (N,4,3)

        vel_body = self._world_to_base_transform(
            calf_world_vel, torch.zeros_like(self.base_pos), self.base_quat)

        # --- Height error (z) -------------------------------------------
        target_h = self.reward_cfg.get("calf_clearance_height_target")
        height_err = torch.square(pos_body[:, :, 2] - target_h)           # (N,4)

        # --- Lateral velocity magnitude (x-y plane) ----------------------
        lateral_vel = torch.norm(vel_body[:, :, :2], dim=2)               # (N,4)

        # --- Final reward ------------------------------------------------
        clearance_reward = height_err * lateral_vel                       # (N,4)
        return clearance_reward.sum(dim=1)                                # (N,)


    def _reward_rear_feet_swing_height(self):
        # Determine which rear feet are in contact
        contact = torch.norm(self.contact_forces[:, self.feet_rear_indices, :3], dim=2) > 1.0
    
        # Extract x, y, z from rear foot positions (shape: [num_envs, 2])
        x = self.feet_rear_pos[:, :, 0]
        y = self.feet_rear_pos[:, :, 1]
        z = self.feet_rear_pos[:, :, 2]
    
        # Terrain height under each foot (shape: [num_envs, 2])
        terrain_h = self.get_terrain_height_at(x, y)
    
        # Relative height above terrain
        rel_h = z - terrain_h
    
        # Penalize only if the swing foot is *below* the desired height
        height_error = torch.relu(self.step_height_for_rear - rel_h)  # only penalize if below
        pos_error = height_error * ~contact  # only apply to swing feet
    
        reward = torch.sum(pos_error, dim=1)  # sum over the two rear feet
        return reward

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)


    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    # def _reward_dof_vel_limits(self):
    #     # Penalize dof velocities too close to the limit
    #     # clip to max error = 1 rad/s per joint to avoid huge penalties
    #     return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)


    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf & ~self.time_out_buf & ~self.out_of_bounds_buf

    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.reward_cfg["max_contact_force"]).clip(min=0.), dim=1)


    def _reward_base_upward_progress(self):
        """ 階段登攀時のベース上昇に対する報酬（連続的に上に移動する場合に与える） """
        terrain_h = self.get_terrain_height_at_for_base(self.base_pos[:, 0], self.base_pos[:, 1])
        rel_h = self.base_pos[:, 2] - terrain_h

        # 差分で高さの変化を取る（上昇時のみ報酬）
        delta_h = rel_h - getattr(self, "prev_rel_h", torch.zeros_like(rel_h))
        self.prev_rel_h = rel_h.clone()

        return torch.relu(delta_h)  # 上昇した分だけ報酬


    def _reward_rear_feet_level_with_front(self):
        # Get world-frame foot positions
        front_z = self.feet_front_pos[:, :, 2]  # (N, 2)
        rear_z  = self.feet_rear_pos[:, :, 2]   # (N, 2)

        # Terrain height under each foot
        terrain_front = self.get_terrain_height_at(
            self.feet_front_pos[:, :, 0], self.feet_front_pos[:, :, 1])
        terrain_rear = self.get_terrain_height_at(
            self.feet_rear_pos[:, :, 0], self.feet_rear_pos[:, :, 1])

        # Relative clearance above terrain
        front_rel = front_z - terrain_front  # (N, 2)
        rear_rel  = rear_z  - terrain_rear   # (N, 2)

        # Diagonal pairs: [FL, FR], [RL, RR]
        # Assume index 0 = FL, 1 = FR, 0 = RL, 1 = RR
        # Compare: FL vs RR, FR vs RL
        rel_FR = front_rel[:, 0]
        rel_FL = front_rel[:, 1]
        rel_RR = rear_rel[:, 0]
        rel_RL = rear_rel[:, 1]

        # Contact info: feet in contact = True
        contact_front = torch.norm(self.contact_forces[:, self.feet_front_indices, :3], dim=2) > 1.0  # (N, 2)
        contact_rear  = torch.norm(self.contact_forces[:, self.feet_rear_indices,  :3], dim=2) > 1.0  # (N, 2)

        # Diagonal contact status
        FR_contact = contact_front[:, 0]
        FL_contact = contact_front[:, 1]
        RR_contact = contact_rear[:, 0]
        RL_contact = contact_rear[:, 1]

        # diag1 = FL–RR; reward if at least one of them is in the air
        diag1_mask = ~(FL_contact & RR_contact)  # reward if not both in contact
        diag1_diff = torch.abs(rel_FL - rel_RR)
        reward_diag1 = torch.exp(-10.0 * diag1_diff) * diag1_mask.float()

        # diag2 = FR–RL; reward if at least one of them is in the air
        diag2_mask = ~(FR_contact & RL_contact)
        diag2_diff = torch.abs(rel_FR - rel_RL)
        reward_diag2 = torch.exp(-10.0 * diag2_diff) * diag2_mask.float()

        # Total reward
        return reward_diag1 + reward_diag2


    def _reward_calf_collision_low_clearance(self):
        penalty = torch.zeros(self.num_envs, device=self.device)

        for calf_idx, foot_idx in self.calf_to_foot_map.items():
            # 1. Check if calf is in contact
            calf_force = torch.norm(self.contact_forces[:, calf_idx, :], dim=1)
            calf_collision = calf_force > 1.0  # shape: (N,)
            feet_tensor_idx = self.link_idx_to_feet_tensor_idx.get(foot_idx, None)
            if feet_tensor_idx is None:
                continue  # skip if not found

            foot_z = self.feet_pos[:, feet_tensor_idx, 2]
            terrain_z = self.get_terrain_height_at(
                self.feet_pos[:, feet_tensor_idx, 0],
                self.feet_pos[:, feet_tensor_idx, 1]
            )
            
            clearance = foot_z - terrain_z

            # 3. Penalize if clearance is low while calf is in contact
            low_clearance = clearance < 0.05
            penalty += calf_collision.float() * low_clearance.float()

        return penalty


    def _reward_feet_distance_diff(self):
        """
        Penalize front/rear feet if their x-position in base frame moves too far
        from the default position. Allow some margin (e.g., 10cm) before penalizing.
        The penalty grows exponentially with the exceeded amount.
        """
        margin = 0.2  # [m] 許容距離
        scale = 10.0   # 指数スケーリング係数

        # デフォルト位置（ベースフレーム基準）
        default_front_x = 0.1934  #self.env_cfg["default_feet_pos_base"]["front"]  # 例: 0.25
        default_rear_x  = -0.1934 #self.env_cfg["default_feet_pos_base"]["rear"]   # 例: -0.25

        # 現在の足位置（ベースフレームでの x 座標）
        front_x = self.front_feet_pos_base[:, :, 0]  # shape: (N, 2)
        rear_x  = self.rear_feet_pos_base[:, :, 0]   # shape: (N, 2)

        # ⛔ 差分そのものは「絶対値」で固定（変更しない）
        front_diff = torch.abs(front_x - default_front_x)  # (N, 2)
        rear_diff  = torch.abs(rear_x  - default_rear_x)   # (N, 2)

        # ✅ マージン以下ならゼロ、それを超えた分だけ指数罰
        front_excess = torch.relu(front_diff - margin)  # (N, 2)
        rear_excess  = torch.relu(rear_diff  - margin)  # (N, 2)

        # 📈 指数ペナルティ（調整可能）
        penalty_front = torch.exp(scale * front_excess) - 1.0
        penalty_rear  = torch.exp(scale * rear_excess)  - 1.0

        # 🎯 合計
        total_penalty = penalty_front.sum(dim=1) + penalty_rear.sum(dim=1)
        return total_penalty


    def _reward_front_leg_retraction(self):
        # FLとFRのx方向ベースフレーム位置（大きすぎると前に伸びすぎ）
        front_x = self.front_feet_pos_base[:, :, 0]
        penalty = torch.relu(front_x - 0.4)  # デフォルト位置より前に出すぎたら罰
        return penalty.sum(dim=1)

    def _reward_thigh_retraction(self):
        return -self.thigh_pos_base[:, :, 0].mean(dim=1)



    def _reward_foot_clearance(self):
        # 1. 足のワールド位置・速度を取得
        foot_world_pos = self.feet_pos             # shape: (N, 4, 3)
        foot_world_vel = self.feet_vel             # shape: (N, 4, 3)

        # 2. ボディ中心位置と速度を減算（相対座標系に変換）
        rel_pos = foot_world_pos - self.base_pos.unsqueeze(1)     # shape: (N, 4, 3)
        rel_vel = foot_world_vel - self.base_lin_vel.unsqueeze(1) # shape: (N, 4, 3)

        # 3. ボディ座標系に変換（回転のみ適用）
        pos_body = self._world_to_base_transform(foot_world_pos, self.base_pos, self.base_quat)  # shape: (N, 4, 3)
        vel_body = self._world_to_base_transform(foot_world_vel, torch.zeros_like(self.base_pos), self.base_quat)

        # 4. z方向の高さ誤差（二乗誤差）
        target_height = self.reward_cfg["foot_clearance_height_target"]  # e.g. 0.1 [m]
        height_error = torch.square(pos_body[:, :, 2] - target_height)  # shape: (N, 4)

        # 5. 横方向の速度（x-y平面のnorm）
        lateral_vel = torch.norm(vel_body[:, :, :2], dim=2)  # shape: (N, 4)

        # 6. 報酬計算（高さ誤差 × 横速度）
        clearance_reward = height_error * lateral_vel

        return torch.sum(clearance_reward, dim=1)  # shape: (N,)


    def _reward_powers(self):
        # Penalize torques
        return torch.sum(torch.abs(self.torques)*torch.abs(self.dof_vel), dim=1)

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)