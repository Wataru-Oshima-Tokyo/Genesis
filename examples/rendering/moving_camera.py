import argparse

import genesis as gs
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init()

    ########################## create a scene ##########################

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(file="xml/go1/go1.xml"),
    )


    ########################## cameras ##########################
    cam_0 = scene.add_camera(
        res=(1280, 960),
        pos=(3.5, 0.0, 2.5),
        lookat=(0, 0, 0.5),
        fov=30,
        GUI=True,
    )
    ########################## build ##########################

    joints_name = (
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
    )

    for name in joints_name:
        joint = robot.get_joint(name)
        print(f"{name}: dofs_idx_local={joint.dofs_idx_local}, dof_start={joint.dof_start}")
    # scene.build()
    # dof_pos_limits = torch.stack(robot.get_dofs_limit(motors_dof_idx), dim=1)

    # print(f"moto dof idx {motors_dof_idx}")
    # print(f"dof_pos_limits {dof_pos_limits}")

    # # for i in range(1000):
    # #     scene.step()
    # #     cam_0.set_pose(pos=(i / 100, 0, 2.5))
    # #     cam_0.render(
    # #         rgb=True,
    # #         # depth        = True,
    # #         # segmentation = True,
    # #     )


if __name__ == "__main__":
    main()
