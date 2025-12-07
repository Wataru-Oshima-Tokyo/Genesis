"""
Example: small 1 m x 1 m grid LiDAR centered on the robot.

We attach a grid-pattern LiDAR to a simple box robot. The sensor casts a 10 x 10
grid of rays straight down over a 1 m square so you can visualize the points
immediately around the robot. Debug drawing must be enabled in the viewer to
see the rays and hit points.
"""

import argparse
import os

import genesis as gs

GRID_SIZE_M = 1.0
GRID_POINTS = 10  # 10 x 10 rays
GRID_RES_M = GRID_SIZE_M / (GRID_POINTS - 1)  # include both edges of the square
SENSOR_HEIGHT = 0.5


def main():
    parser = argparse.ArgumentParser(description="Genesis 1 m x 1 m grid LiDAR sample")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU")
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(gravity=(0.0, 0.0, -9.81)),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-2.0, 0.0, 1.75),
            camera_lookat=(0.0, 0.0, 0.25),
            max_FPS=60,
        ),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=True,
    )
    scene.add_entity(gs.morphs.Plane())

    robot = scene.add_entity(
        gs.morphs.Box(
            size=(0.25, 0.25, 0.25),
            pos=(0.0, 0.0, 0.125),
            fixed=True,
        )
    )

    # Small obstacles inside the 1 m square patch to make hits visible
    scene.add_entity(gs.morphs.Box(size=(0.15, 0.15, 0.35), pos=(0.25, -0.2, 0.175), fixed=True))
    scene.add_entity(gs.morphs.Cylinder(radius=0.08, height=0.3, pos=(-0.25, 0.2, 0.15), fixed=True))

    grid_pattern = gs.sensors.GridPattern(
        resolution=GRID_RES_M,
        size=(GRID_SIZE_M, GRID_SIZE_M),
        direction=(0.0, 0.0, -1.0),  # cast downward to the ground
    )

    lidar_sensor = scene.add_sensor(
        gs.sensors.Lidar(
            pattern=grid_pattern,
            entity_idx=robot.idx,
            pos_offset=(0.0, 0.0, SENSOR_HEIGHT),
            euler_offset=(0.0, 0.0, 0.0),
            return_world_frame=True,
            draw_debug=True,
        )
    )

    scene.build()
    gs.logger.info(
        f"Grid LiDAR configured for {GRID_POINTS}x{GRID_POINTS} rays over {GRID_SIZE_M}m x {GRID_SIZE_M} around the robot."
    )

    step_count = 0
    try:
        while True:
            scene.step()
            step_count += 1

            # Periodically report the highest hit point within the 1 m x 1 m grid
            if step_count % 30 == 0:
                lidar_data = lidar_sensor.read()
                max_height = lidar_data.points[..., 2].max().item()
                gs.logger.info(f"Max hit height in grid: {max_height:.3f} m")

            if "PYTEST_VERSION" in os.environ:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
