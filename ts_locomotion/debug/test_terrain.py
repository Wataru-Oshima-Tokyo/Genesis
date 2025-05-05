import argparse
import genesis as gs
import numpy as np
import torch
import time

def get_height_at_xy(height_field, x, y, horizontal_scale, vertical_scale, center_x, center_y):
    # shift x, y by terrain's center
    x_shifted = x + center_x
    y_shifted = y + center_y

    j = int(x_shifted / horizontal_scale)
    i = int(y_shifted / horizontal_scale)

    if 0 <= i < height_field.shape[0] and 0 <= j < height_field.shape[1]:
        z = height_field[i, j] * vertical_scale
        return z
    else:
        raise ValueError(f"Requested (x={x}, y={y}) is outside the terrain bounds.")

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
        show_viewer=True,
    )

    ########################## parameters ##########################
    subterrain_size = 3.0
    horizontal_scale = 0.05
    vertical_scale = 0.005
    n_subterrains = (1, 1)

    total_width = n_subterrains[0] * subterrain_size
    total_height = n_subterrains[1] * subterrain_size

    center_x = total_width / 2
    center_y = total_height / 2

    grid = [[None for _ in range(1)] for _ in range(1)]
    grid[0][0] = "pyramid_steep_down_stairs_terrain"
    # grid[0][1] = "discrete_obstacles_terrain"
    # grid[1][0] = "wave_terrain"
    # grid[1][1] = "pyramid_steep_down_stairs_terrain"
    ########################## create terrain ##########################
    terrain = scene.add_entity(
        morph = gs.morphs.Terrain(
            pos=(-center_x, -center_y, 0),
            subterrain_size=(subterrain_size, subterrain_size),
            n_subterrains=n_subterrains,
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            subterrain_types=grid,
        ),
    )
    # scene.add_entity(terrain)
    ball = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.1,
            pos=(1.0, 1.0, 1.0),
        ),
    )
    ########################## build ##########################
    scene.build()

    ########################## get heightfield ##########################
    height_field = terrain.geoms[0].metadata["height_field"]

    ########################## query height ##########################
    x_query = 0.0
    y_query = 0.0
    z_query = get_height_at_xy(height_field, x_query, y_query, horizontal_scale, vertical_scale, center_x, center_y)
    # print(f"Height at ({x_query},{y_query}): {z_query}")

    ########################## add a ball ##########################
    ball.set_pos(torch.tensor((x_query, y_query, z_query + 0.4)))
    ########################## simulation loop ##########################
    for step in range(10000):
        # Get current position of the ball
        pos = ball.get_pos()  # returns torch.tensor([x, y, z])
        x = pos[0].item()
        y = pos[1].item()
        z = pos[2].item()

        # Get terrain height at (x, y)
        try:
            terrain_z = get_height_at_xy(height_field, x, y, horizontal_scale, vertical_scale, center_x, center_y)
        except ValueError as e:
            print(f"[Step {step}] Position ({x:.3f}, {y:.3f}) out of bounds: {e}")
            terrain_z = float('nan')

        # Compute height difference
        height_diff = z - terrain_z if not np.isnan(terrain_z) else float('nan')

        # Print debug info
        print(f"[Step {step}] Ball pos: ({x:.3f}, {y:.3f}, {z:.3f}) | Terrain height: {terrain_z:.3f} | Height diff: {height_diff:.3f}")

        scene.step()
        time.sleep(5)

if __name__ == "__main__":
    main()
