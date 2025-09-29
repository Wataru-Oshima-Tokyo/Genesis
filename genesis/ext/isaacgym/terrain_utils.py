# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np
from scipy import interpolate
import genesis as gs
import random
# from scipy.interpolate import RegularGridInterpolator


def fractal_terrain(terrain, levels=8, scale=1.0):
    """
    Generates a fractal terrain

    Parameters
        terrain (SubTerrain): the terrain
        levels (int, optional): granurarity of the fractal terrain. Defaults to 8.
        scale (float, optional): scales vertical variation. Defaults to 1.0.
    """
    width = terrain.width
    length = terrain.length
    height = np.zeros((width, length), dtype=gs.np_float)
    for level in range(1, levels + 1):
        step = 2 ** (levels - level)
        for y in range(0, width, step):
            y_skip = (1 + y // step) % 2
            for x in range(step * y_skip, length, step * (1 + y_skip)):
                x_skip = (1 + x // step) % 2
                xref = step * (1 - x_skip)
                yref = step * (1 - y_skip)
                mean = height[y - yref : y + yref + 1 : 2 * step, x - xref : x + xref + 1 : 2 * step].mean()
                variation = 2 ** (-level) * np.random.uniform(-1, 1)
                height[y, x] = mean + scale * variation

    height /= terrain.vertical_scale
    terrain.height_field_raw = height
    return terrain


def random_uniform_terrain(
    terrain,
    min_height,
    max_height,
    step=1,
    downsampled_scale=None,
):
    """
    Generate a uniform noise terrain

    Parameters
        terrain (SubTerrain): the terrain
        min_height (float): the minimum height of the terrain [meters]
        max_height (float): the maximum height of the terrain [meters]
        step (float): minimum height change between two points [meters]
        downsampled_scale (float): distance between two randomly sampled points ( musty be larger or equal to terrain.horizontal_scale)

    """
    if downsampled_scale is None:
        downsampled_scale = terrain.horizontal_scale
    scaled_width = terrain.width * terrain.horizontal_scale
    scaled_length = terrain.length * terrain.horizontal_scale

    # switch parameters to discrete units
    min_height = int(min_height / terrain.vertical_scale)
    max_height = int(max_height / terrain.vertical_scale)
    step = int(step / terrain.vertical_scale)

    heights_range = np.arange(min_height, max_height + step, step)
    height_field_downsampled = np.random.choice(
        heights_range,
        (
            int(scaled_width / downsampled_scale),
            int(scaled_length / downsampled_scale),
        ),
    )

    x = np.linspace(0, scaled_width, height_field_downsampled.shape[0])
    y = np.linspace(0, scaled_length, height_field_downsampled.shape[1])

    f = interpolate.RectBivariateSpline(x, y, height_field_downsampled)

    x_upsampled = np.linspace(0, scaled_width, terrain.width)
    y_upsampled = np.linspace(0, scaled_length, terrain.length)
    z_upsampled = np.rint(f(x_upsampled, y_upsampled))

    terrain.height_field_raw += z_upsampled
    return terrain






def sloped_terrain(terrain, slope=1):
    """
    Generate a sloped terrain

    Parameters:
        terrain (SubTerrain): the terrain
        slope (int): positive or negative slope
    Returns:
        terrain (SubTerrain): update terrain
    """

    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = xx.reshape(terrain.width, 1)
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * terrain.width)
    terrain.height_field_raw[:, np.arange(terrain.length)] += max_height * xx / terrain.width
    return terrain


def pyramid_sloped_terrain(terrain, slope=1, platform_size=1.0):
    """
    Generate a sloped terrain

    Parameters:
        terrain (terrain): the terrain
        slope (int): positive or negative slope
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    center_x = int(terrain.width / 2)
    center_y = int(terrain.length / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = (center_x - np.abs(center_x - xx)) / center_x
    yy = (center_y - np.abs(center_y - yy)) / center_y
    xx = xx.reshape(terrain.width, 1)
    yy = yy.reshape(1, terrain.length)
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * (terrain.width / 2))
    terrain.height_field_raw += max_height * xx * yy

    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.width // 2 - platform_size
    x2 = terrain.width // 2 + platform_size
    y1 = terrain.length // 2 - platform_size
    y2 = terrain.length // 2 + platform_size

    min_h = min(terrain.height_field_raw[x1, y1], 0)
    max_h = max(terrain.height_field_raw[x1, y1], 0)
    terrain.height_field_raw = np.clip(terrain.height_field_raw, min_h, max_h)
    return terrain


def discrete_obstacles_terrain(terrain, slope=-0.5, pit_size_m=0.2, pit_gap_m=0.4, pit_depth_m=0.2, platform_size_m=0.5):

    """
    Generate a sloped terrain with uniformly spaced pit holes and a flat platform in the center.

    Parameters:
        terrain: terrain object with attributes width, length, height_field_raw, horizontal_scale, vertical_scale
        slope (float): slope direction and magnitude
        platform_size_m (float): size of the center flat platform [meters]
        pit_size_m (float): width/length of each square pit [meters]
        pit_gap_m (float): spacing between pit centers [meters]
        pit_depth_m (float): depth of each pit [meters]
    Returns:
        terrain: modified terrain
    """

    width, length = terrain.width, terrain.length
    center_x = width // 2
    center_y = length // 2

    # Convert dimensions to terrain units
    platform_size = int(platform_size_m / terrain.horizontal_scale)
    pit_size = int(pit_size_m / terrain.horizontal_scale)
    pit_gap = int(pit_gap_m / terrain.horizontal_scale)
    pit_half = pit_size // 2
    pit_depth = int(pit_depth_m / terrain.vertical_scale)

    # Generate pyramid slope
    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = (center_x - np.abs(center_x - xx)) / center_x
    yy = (center_y - np.abs(center_y - yy)) / center_y
    xx = xx.reshape(terrain.width, 1)
    yy = yy.reshape(1, terrain.length)
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * (terrain.width / 2))
    terrain.height_field_raw[:, :] = (max_height * xx * yy).astype(terrain.height_field_raw.dtype)

    # Flatten the center platform
    half_platform = platform_size // 2
    px1 = center_x - half_platform
    px2 = center_x + half_platform
    py1 = center_y - half_platform
    py2 = center_y + half_platform
    center_height = terrain.height_field_raw[center_x, center_y]
    terrain.height_field_raw[px1:px2, py1:py2] = center_height

    # Uniform pit placement
    k = 0
    for i in range(pit_gap // 2, width, pit_gap):
        for j in range(pit_gap // 2, length, pit_gap):
            if (px1 - pit_half <= i <= px2 + pit_half and
                py1 - pit_half <= j <= py2 + pit_half):
                continue  # skip the center platform area

            x1 = max(i - pit_half, 0)
            x2 = min(i + pit_half, width)
            y1 = max(j - pit_half, 0)
            y2 = min(j + pit_half, length)
            if k%2 == 0:
                terrain.height_field_raw[x1:x2, y1:y2] -= pit_depth
            k+=1
    return terrain


def wave_terrain(terrain, num_waves=1, amplitude=1.0):
    """
    Generate a wavy terrain

    Parameters:
        terrain (terrain): the terrain
        num_waves (int): number of sine waves across the terrain length
        amplitude (float): amplitude of the sine waves [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    amplitude = 0.5 * amplitude / terrain.vertical_scale
    if num_waves > 0:
        div = terrain.length / (num_waves * np.pi * 2)
        x = np.arange(0, terrain.width)
        y = np.arange(0, terrain.length)
        xx, yy = np.meshgrid(x, y, sparse=True)
        xx = xx.reshape(terrain.width, 1)
        yy = yy.reshape(1, terrain.length)
        terrain.height_field_raw += amplitude * np.cos(yy / div) + amplitude * np.sin(xx / div)
    return terrain


def stairs_terrain(terrain, step_width, step_height):
    """
    Generate a stairs

    Parameters:
        terrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float):  the height of the step [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)

    num_steps = terrain.width // step_width
    height = step_height
    for i in range(num_steps):
        terrain.height_field_raw[i * step_width : (i + 1) * step_width, :] += height
        height += step_height
    return terrain


def pyramid_stairs_terrain(terrain, step_width, step_height, platform_size=1.0):
    """
    Generate stairs

    Parameters:
        terrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float): the step_height [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    height = 0
    start_x = 0
    stop_x = terrain.width
    start_y = 0
    stop_y = terrain.length
    while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
        start_x += step_width
        stop_x -= step_width
        start_y += step_width
        stop_y -= step_width
        height += step_height
        terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = height
    return terrain


def stepping_stones_terrain(terrain, stone_size, stone_distance, max_height, platform_size=1.0, depth=-10):
    """
    Generate a stepping stones terrain

    Parameters:
        terrain (terrain): the terrain
        stone_size (float): horizontal size of the stepping stones [meters]
        stone_distance (float): distance between stones (i.e size of the holes) [meters]
        max_height (float): maximum height of the stones (positive and negative) [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        depth (float): depth of the holes (default=-10.) [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(-max_height - 1, max_height, step=1)

    start_x = 0
    start_y = 0
    terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)
    if terrain.length >= terrain.width:
        while start_y < terrain.length:
            stop_y = min(terrain.length, start_y + stone_size)
            start_x = np.random.randint(0, stone_size)
            # fill first hole
            stop_x = max(0, start_x - stone_distance)
            terrain.height_field_raw[0:stop_x, start_y:stop_y] = np.random.choice(height_range)
            # fill row
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size)
                terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(height_range)
                start_x += stone_size + stone_distance
            start_y += stone_size + stone_distance
    elif terrain.width > terrain.length:
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            start_y = np.random.randint(0, stone_size)
            # fill first hole
            stop_y = max(0, start_y - stone_distance)
            terrain.height_field_raw[start_x:stop_x, 0:stop_y] = np.random.choice(height_range)
            # fill column
            while start_y < terrain.length:
                stop_y = min(terrain.length, start_y + stone_size)
                terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(height_range)
                start_y += stone_size + stone_distance
            start_x += stone_size + stone_distance

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain



def stamble_terrain(
        terrain,
        patch_size_m=0.4,
        gap_m=0.1):   
    """
    Generate an aggressive terrain with a uniform grid of steps and pits,
    each separated by a 0.1m gap, and a flat 0.7x0.7m platform in the center.
    Each patch has a randomly chosen height between 0.05m and 0.15m.
    """
    # Parameters
    platform_size_m = 0.5
    height_range_m = (0.10, 0.25)

    # Convert to terrain units
    patch_size = int(patch_size_m / terrain.horizontal_scale)
    gap = int(gap_m / terrain.horizontal_scale)
    platform_size = int(platform_size_m / terrain.horizontal_scale)

    # Terrain size
    width, length = terrain.width, terrain.length
    center_x = width // 2
    center_y = length // 2
    half_platform = platform_size // 2

    # Clear terrain
    terrain.height_field_raw[:, :] = 0

    # Place patches in a grid
    x = 0
    row = 0
    while x + patch_size < width:
        y = 0
        col = 0
        while y + patch_size < length:
            # Patch center
            patch_cx = x + patch_size // 2
            patch_cy = y + patch_size // 2

            # Skip platform region
            if (abs(patch_cx - center_x) < half_platform + gap and
                abs(patch_cy - center_y) < half_platform + gap):
                y += patch_size + gap
                col += 1
                continue

            # Gap-adjusted bounds
            x1 = x + gap
            x2 = min(x + patch_size - gap, width)
            y1 = y + gap
            y2 = min(y + patch_size - gap, length)

            # Random height between 5cm and 15cm
            height_m = random.uniform(*height_range_m)
            height = int(height_m / terrain.vertical_scale)
            height = height if (row + col) % 2 == 0 else -height

            if x1 < x2 and y1 < y2:
                terrain.height_field_raw[x1:x2, y1:y2] += height

            y += patch_size + gap
            col += 1
        x += patch_size + gap
        row += 1

    # Center flat platform
    x_start = max(0, center_x - half_platform)
    x_end   = min(width, center_x + half_platform)
    y_start = max(0, center_y - half_platform)
    y_end   = min(length, center_y + half_platform)
    terrain.height_field_raw[x_start:x_end, y_start:y_end] = 0

    return terrain

def blocky_terrain(
        terrain,
        patch_size_m=0.4,
        gap_m=0.1):   
    """
    Generate an aggressive terrain with a uniform grid of steps and pits,
    each separated by a 0.1m gap, and a flat 0.7x0.7m platform in the center.
    Each patch has a randomly chosen height between 0.05m and 0.15m.
    """
    # Parameters
    platform_size_m = 0.5
    height_range_m = (0.10, 0.25)

    # Convert to terrain units
    patch_size = int(patch_size_m / terrain.horizontal_scale)
    gap = int(gap_m / terrain.horizontal_scale)
    platform_size = int(platform_size_m / terrain.horizontal_scale)

    # Terrain size
    width, length = terrain.width, terrain.length
    center_x = width // 2
    center_y = length // 2
    half_platform = platform_size // 2

    # Clear terrain
    terrain.height_field_raw[:, :] = 0

    # Place patches in a grid
    x = 0
    row = 0
    while x + patch_size < width:
        y = 0
        col = 0
        while y + patch_size < length:
            # Patch center
            patch_cx = x + patch_size // 2
            patch_cy = y + patch_size // 2

            # Skip platform region
            if (abs(patch_cx - center_x) < half_platform + gap and
                abs(patch_cy - center_y) < half_platform + gap):
                y += patch_size + gap
                col += 1
                continue

            # Gap-adjusted bounds
            x1 = x + gap
            x2 = min(x + patch_size - gap, width)
            y1 = y + gap
            y2 = min(y + patch_size - gap, length)

            # Random height between 5cm and 15cm
            height_m = random.uniform(*height_range_m)
            height = int(height_m / terrain.vertical_scale)
            height = height if (row + col) % 2 == 0 else -height

            if x1 < x2 and y1 < y2:
                terrain.height_field_raw[x1:x2, y1:y2] += height

            y += patch_size + gap
            col += 1
        x += patch_size + gap
        row += 1

    # Center flat platform
    x_start = max(0, center_x - half_platform)
    x_end   = min(width, center_x + half_platform)
    y_start = max(0, center_y - half_platform)
    y_end   = min(length, center_y + half_platform)
    terrain.height_field_raw[x_start:x_end, y_start:y_end] = 0

    return terrain



def debug_terrain(terrain):
    slope=-0.5 
    platform_size_m=0.5 
    pit_size_m=0.2
    pit_gap_m=0.4
    pit_depth_m=0.3
    """
    Generate a sloped terrain with uniformly spaced pit holes and a flat platform in the center.

    Parameters:
        terrain: terrain object with attributes width, length, height_field_raw, horizontal_scale, vertical_scale
        slope (float): slope direction and magnitude
        platform_size_m (float): size of the center flat platform [meters]
        pit_size_m (float): width/length of each square pit [meters]
        pit_gap_m (float): spacing between pit centers [meters]
        pit_depth_m (float): depth of each pit [meters]
    Returns:
        terrain: modified terrain
    """

    width, length = terrain.width, terrain.length
    center_x = width // 2
    center_y = length // 2

    # Convert dimensions to terrain units
    platform_size = int(platform_size_m / terrain.horizontal_scale)
    pit_size = int(pit_size_m / terrain.horizontal_scale)
    pit_gap = int(pit_gap_m / terrain.horizontal_scale)
    pit_half = pit_size // 2
    pit_depth = int(pit_depth_m / terrain.vertical_scale)

    # Generate pyramid slope
    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = (center_x - np.abs(center_x - xx)) / center_x
    yy = (center_y - np.abs(center_y - yy)) / center_y
    xx = xx.reshape(terrain.width, 1)
    yy = yy.reshape(1, terrain.length)
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * (terrain.width / 2))
    terrain.height_field_raw[:, :] = (max_height * xx * yy).astype(terrain.height_field_raw.dtype)

    # Flatten the center platform
    half_platform = platform_size // 2
    px1 = center_x - half_platform
    px2 = center_x + half_platform
    py1 = center_y - half_platform
    py2 = center_y + half_platform
    center_height = terrain.height_field_raw[center_x, center_y]
    terrain.height_field_raw[px1:px2, py1:py2] = center_height

    # Uniform pit placement
    k = 0
    for i in range(pit_gap // 2, width, pit_gap):
        for j in range(pit_gap // 2, length, pit_gap):
            if (px1 - pit_half <= i <= px2 + pit_half and
                py1 - pit_half <= j <= py2 + pit_half):
                continue  # skip the center platform area

            x1 = max(i - pit_half, 0)
            x2 = min(i + pit_half, width)
            y1 = max(j - pit_half, 0)
            y2 = min(j + pit_half, length)
            if k%2 == 0:
                terrain.height_field_raw[x1:x2, y1:y2] -= pit_depth
            k+=1
    return terrain

def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[: num_rows - 1, :] += hf[1:num_rows, :] - hf[: num_rows - 1, :] > slope_threshold
        move_x[1:num_rows, :] -= hf[: num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold
        move_y[:, : num_cols - 1] += hf[:, 1:num_cols] - hf[:, : num_cols - 1] > slope_threshold
        move_y[:, 1:num_cols] -= hf[:, : num_cols - 1] - hf[:, 1:num_cols] > slope_threshold
        move_corners[: num_rows - 1, : num_cols - 1] += (
            hf[1:num_rows, 1:num_cols] - hf[: num_rows - 1, : num_cols - 1] > slope_threshold
        )
        move_corners[1:num_rows, 1:num_cols] -= (
            hf[: num_rows - 1, : num_cols - 1] - hf[1:num_rows, 1:num_cols] > slope_threshold
        )
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flat
    vertices[:, 1] = yy.flat
    vertices[:, 2] = (hf * vertical_scale).flat
    triangles = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start + 1 : stop : 2, 0] = ind0
        triangles[start + 1 : stop : 2, 1] = ind2
        triangles[start + 1 : stop : 2, 2] = ind3

    return vertices, triangles


class SubTerrain:
    def __init__(self, terrain_name="terrain", width=256, length=256, vertical_scale=1.0, horizontal_scale=1.0):
        self.terrain_name = terrain_name
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.width = width
        self.length = length
        self.height_field_raw = np.zeros((self.width, self.length), dtype=gs.np_float)
