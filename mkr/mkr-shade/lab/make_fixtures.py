"""Copy a few quads of chosen lab cases into tests/fixtures/ (graph + inputs + Blender's values) for `cargo test`.

    python3 mkr/mkr-shade/lab/make_fixtures.py LAB_DIR"""
import json
import os
import sys

import numpy as np

CASES = ["noise_3d_fbm_distort", "noise_4d_fbm_detail", "noise_2d_ridged_multifractal_detail",
         "voronoi_3d_f1_euclidean_distance", "voronoi_2d_smooth_f1_euclidean_color", "voronoi_3d_f2_manhattan_distance",
         "wave_bands_diagonal_sin_distort", "mixrgb_overlay", "mixrgb_soft_light", "mix_hue", "ramp_cardinal_rgb_4",
         "ramp_b_spline_hsv_4", "hue_sat", "separate_hsv", "maprange_smoothstep", "math_power", "math_wrap_",
         "vmath_refract", "mapping_texture", "gradient_radial", "magic_10", "brick_smooth", "checker",
         "white_noise_3d", "color_to_float", "mmd_sphere_uv", "layer_weight_facing", "bump_waves", "normal_map_world",
         "camera_data_view_vector"]
PER_CASE = 24


def main():
    lab = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "..", "tests", "fixtures")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(3)
    for name in CASES:
        path = os.path.join(lab, name + ".json")
        if not os.path.exists(path):
            continue
        doc = json.load(open(path))
        n = doc["n"]
        inp = np.fromfile(os.path.join(lab, name + ".in.bin"), np.float32).reshape(-1, 9)
        ref = np.fromfile(os.path.join(lab, name + ".ref.bin"), np.float32).reshape(-1, 4)
        # grazing rows (N.I < 0.08) of shading cases hang on Blender's custom-normal precision: left out
        pool = np.arange(n * n) if doc["kind"] == "field" else np.arange(3 * n, n * n)
        quads = sorted(rng.choice(pool, PER_CASE, replace=False).tolist())
        fx = {"name": name, "kind": doc["kind"], "n": n, "blender": doc["blender"], "material": doc["material"],
              "rows": doc["rows"], "tolerance": 0.002 if doc["kind"] == "field" else 0.005,
              "samples": [{"quad": q, "in": inp[q].tolist(), "ref": ref[q, :3].tolist()} for q in quads]}
        with open(os.path.join(out_dir, name + ".json"), "w") as fh:
            json.dump(fx, fh)
    print("fixtures in", os.path.normpath(out_dir), len(os.listdir(out_dir)))


main()
