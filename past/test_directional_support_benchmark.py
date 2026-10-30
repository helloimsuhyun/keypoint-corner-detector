import numpy as np

from directional_support_benchmark import (
    SyntheticPatch,
    antipodal_sphere,
    evaluate_patch,
    exact_support_mask,
    make_synthetic_patches,
    tangent_polar_mask,
)


def _with_query(patch: SyntheticPatch) -> SyntheticPatch:
    return SyntheticPatch(
        patch.name,
        np.vstack((patch.points_mm, np.zeros((1, 3)))),
        np.vstack((patch.normals, np.full((1, 3), np.nan))),
        patch.desired_corner,
        patch.sharp_corner,
        patch.description,
    )


def test_antipodal_sphere_contains_exact_opposites():
    directions = antipodal_sphere(512)
    np.testing.assert_allclose(directions[:256], -directions[256:])
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)


def test_tangent_polar_and_exact_support_are_identical():
    directions = antipodal_sphere(512)
    for patch in make_synthetic_patches():
        points = _with_query(patch).points_mm
        local = points[np.linalg.norm(points, axis=1) <= 26.0]
        np.testing.assert_array_equal(
            exact_support_mask(local, directions),
            tangent_polar_mask(local, directions),
        )


def test_reflected_convex_and_concave_trihedra_have_equal_solid_angle():
    patches = {patch.name: _with_query(patch) for patch in make_synthetic_patches()}
    directions = antipodal_sphere(1024)
    scores = {}
    for name in ("sharp_convex_corner", "concave_trihedral_mirror"):
        row, _, _ = evaluate_patch(
            patches[name],
            directions,
            radius_mm=26.0,
            tolerance_mm=0.0,
            quantile=1.0,
            don_small_radius_mm=8.0,
        )
        scores[name] = row["exact_support_score"]
    assert scores["sharp_convex_corner"] == scores["concave_trihedral_mirror"]


def test_robust_directional_support_separates_sharp_vertex_from_smooth_queries():
    patches = {patch.name: _with_query(patch) for patch in make_synthetic_patches()}
    directions = antipodal_sphere(1024)
    scores = {}
    for name in ("sharp_convex_corner", "face_interior", "sphere_r70"):
        row, _, _ = evaluate_patch(
            patches[name],
            directions,
            radius_mm=26.0,
            tolerance_mm=0.6,
            quantile=0.95,
            don_small_radius_mm=8.0,
        )
        scores[name] = row["support_score_q95"]
    assert scores["sharp_convex_corner"] > 0.10
    assert scores["face_interior"] < 0.05
    assert scores["sphere_r70"] < 0.05


def test_planar_visibility_crop_corner_is_an_unavoidable_local_false_positive():
    patches = {patch.name: _with_query(patch) for patch in make_synthetic_patches()}
    directions = antipodal_sphere(1024)
    crop_row, _, _ = evaluate_patch(
        patches["visibility_crop_corner"],
        directions,
        radius_mm=26.0,
        tolerance_mm=0.6,
        quantile=0.95,
        don_small_radius_mm=8.0,
    )
    sharp_row, _, _ = evaluate_patch(
        patches["sharp_convex_corner"],
        directions,
        radius_mm=26.0,
        tolerance_mm=0.6,
        quantile=0.95,
        don_small_radius_mm=8.0,
    )
    assert crop_row["support_score_q95"] > 0.10
    assert crop_row["support_score_q95"] > sharp_row["support_score_q95"]
