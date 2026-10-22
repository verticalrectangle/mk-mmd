"""Camera clip ranges fitted to what a camera sees (mkmmd.core.clip), bpy-free: the least depth of geometry inside a
camera's view is exact (never farther than any point of it that shows, so nothing is cut off), a box the camera is in
counts from the lens, the coarse-to-fine search finds what the plain minimum over every triangle finds, and the start is
half of it. The clip stage keying it on a build is checked in test_build_stages.py."""
import numpy as np
import pytest

from mkmmd.core import clip as CL

VIEW = CL.View(35.0, 36.0, (1920, 1080), shift=(0.05, -0.02))


def sampled(tri, view, n=400):
    """The least depth over a dense grid of points of one triangle that are inside the view (inf: none)."""
    a, b = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    keep = a + b <= 1
    P = tri[0] + a[keep][:, None] * (tri[1] - tri[0]) + b[keep][:, None] * (tri[2] - tri[0])
    inside = ((P @ view.planes.T) >= -1e-12).all(1)
    return float((-P[inside, 2]).min()) if inside.any() else np.inf


def test_the_least_depth_of_a_triangle_is_where_its_clipped_corner_is_never_beyond_a_point_that_shows():
    rng = np.random.default_rng(3)
    met = 0
    for _ in range(150):
        c = np.array([rng.uniform(-3, 3), rng.uniform(-2, 2), -rng.uniform(-1, 6)])
        tri = c + rng.normal(0, rng.choice([0.05, 0.5, 3.0]), (3, 3))
        exact, near = CL.tri_depth(tri[None], VIEW)[0], sampled(tri, VIEW)
        step = max(np.linalg.norm(tri[i] - tri[i - 1]) for i in range(3)) / 400
        if np.isfinite(near):
            met += 1
            assert exact <= near + 1e-12                               # nothing that shows is nearer than it ...
            assert near - exact < 3 * step, (exact, near, step)        # ... and it is that near (to the grid's step)
    assert met > 40


def test_a_floor_whose_corners_are_all_out_of_view_shows_from_where_the_bottom_of_the_view_meets_it():
    """A camera 0.3 m above a big floor quad, looking level: none of its corners is in view, the edges cross the view's
    sides far off, and it first shows at 0.3 m over the bottom side's tangent, where the view's lower edges meet it."""
    view = CL.View(50.0, 36.0, (640, 360))
    quad = np.array([[-40.0, -0.3, 1.0], [40.0, -0.3, 1.0], [40.0, -0.3, -80.0], [-40.0, -0.3, -80.0]])
    tris = quad[[[0, 1, 2], [0, 2, 3]]]
    assert CL.tri_depth(tris, view).min() == pytest.approx(0.3 / -view.yb, rel=1e-9)
    assert -view.yb == pytest.approx(1.1 * 180 * 36.0 / 50.0 / 640)    # the bottom side, widened by a tenth


def test_a_box_counts_from_the_lens_when_the_camera_is_in_it_and_from_its_near_face_when_it_is_ahead():
    eye = np.eye(4)
    d = CL.box_depth([[-1, -1, -5], [-0.2, -0.2, -0.2]], [[1, 1, -3], [0.2, 0.2, 0.2]], eye, VIEW)
    assert d[0] == pytest.approx(3.0) and d[1] == 0.0
    behind = CL.box_depth([[-1, -1, 2]], [[1, 1, 4]], eye, VIEW)
    beside = CL.box_depth([[30, -1, -3]], [[31, 1, -2]], eye, VIEW)
    assert np.isinf(behind[0]) and np.isinf(beside[0])


def test_the_search_finds_what_the_minimum_over_every_triangle_finds():
    rng = np.random.default_rng(7)
    for _ in range(30):
        items, every = [], []
        for _o in range(5):
            n = int(rng.integers(3, 600))
            V = np.array([rng.uniform(-4, 4), rng.uniform(-3, 3), rng.uniform(-8, 2)]) + rng.normal(0, 0.6, (3 * n, 3))
            F = np.arange(3 * n).reshape(-1, 3)
            a = rng.uniform(0, 2 * np.pi)
            M = np.eye(4)
            M[:3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
            M[:3, 3] = rng.normal(0, 0.5, 3)
            local = (V - M[:3, 3]) @ M[:3, :3]                    # M @ local = V
            piece = CL.Piece(local, F, size=48)
            items.append(((local.min(0), local.max(0)), M, lambda piece=piece, local=local: (piece, local)))
            every.append(V[F])
        assert CL.least_depth(items, VIEW) == pytest.approx(CL.tri_depth(np.concatenate(every), VIEW).min(), abs=1e-12)


def test_a_chunk_box_holds_its_triangles_on_any_frame():
    rng = np.random.default_rng(1)
    V, F = rng.normal(0, 1, (900, 3)), rng.integers(0, 900, (1200, 3))
    piece = CL.Piece(V, F, size=64)
    moved = V + rng.normal(0, 0.3, V.shape)                        # a deformed frame: the chunks stay, the boxes follow
    lo, hi = piece.boxes(moved)
    for c, idx in enumerate(piece.chunks):
        P = moved[F[idx]].reshape(-1, 3)
        assert (P >= lo[c] - 1e-12).all() and (P <= hi[c] + 1e-12).all()
    assert sorted(np.concatenate(piece.chunks).tolist()) == list(range(len(F)))


def test_the_camera_starts_half_way_to_the_nearest_geometry_never_at_the_lens():
    assert CL.near(4.0) == 2.0 and CL.near(1e-5) == CL.NEAR_MIN and CL.near(np.inf) == CL.NEAR_NONE
