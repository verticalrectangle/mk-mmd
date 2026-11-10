"""The strap solver (mkmmd.solvers.strap): a band held by a prop, carried by its wearer and hanging between the two."""
import numpy as np

from mkmmd.solvers import strap as ST


def test_a_free_run_hangs_rests_on_her_follows_its_ends_and_settles():
    n, F = 21, 75
    rest = np.stack([np.linspace(0.0, 0.4, n), np.zeros(n), np.ones(n)], 1)      # a level run, 2 cm between points
    kind = np.array([0] + [2] * 12 + [1] * 7 + [0])                              # a button, a free run, on her, a button
    held = np.tile(rest, (F, 1, 1))
    held[30:, :13, 1] += 0.03 * np.minimum((np.arange(30, F)[:, None] - 30) / 6.0, 1.0)   # the first button swings 3 cm
    carried = np.tile(rest, (F, 1, 1))
    faces = np.tile([0.0, 0.0, 1.0], (F, n, 1))
    under = np.full((F, n, 3), np.nan)
    under[:, 5:9] = rest[5:9] - [0.0, 0.0, 0.006]                                # her, 6 mm under the middle of the run
    carried[:, 5:9, 2] += 0.02                                                   # (the way out of her there: up)
    x, facing, rep = ST.simulate(rest, kind, held, carried, faces, faces, under, slack=0.01, standoff=0.004)
    assert np.allclose(x[:, [0, 20]], held[:, [0, 20]]) and np.allclose(x[:, 13:20], rest[13:20])   # where they are taken
    assert np.allclose(np.linalg.norm(facing, axis=2), 1.0)
    assert 0.0015 < 1.0 - x[29, 1:13, 2].min() < 0.03                            # the run hangs, a shallow curve,
    assert x[:, 5:9, 2].min() > 1.0 - 0.006 + 0.004 - 1e-4                       # lies on her, a standoff off,
    assert abs(x[-1, 6, 1] - 0.03 * (1.0 - 6 / 13.0)) < 0.006                    # goes with its end
    assert np.abs(x[-1] - x[-2]).max() < 1e-4                                    # and comes to rest
    assert rep["runs"] == 1 and rep["free"] == 12 and rep["stretch_pct_max"] < 4.0
