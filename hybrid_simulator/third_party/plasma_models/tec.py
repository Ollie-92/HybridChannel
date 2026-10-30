"""Path sampling and slant TEC integration (ITU-R P.531-16 Eq. 1)."""
import numpy as np


def sample_polyline_midpoints(vertices_ecef_m):
    v = np.asarray(vertices_ecef_m, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) < 2:
        raise ValueError("vertices_ecef_m must have shape [N>=2,3]")
    delta = np.diff(v, axis=0); ds = np.linalg.norm(delta, axis=1)
    keep = ds > 0
    if not np.any(keep): raise ValueError("path has zero length")
    return .5*(v[:-1]+v[1:])[keep], ds[keep]


def integrate_tec(vertices_ecef_m, model, time=None):
    """Return integral Ne ds [electrons/m^2] along the supplied actual path."""
    mids, ds = sample_polyline_midpoints(vertices_ecef_m)
    ne = np.asarray(model.electron_density_ecef(mids, time), dtype=float)
    if ne.shape != ds.shape or np.any(ne < 0) or np.any(~np.isfinite(ne)):
        raise ValueError("invalid electron-density samples")
    return float(np.sum(ne*ds))

