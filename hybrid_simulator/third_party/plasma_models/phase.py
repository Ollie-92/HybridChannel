"""Carrier phase accumulated with the P.531 phase index."""
from dataclasses import dataclass
import numpy as np

from .constants import C0_M_S, FIRST_ORDER_RANGE_COEFFICIENT
from .refractive_index import phase_refractive_index
from .tec import sample_polyline_midpoints


@dataclass(frozen=True)
class PhaseResult:
    total_phase_rad: float
    vacuum_same_path_phase_rad: float
    medium_phase_advance_rad: float
    analytical_phase_advance_rad: float


def itu_first_order_phase_advance(stec_e_m2, frequency_hz):
    """Signed first-order phase contribution [rad], negative means physical advance."""
    return -2*np.pi*FIRST_ORDER_RANGE_COEFFICIENT*np.asarray(stec_e_m2)/(C0_M_S*np.asarray(frequency_hz))


def compute_phase_advance(vertices_ecef_m, frequency_hz, model, time=None):
    mids, ds = sample_polyline_midpoints(vertices_ecef_m)
    ne = np.asarray(model.electron_density_ecef(mids, time))
    np_idx = phase_refractive_index(ne, frequency_hz)
    k0 = 2*np.pi*frequency_hz/C0_M_S
    vacuum = float(k0*ds.sum()); medium_delta = float(k0*np.sum((np_idx-1.0)*ds))
    stec = float(np.sum(ne*ds))
    return PhaseResult(vacuum+medium_delta, vacuum, medium_delta,
                       float(itu_first_order_phase_advance(stec, frequency_hz)))

