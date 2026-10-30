"""Group propagation time and its explicit geometric/medium decomposition."""
from dataclasses import dataclass
import numpy as np

from .constants import C0_M_S, FIRST_ORDER_RANGE_COEFFICIENT
from .refractive_index import group_refractive_index
from .tec import sample_polyline_midpoints


@dataclass(frozen=True)
class GroupDelayResult:
    straight_distance_m: float
    refracted_path_length_m: float
    straight_vacuum_time_s: float
    refracted_vacuum_time_s: float
    geometric_bending_delay_s: float
    medium_excess_group_delay_s: float
    total_group_time_s: float
    total_arrival_difference_s: float
    analytical_medium_delay_s: float


def itu_first_order_group_delay(stec_e_m2, frequency_hz):
    """First-order analytical delay K*STEC/(c*f^2), equivalent to P.531 Eq. (6)."""
    return FIRST_ORDER_RANGE_COEFFICIENT*np.asarray(stec_e_m2)/(C0_M_S*np.asarray(frequency_hz)**2)


def compute_group_delay(vertices_ecef_m, tx_ecef_m, rx_ecef_m, frequency_hz, model, time=None):
    mids, ds = sample_polyline_midpoints(vertices_ecef_m)
    ne = model.electron_density_ecef(mids, time)
    ng = group_refractive_index(ne, frequency_hz)
    length = float(ds.sum()); straight = float(np.linalg.norm(np.asarray(rx_ecef_m)-np.asarray(tx_ecef_m)))
    medium = float(np.sum((ng-1.0)*ds)/C0_M_S)
    stec = float(np.sum(np.asarray(ne)*ds))
    bend = (length-straight)/C0_M_S
    return GroupDelayResult(straight, length, straight/C0_M_S, length/C0_M_S, bend,
        medium, length/C0_M_S+medium, bend+medium,
        float(itu_first_order_group_delay(stec, frequency_hz)))

