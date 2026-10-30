"""ITU-R P.531-16 phase and group refractive indices."""
import numpy as np

from .constants import ELEMENTARY_CHARGE_C, ELECTRON_MASS_KG, VACUUM_PERMITTIVITY_F_M


def plasma_parameter(ne_m3, frequency_hz):
    """Return X=omega_p^2/omega^2 (dimensionless), using SI quantities."""
    ne = np.asarray(ne_m3, dtype=float)
    f = np.asarray(frequency_hz, dtype=float)
    if np.any(ne < 0) or np.any(~np.isfinite(ne)) or np.any(f <= 0) or np.any(~np.isfinite(f)):
        raise ValueError("Ne must be non-negative and frequency positive")
    omega = 2.0*np.pi*f
    return ELEMENTARY_CHARGE_C**2*ne/(VACUUM_PERMITTIVITY_F_M*ELECTRON_MASS_KG*omega**2)


def phase_refractive_index(ne_m3, frequency_hz):
    """Phase index np=sqrt(1-X), ITU-R P.531-16 Eq. (2)."""
    x = plasma_parameter(ne_m3, frequency_hz)
    if np.any(x >= 1.0):
        raise ValueError("frequency is at or below local plasma cutoff (X >= 1)")
    return np.sqrt(1.0-x)


def group_refractive_index(ne_m3, frequency_hz):
    """Group index ng=1/np, ITU-R P.531-16 Eq. (3)."""
    return 1.0/phase_refractive_index(ne_m3, frequency_hz)


def phase_index_gradient_ecef(position_ecef_m, frequency_hz, model, time=None,
                              difference_step_m=1000.0):
    """Central-difference gradient of np [1/m]; numerical GO support, not an ITU equation."""
    r = np.asarray(position_ecef_m, dtype=float)
    h = float(difference_step_m)
    if r.shape != (3,) or h <= 0 or not np.isfinite(h):
        raise ValueError("position must be [3] and difference_step_m positive")
    eye = np.eye(3)*h
    plus = model.electron_density_ecef(r+eye, time)
    minus = model.electron_density_ecef(r-eye, time)
    return (phase_refractive_index(plus, frequency_hz)-phase_refractive_index(minus, frequency_hz))/(2*h)

