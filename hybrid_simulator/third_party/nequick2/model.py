"""Python adapter for the unmodified official ITU P.531 NeQuick2 source."""
from dataclasses import dataclass
import ctypes
import os
from pathlib import Path
import threading

import numpy as np
from pyproj import Transformer

_LOCK = threading.Lock()


@dataclass
class NeQuick2ElectronDensity:
    library_path: Path
    data_directory: Path
    flux_f107: float = 193.0

    def __post_init__(self):
        self.library_path = Path(self.library_path).resolve()
        self.data_directory = Path(self.data_directory).resolve()
        if not 63.0 <= self.flux_f107 <= 193.0:
            raise ValueError("official NeQuick2 requires F10.7 in [63,193]")
        self._lib = ctypes.CDLL(str(self.library_path))
        self._fn = self._lib.nequick_eval_c
        self._fn.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double,
                             ctypes.c_int, ctypes.c_double, ctypes.c_double]
        self._fn.restype = ctypes.c_double
        self._to_geo = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)

    def electron_density_ecef(self, position_ecef_m, time=None):
        p = np.asarray(position_ecef_m, dtype=float)
        scalar = p.shape == (3,)
        if scalar: p = p[None,:]
        if p.ndim != 2 or p.shape[1] != 3 or time is None:
            raise ValueError("positions must be [3] or [N,3] and UTC is required")
        lon, lat, alt = self._to_geo.transform(p[:,0],p[:,1],p[:,2])
        hour = time.hour + time.minute/60 + (time.second+time.microsecond/1e6)/3600
        out = np.empty(len(p))
        with _LOCK:
            previous = Path.cwd()
            try:
                os.chdir(self.data_directory)
                for i in range(len(p)):
                    out[i] = self._fn(alt[i]/1000,lat[i],lon[i],time.month,
                                      self.flux_f107,hour)
            finally:
                os.chdir(previous)
        if np.any(~np.isfinite(out)) or np.any(out < 0):
            raise ValueError("NeQuick2 returned invalid density")
        return float(out[0]) if scalar else out


@dataclass(frozen=True)
class ScaledElectronDensity:
    """Explicit Table-3-only scaling of the NeQuick profile to target VTEC."""
    base: object
    scale: float

    def electron_density_ecef(self, position_ecef_m, time=None):
        return self.scale*np.asarray(self.base.electron_density_ecef(position_ecef_m,time))
