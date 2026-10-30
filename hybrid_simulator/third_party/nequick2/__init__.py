"""NeQuick2 density provider backed by the bundled official Fortran source."""
from pathlib import Path
import shutil

def create_model(run_directory, flux_f107=150.0):
    """Load a private engine copy; coefficient/scratch I/O stays in Result/.

    The bundled .so is a Linux artifact from the previously used environment.
    For another platform build the C wrapper with tools/build_nequick.py.
    Importing this package alone does not load a shared library or CUDA.
    """
    from .model import NeQuick2ElectronDensity
    root=Path(__file__).resolve().parents[2]
    target=Path(run_directory).resolve()/'nequick_engine'
    if not target.is_relative_to((root/'results').resolve()):
        raise ValueError('NeQuick2 runtime must be inside this workspace results/')
    target.mkdir(parents=True,exist_ok=True)
    vendor=Path(__file__).parent/'vendor'
    names=['libnequick_p531.so','R12.dat','modip.asc']
    names += [p.name for p in vendor.glob('ccir*.asc')]
    for name in names:
        shutil.copy2(vendor/name,target/name)
    return NeQuick2ElectronDensity(target/'libnequick_p531.so',target,flux_f107)
