"""Physical constants used by the P.531 implementation (SI units)."""

C0_M_S = 299_792_458.0
ELEMENTARY_CHARGE_C = 1.602_177_33e-19  # ITU-R P.531-16 section 4.2.1
ELECTRON_MASS_KG = 9.109_389_7e-31     # ITU-R P.531-16 section 4.2.1
VACUUM_PERMITTIVITY_F_M = 8.854_187_818e-12
TECU_E_M2 = 1.0e16

# Derived rather than hard-coded in the physics routines. This is close to 40.3.
FIRST_ORDER_RANGE_COEFFICIENT = (
    ELEMENTARY_CHARGE_C**2
    / (8.0 * 3.141592653589793**2 * VACUUM_PERMITTIVITY_F_M * ELECTRON_MASS_KG)
)

