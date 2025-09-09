# ellipsoid_distance.py
# Pure-Python WGS-84 Vincenty inverse distance with rich debug for narration.
# References:
# - WGS-84 parameters (a, f) per NGA / Wikipedia. 
# - Vincenty inverse (1975) algorithm & known failure modes near antipodes.
# - Karney (2011/2013) notes on convergence (for explanation in text).
# See accompanying README comments in your generator for citations.

import math

WGS84_A = 6378137.0                # semi-major axis [m]
WGS84_F = 1 / 298.257223563        # flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)

def _vincenty_inverse_wgs84(lat1_deg, lon1_deg, lat2_deg, lon2_deg,
                            max_iter=200, tol=1e-12):
    """
    Vincenty inverse on WGS-84 ellipsoid.
    Returns (distance_km, debug_dict, converged_bool).
    """
    # Coincident points
    if (lat1_deg == lat2_deg) and (lon1_deg == lon2_deg):
        return 0.0, {
            "a": WGS84_A, "f": WGS84_F, "b": WGS84_B,
            "iterations": 0, "converged": True,
            "note": "coincident"
        }, True

    φ1 = math.radians(lat1_deg); φ2 = math.radians(lat2_deg)
    L = math.radians(lon2_deg - lon1_deg)  # difference in longitude (radians)

    # Reduced latitudes
    U1 = math.atan((1.0 - WGS84_F) * math.tan(φ1))
    U2 = math.atan((1.0 - WGS84_F) * math.tan(φ2))
    sinU1, cosU1 = math.sin(U1), math.cos(U1)
    sinU2, cosU2 = math.sin(U2), math.cos(U2)

    λ = L
    prev = None
    iters = 0

    sinσ = cosσ = σ = sinα = cos2α = cos2σm = 0.0

    for iters in range(1, max_iter + 1):
        sinλ, cosλ = math.sin(λ), math.cos(λ)
        sinσ = math.sqrt((cosU2 * sinλ) ** 2 +
                         (cosU1 * sinU2 - sinU1 * cosU2 * cosλ) ** 2)
        if sinσ == 0.0:
            # coincident or numerically identical
            return 0.0, {
                "a": WGS84_A, "f": WGS84_F, "b": WGS84_B,
                "iterations": iters, "converged": True,
                "note": "sinσ==0"
            }, True

        cosσ = sinU1 * sinU2 + cosU1 * cosU2 * cosλ
        σ = math.atan2(sinσ, cosσ)
        sinα = (cosU1 * cosU2 * sinλ) / max(1e-16, sinσ)
        cos2α = 1.0 - sinα * sinα

        if cos2α != 0.0:
            cos2σm = cosσ - 2.0 * sinU1 * sinU2 / cos2α
        else:
            cos2σm = 0.0  # equatorial line

        C = (WGS84_F / 16.0) * cos2α * (4.0 + WGS84_F * (4.0 - 3.0 * cos2α))
        λ_next = L + (1.0 - C) * WGS84_F * sinα * (
            σ + C * sinσ * (
                cos2σm + C * cosσ * (-1.0 + 2.0 * (cos2σm ** 2))
            )
        )
        if prev is not None and abs(λ_next - λ) < tol:
            λ = λ_next
            break
        prev = λ
        λ = λ_next
    else:
        # No convergence
        return None, {
            "a": WGS84_A, "f": WGS84_F, "b": WGS84_B,
            "iterations": iters, "converged": False,
            "U1": U1, "U2": U2, "L": L
        }, False

    u2 = cos2α * ((WGS84_A * WGS84_A - WGS84_B * WGS84_B) / (WGS84_B * WGS84_B))
    A = 1.0 + (u2 / 16384.0) * (4096.0 + u2 * (-768.0 + u2 * (320.0 - 175.0 * u2)))
    B = (u2 / 1024.0) * (256.0 + u2 * (-128.0 + u2 * (74.0 - 47.0 * u2)))
    Δσ = B * sinσ * (
        cos2σm + (B / 4.0) * (cosσ * (-1.0 + 2.0 * (cos2σm ** 2)) -
        (B / 6.0) * cos2σm * (-3.0 + 4.0 * (sinσ ** 2)) * (-3.0 + 4.0 * (cos2σm ** 2)))
    )

    s_m = WGS84_B * A * (σ - Δσ)  # meters
    d_km = s_m / 1000.0

    debug = {
        "a": WGS84_A, "f": WGS84_F, "b": WGS84_B,
        "U1": U1, "U2": U2, "L": L,
        "lambda_final": λ, "iterations": iters, "converged": True,
        "sin_sigma": sinσ, "cos_sigma": cosσ, "sigma": σ,
        "sin_alpha": sinα, "cos2_alpha": cos2α, "cos2_sigma_m": cos2σm,
        "C": C, "u2": u2, "A": A, "B": B, "Delta_sigma": Δσ,
        "s_m": s_m
    }
    return d_km, debug, True

def vincenty_distance_wgs84_km(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    """
    Public entry-point: stable Vincenty distance on WGS-84.
    If the iteration doesn't converge (very rare), we return None for distance_km
    and a debug dict noting converged=False so callers can decide fallback.
    """
    return _vincenty_inverse_wgs84(lat1_deg, lon1_deg, lat2_deg, lon2_deg)

