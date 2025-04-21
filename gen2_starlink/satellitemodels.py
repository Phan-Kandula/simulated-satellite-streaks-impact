"""Building a series of surfaces that make up the Gen 2 Starlink satellite model.

Usage: For the most part, this file is imported by the code that does the analysis and
doesn't need to be interacted with. By default, the Gen 2 Starlink satellite model uses
the ABG reflection model for the Chassis. However, a special interpolation code is
implemented for more accuracy if necessary. The interpolation code takes much longer to
execute, and therefore note recommended for the 10-year survey analysis. While the AGB
model is less accurate, the difference is minimal for the 10-year survey.
"""

import lumos
import numpy as np
import pandas as pd
from lumos.brdf.library import BINOMIAL, LAMBERTIAN, PHONG
from lumos.geometry import Surface

al = pd.read_csv("chassis_bare.csv", sep=" ")


def convert_to_cartesian(
    phi_i: float, theta_i: float, phi_o: float, theta_o: float
) -> tuple[float, float, float, float, float, float]:
    """Convert the incoming and reflected light rays in spherical coordinates to cartesian coordinates.

    Parameters
    ----------
        phi_i (float): the azimuthal angle of the incoming light ray in radians
        theta_i (float): the polar angle of the incoming light ray in radians
        phi_o (float): the azimuthal angle of the outgoing light ray in radians
        theta_o (float): the polar angle of the outgoing light ray in radians

    Returns
    -------
        Returns a tuple that represents the incoming and outgoing light ray vectors in
        cartesian coordinates: (ix, iy, iz, ox, oy, oz).
        - ix, iy, iz: components of the incoming light ray vector
        - ox, oy, oz: components of the outgoing light ray vector

    """
    ix, iy, iz = lumos.conversions.spherical_to_unit(
        np.deg2rad(phi_i), np.deg2rad(theta_i)
    )
    ox, oy, oz = lumos.conversions.spherical_to_unit(
        np.deg2rad(phi_o), np.deg2rad(theta_o)
    )

    return ix, iy, iz, ox, oy, oz


def get_distance_between_projections(
    phi_i: float,
    theta_i: float,
    phi_o: float,
    theta_o: float,
    normal: (float, float, float),
) -> float:
    """Get the distance between of the outgoing light ray and the reflection of the incoming light ray.

    Parameters
    ----------
        phi_i (float): the azimuthal angle of the incoming light ray in radians
        theta_i (float): the polar angle of the incoming light ray in radians
        phi_o (float): the azimuthal angle of the outgoing light ray in radians
        theta_o (float): the polar angle of the outgoing light ray in radians
        normal (float, float, float): The normal vector of the surface

    Returns
    -------
        Returns the euclidean distance between the specularly reflected light ray, which
        is the reflection of the incoming light ray, and the outgoing light ray.

    """
    ix, iy, iz, ox, oy, oz = convert_to_cartesian(phi_i, theta_i, phi_o, theta_o)

    nx, ny, nz = normal

    dot = ix * nx + iy * ny + iz * nz
    rx = 2 * dot * nx - ix
    ry = 2 * dot * ny - iy
    rz = 2 * dot * nz - iz

    dot = ox * nx + oy * ny + oz * nz
    rho_x = ox - dot * nx
    rho_y = oy - dot * ny
    rho_z = oz - dot * nz

    dot = rx * nx + ry * ny + rz * nz
    rho0_x = rx - dot * nx
    rho0_y = ry - dot * ny
    rho0_z = rz - dot * nz

    return np.sqrt(
        (rho_x - rho0_x) ** 2 + (rho_y - rho0_y) ** 2 + (rho_z - rho0_z) ** 2
    )


def idw_weight(phi_i_data: float, phi_i: float, scale: float = 2) -> float:
    """Calculate the weight of data point for interpolation based on the distance between projections.

    Makes sure the BRDF data near the points that need to interpolated are weighted more than far away points

    Parameters
    ----------
        phi_i_data (float): The azimuthal angle of the incoming light ray from the
            BRDF table used for interpolation.
        phi_i (float): The azimuthal angle of the incoming light ray you want to
            simulate
        scale (float): The scaling factor for weight.

    Returns
    -------
        The weight of each BRDF measurement.

    """
    if isinstance(phi_i, np.ndarray):
        phi_is = np.zeros(len(phi_i)) + phi_i_data
    else:
        phi_is = phi_i_data
    distance = np.abs(np.power(phi_is, scale) - np.power(np.array(phi_i), scale))
    return 1 / np.where(distance == 0, 1, distance)


def idw_func(
    phi_i: float,
    theta_i: float,
    phi_o: float,
    theta_o: float,
    normal: (float, float, float),
) -> float:
    """Interpolate BRDF using inverse distance weights method for the aluminum chassis.

    Parameters
    ----------
        phi_i (float): the azimuthal angle of the incoming light ray in radians
        theta_i (float): the polar angle of the incoming light ray in radians
        phi_o (float): the azimuthal angle of the outgoing light ray in radians
        theta_o (float): the polar angle of the outgoing light ray in radians
        normal (tuple(float, float, float)): The normal vector of the surface

    Returns
    -------
        The BRDF of the aluminum chassis for the given incoming and outgoing angles.
    """
    D = get_distance_between_projections(phi_i, theta_i, phi_o, theta_o, normal)

    abs_05 = np.abs(np.array(D_05)[:, np.newaxis] - np.array(D))
    index_05 = np.argmin(abs_05, axis=0)

    abs_41 = np.abs(np.array(D_41)[:, np.newaxis] - np.array(D))
    index_41 = np.argmin(abs_41, axis=0)

    abs_60 = np.abs(np.array(D_60)[:, np.newaxis] - np.array(D))
    index_60 = np.argmin(abs_60, axis=0)

    abs_75 = np.abs(np.array(D_75)[:, np.newaxis] - np.array(D))
    index_75 = np.argmin(abs_75, axis=0)

    d_i_05 = idw_weight(5, phi_i, 2)
    d_i_41 = idw_weight(41.4, phi_i, 2)
    d_i_60 = idw_weight(60, phi_i, 2)
    d_i_75 = idw_weight(75, phi_i, 2)

    D = np.array([D])
    z = np.zeros(len(D))
    if isinstance(d_i_05, np.ndarray):
        for i in range(len(z)):
            numerator = np.sum(
                [
                    np.array(al.loc[al["phi_i"] == 5]["brdf"])[index_05[i]] * d_i_05[i],
                    np.array(al.loc[al["phi_i"] == 41.4]["brdf"])[index_41[i]]
                    * d_i_41[i],
                    np.array(al.loc[al["phi_i"] == 60]["brdf"])[index_60[i]]
                    * d_i_60[i],
                    np.array(al.loc[al["phi_i"] == 75]["brdf"])[index_75[i]]
                    * d_i_75[i],
                ]
            )

            denominator = np.sum([d_i_05[i], d_i_41[i], d_i_60[i], d_i_75[i]])

            z[i] = numerator / denominator

        return z

    for i in range(len(z)):
        numerator = np.sum(
            [
                np.array(al.loc[al["phi_i"] == 5]["brdf"])[index_05[0]] * d_i_05,
                np.array(al.loc[al["phi_i"] == 41.4]["brdf"])[index_41[0]] * d_i_41,
                np.array(al.loc[al["phi_i"] == 60]["brdf"])[index_60[0]] * d_i_60,
                np.array(al.loc[al["phi_i"] == 75]["brdf"])[index_75[0]] * d_i_75,
            ]
        )

        denominator = np.sum([d_i_05, d_i_41, d_i_60, d_i_75])

        z[i] = numerator / denominator

    return z[0]


def aluminum_brdf(
    w_i: (float, float, float),
    normal: (float, float, float),
    w_o: (float, float, float),
) -> float:
    """Call the idw function to get the brdf of the aluminum chassis.

    Parameters
    ----------
        w_i (float, float, float): Incoming light ray in cartesian coordinates
        normal (float, float, float): Normal vector of the surface in cartesian
            coordinates
        w_o (float, float, float): Outgoing light ray in cartesian coordinates

    Returns
    -------
        Returns the BRDF of the aluminum chassis based on the incoming and outgoing
        light rays

    """
    ix, iy, iz = w_i
    ox, oy, oz = w_o
    phi_i, theta_i = lumos.conversions.unit_to_spherical(ix, iy, iz)
    phi_o, theta_o = lumos.conversions.unit_to_spherical(ox, oy, oz)

    return idw_func(phi_i, theta_i, phi_o, theta_o, normal)


D_05 = get_distance_between_projections(
    al.loc[al["phi_i"] == 5]["phi_i"],
    al.loc[al["phi_i"] == 5]["theta_i"],
    al.loc[al["phi_i"] == 5]["phi_o"],
    al.loc[al["phi_i"] == 5]["theta_o"],
    (0, 0, 1),
)

D_41 = get_distance_between_projections(
    al.loc[al["phi_i"] == 41.4]["phi_i"],
    al.loc[al["phi_i"] == 41.4]["theta_i"],
    al.loc[al["phi_i"] == 41.4]["phi_o"],
    al.loc[al["phi_i"] == 41.4]["theta_o"],
    (0, 0, 1),
)

D_60 = get_distance_between_projections(
    al.loc[al["phi_i"] == 60]["phi_i"],
    al.loc[al["phi_i"] == 60]["theta_i"],
    al.loc[al["phi_i"] == 60]["phi_o"],
    al.loc[al["phi_i"] == 60]["theta_o"],
    (0, 0, 1),
)

D_75 = get_distance_between_projections(
    al.loc[al["phi_i"] == 75]["phi_i"],
    al.loc[al["phi_i"] == 75]["theta_i"],
    al.loc[al["phi_i"] == 75]["phi_o"],
    al.loc[al["phi_i"] == 75]["theta_o"],
    (0, 0, 1),
)


def get_surfaces() -> list:
    """Get the list of surfaces that make up the satellite.

    The surface object contain brightness properties that LUMOS uses to calculate BRDF.
    There are two different reflection models that can be used to build the chassis of
    the satellite: the ABG model and the interpolation method coded above. The boolean
    use_abg_chassis can be toggled for the ABG model or interpolation method. The ABG
    model is very fast but not accurate as the slower interpolation method.

    Returns:
        Returns the list of surfaces.

    """
    use_abg_chassis = False

    # Constants
    chassis_area = 12  # m^2
    mirror_area = 0.975
    chassis_mirror_area = chassis_area * mirror_area
    chassis_al_area = chassis_area * (1 - mirror_area)
    pantograph_area = 1.03  # m^2
    boom_area = 2.2
    dishes_ka_area = 0.66
    dishes_kae_area = 0.19
    twist_area = 0.6

    chassis_normal = np.array([0, 0, -1])
    pantograph_normal = np.array([0, np.sqrt(1 / 2), -np.sqrt(1 / 2)])

    B = np.array([[2.262, -75.191]])
    C = np.array(
        [
            [
                999.894,
                997.658,
                986.822,
                389.102,
                -1000,
                929.676,
                -420.03,
                254.474,
                -96.152,
                43.231,
            ]
        ]
    )
    lab_chassis_brdf = BINOMIAL(B, C, d=3.0, l1=-5)

    B = np.array([[1.078, -32.658]])
    C = np.array([[10000.0, -6040.756, 1392.37, -69.043, -52.863, 52.601]])
    lab_mirror_brdf = BINOMIAL(B, C, d=3.0, l1=-3)

    B = np.array([[3.75, -84.078]])
    C = np.array(
        [
            [
                -1000.0,
                -1000.0,
                966.661,
                1000.0,
                -444.015,
                381.728,
                -233.578,
                273.263,
                -150.157,
                67.284,
            ]
        ]
    )
    lab_al_bare_brdf = BINOMIAL(B, C, d=3.0, l1=-5)

    B = np.array([[-1.473, 0.752], [-2.331, 1.024], [5.148, -1.672]])
    C = np.array(
        [
            [
                -6.810e03,
                -9.991e03,
                -9.525e03,
                3.044e03,
                1.819e03,
                -7.336e02,
                1.606e02,
                -4.217e01,
                1.384e01,
                -3.637e00,
            ],
            [
                8.267e03,
                9.423e03,
                1.000e04,
                1.430e03,
                -9.889e03,
                6.254e03,
                -2.682e03,
                1.078e03,
                -3.899e02,
                8.700e01,
            ],
            [
                -6.625e03,
                -9.465e03,
                -1.803e03,
                7.089e03,
                -2.907e02,
                -3.307e03,
                2.760e03,
                -1.458e03,
                5.716e02,
                -1.287e02,
            ],
        ]
    )
    lab_c138_brdf = BINOMIAL(B, C, d=3.0, l1=-5)

    B = np.array([[-2.014, 3.935], [0.906, -15.296], [-0.196, 13.039]])
    C = np.array(
        [
            [1568.713, -6957.513, 4197.732, -1078.46, 191.648, -29.161],
            [-7807.871, 6352.045, -4865.936, 2780.032, -896.354, 169.988],
            [10000.0, -8481.038, 3785.344, -2006.663, 871.593, -196.501],
        ]
    )
    lab_lrb_brdf = BINOMIAL(B, C, d=3.0, l1=-3)

    B = np.array([[-0.365, -4.975]])
    C = np.array([[1.746, 5.717, 6.474, 1.831, 4.67, 1.163]])
    lab_a700_brdf = BINOMIAL(B, C, d=1.0, l1=-3)

    A = 0.019989862079564824
    B = 2.2851243807209376e-06
    G = 2.2640286229702555
    lab_al_abg_brdf = lumos.brdf.library.ABG(A, B, G)

    phong_offset_chassis = lumos.brdf.library.PHONG(0.056, 0, 0.194)

    SURFACES_LAB_BRDFS = [
        Surface(pantograph_area, pantograph_normal, lab_c138_brdf),
        Surface(chassis_mirror_area, chassis_normal, lab_mirror_brdf),
        Surface(
            boom_area,
            chassis_normal,
            lumos.brdf.library.ABG(9.737e-05, 8.448e-06, 2.686),
        ),
        Surface(dishes_ka_area, chassis_normal, lab_lrb_brdf),
        Surface(dishes_kae_area, chassis_normal, lab_lrb_brdf),
        Surface(twist_area, chassis_normal, lab_a700_brdf),
        Surface(1, chassis_normal, phong_offset_chassis),
    ]

    if use_abg_chassis:
        SURFACES_LAB_BRDFS.append(
            Surface(chassis_al_area, chassis_normal, lab_al_abg_brdf)
        )
        print("using ABG chassis")
    else:
        SURFACES_LAB_BRDFS.append(
            Surface(chassis_al_area, chassis_normal, aluminum_brdf)
        )
        print("Using interpolated chassis")

    return SURFACES_LAB_BRDFS
