"""The drawings of the picture-book parent cut-outs (pure Python + numpy, no bpy: importable from tests): the control points
of every paper piece and detail, in the figure's own frame, metres: x across (+x the figure's left, the viewer's right),
z up from the soles (z = 0); the card's frame lifts it by the white border (parent_cutout_layout).

Proportions from photographs of 1960s families and clothes (men in narrow-lapel suits, slim ties and short side-parted
hair; women in sleeveless A-line shifts with bouffant and flip hairdos), at about 1.8 times life size: the father 3.21 m
to his crown, 7.2 heads tall (the chin at 2.77, shoulders 0.93 m across, the crotch at 1.47, a jacket covering the seat to
1.35), the mother 3.09 m to the top of her bouffant (her crown at about 2.98, 7.4 heads, the dress hemmed above the knee at
1.00). Each outline is a closed centripetal Catmull-Rom spline through its points (`corners` are sharp: scissors turn
there); a stroke is a limb drawn along a centre line with its half widths.

    FIGURES[figure] = Figure(pieces, head, hand)
      pieces    {piece: Piece(points, corners, stroke, details, joint)}: "body" (joint None) first, then the pieces on
                paper-fastener brads at `joint` (x, z): the father's "hand" (his left forearm and the scolding hand, turning
                about the brad at his elbow), the mother's "arm_L" / "arm_R" (static, on brads at her shoulders). A piece is
                an outline (`points`, `corners`) or a `stroke` (centre points, half widths)
      head      (x, z) of the middle of the face (the card's `look` point)
      hand      the father's Hand(pivot, tip): the elbow's brad and the index fingertip as drawn (angle 0); else None
    Detail(role, level, points, corners): a paper layer glued on its piece, `level` 1 (on the paper) .. 4 (on top)."""
from collections import namedtuple

import numpy as np

Piece = namedtuple("Piece", "points corners stroke details joint")
Detail = namedtuple("Detail", "role level points corners")
Hand = namedtuple("Hand", "pivot tip")
Figure = namedtuple("Figure", "pieces head hand")


def _arc(cx, cz, r, a0, a1, n):
    """n points on a circle about (cx, cz) from a0 to a1 degrees (both ends)."""
    a = np.radians(np.linspace(a0, a1, n))
    return [(cx + r * np.cos(t), cz + r * np.sin(t)) for t in a]


def _mirror(points):
    """The points mirrored across x = 0, in reverse order (a mirrored outline keeps its winding)."""
    return [(-x, z) for x, z in reversed(points)]


def _circle(cx, cz, r, n=10):
    return _arc(cx, cz, r, 0.0, 360.0 * (n - 1) / n, n)


# ===================================================================================================================
# the father: a suit, slim tie, side-parted hair; his left forearm up on a brad at the elbow, the index finger raised
# ===================================================================================================================
ELBOW = (0.62, 2.02)                 # the brad at his left elbow (the hand piece turns about it)
SHOULDER = (0.37, 2.55)              # the left shoulder joint the upper arm hangs from, out to the elbow
UPPER_R = 0.095                      # the upper arm's half width and the round end round the brad


def _upper_arm():
    """The left upper arm's outline from the armpit down its inner edge, round the elbow's end and up its outer edge to the
    shoulder (the arm hangs 25 degrees out from the side)."""
    axis = np.subtract(ELBOW, SHOULDER)
    axis = axis / np.hypot(*axis)
    n = np.array([-axis[1], axis[0]])
    n = n if n[0] > 0 else -n                                     # the outer side
    a_out = np.degrees(np.arctan2(n[1], n[0]))
    armpit = np.array((0.338, 2.405))
    inner, outer = np.array(ELBOW) - UPPER_R * n, np.array(ELBOW) + UPPER_R * n
    top = np.array(SHOULDER) + UPPER_R * n
    return ([tuple(armpit)] + [tuple(0.5 * (armpit + inner) + 0.004 * n)] +
            _arc(ELBOW[0], ELBOW[1], UPPER_R, a_out - 180.0, a_out, 7) + [tuple(0.5 * (outer + top) + 0.008 * n)])


def _father_head():
    """The head's right half (+x) from the jaw up over the ear to the crown, without the crown point."""
    return [(0.104, 2.765), (0.128, 2.80), (0.140, 2.87), (0.155, 2.885), (0.172, 2.935), (0.170, 2.99), (0.152, 3.02),
            (0.150, 3.07), (0.132, 3.145), (0.080, 3.195)]


def _father_body():
    leg_in = [(0.032, 1.40), (0.048, 1.10), (0.062, 0.80), (0.080, 0.45), (0.092, 0.19)]
    shoe = [(0.084, 0.12), (0.074, 0.05), (0.10, 0.0), (0.25, -0.004), (0.355, 0.006), (0.405, 0.05), (0.392, 0.105),
            (0.34, 0.14)]
    leg_out = [(0.338, 0.185), (0.330, 0.45), (0.320, 0.85), (0.310, 1.15), (0.305, 1.325), (0.335, 1.345)]
    side = [(0.355, 1.375), (0.342, 1.58), (0.312, 1.86), (0.318, 2.10), (0.330, 2.32)]
    shoulder = [(0.465, 2.58), (0.445, 2.625), (0.30, 2.66), (0.17, 2.685), (0.105, 2.705)]
    right = leg_in + shoe + leg_out + side + _upper_arm() + shoulder + _father_head()      # crotch .. crown, +x side
    arm = [(-0.485, 2.47), (-0.505, 2.24), (-0.522, 1.98), (-0.532, 1.76), (-0.540, 1.565),        # sleeve, outer edge
           (-0.538, 1.50), (-0.552, 1.42), (-0.548, 1.30), (-0.522, 1.21), (-0.472, 1.178), (-0.428, 1.20),
           (-0.405, 1.27), (-0.393, 1.36), (-0.375, 1.43), (-0.390, 1.50),                          # the hanging hand
           (-0.393, 1.565), (-0.378, 1.76), (-0.360, 1.98), (-0.350, 2.20), (-0.338, 2.385)]        # sleeve, inner edge
    left = (_mirror(_father_head()) + _mirror(shoulder) + arm + _mirror(side) + _mirror(leg_out) + _mirror(shoe) +
            _mirror(leg_in))
    pts = [(0.0, 1.47)] + right + [(0.0, 3.21)] + left
    return pts, [0]


def _at_elbow(points):
    return [(ELBOW[0] + x, ELBOW[1] + z) for x, z in points]


def _father_hand():
    """The forearm and the scolding hand, drawn upright from the elbow's brad: a round end on the brad, the sleeve, the
    shirt cuff, then his left hand raised palm to the viewer: the index finger up from the fist's edge on the thumb side
    (toward his head, -x), the other three fingers curled and stacked down the far side (+x)."""
    return _at_elbow(
        _arc(0.0, 0.0, 0.09, -165.0, -15.0, 6) +
        [(0.092, 0.08), (0.088, 0.25), (0.080, 0.455), (0.078, 0.465), (0.074, 0.52), (0.080, 0.548),  # sleeve, cuff
         (0.100, 0.566), (0.108, 0.590), (0.100, 0.610), (0.110, 0.633), (0.103, 0.656), (0.110, 0.680),
         (0.098, 0.706), (0.070, 0.724), (0.036, 0.728), (0.004, 0.722),                           # the curled fingers
         (-0.020, 0.745), (-0.024, 0.800), (-0.028, 0.862), (-0.040, 0.898), (-0.058, 0.910), (-0.077, 0.900),
         (-0.088, 0.866), (-0.092, 0.800),                                                         # the index finger
         (-0.094, 0.730), (-0.100, 0.670), (-0.104, 0.618), (-0.096, 0.572), (-0.080, 0.546),       # the thumb's root
         (-0.078, 0.52), (-0.080, 0.465), (-0.082, 0.455), (-0.088, 0.25), (-0.092, 0.08)])


FATHER_TIP = _at_elbow([(-0.058, 0.910)])[0]      # the index fingertip as drawn (the top of the hand's outline)


def _father_details():
    hair = ([(0.155, 3.03), (0.157, 3.075), (0.138, 3.152), (0.084, 3.203), (0.0, 3.218), (-0.084, 3.203),
             (-0.138, 3.152), (-0.157, 3.075), (-0.156, 3.02), (-0.150, 2.965), (-0.136, 2.975), (-0.128, 3.035),
             (-0.085, 3.060), (-0.020, 3.072), (0.040, 3.096), (0.060, 3.080), (0.105, 3.070), (0.128, 3.042),
             (0.137, 2.978), (0.151, 2.968)], [])
    shirt = ([(-0.115, 2.715), (0.115, 2.715), (0.008, 2.155), (-0.008, 2.155)], [0, 1, 2, 3])
    lapel = [(0.105, 2.712), (0.176, 2.700), (0.198, 2.565), (0.168, 2.548), (0.214, 2.500), (0.170, 2.340),
             (0.060, 2.205), (0.022, 2.162), (0.006, 2.158), (0.008, 2.192), (0.030, 2.300), (0.068, 2.480)]
    lapels = [(lapel, [0, 2, 3, 4, 7, 8]), (_mirror(lapel), [3, 4, 7, 8, 9, 11])]
    tie = ([(-0.032, 2.672), (0.032, 2.672), (0.024, 2.615), (0.046, 2.275), (0.0, 2.228), (-0.046, 2.275),
            (-0.024, 2.615)], [0, 1, 2, 3, 4, 5, 6])
    collar = ([(-0.118, 2.700), (-0.106, 2.745), (0.0, 2.700), (0.106, 2.745), (0.118, 2.700), (0.046, 2.628),
               (0.0, 2.668), (-0.046, 2.628)], [0, 2, 4, 5, 6, 7])
    pocket = ([(0.150, 2.318), (0.252, 2.318), (0.246, 2.350), (0.226, 2.382), (0.205, 2.352), (0.186, 2.386),
               (0.160, 2.350)], [0, 1, 3, 4, 5])
    shoe = [(0.088, 0.142), (0.072, 0.06), (0.094, -0.006), (0.25, -0.010), (0.360, 0.000), (0.412, 0.050),
            (0.398, 0.112), (0.346, 0.150), (0.24, 0.158), (0.15, 0.154)]
    cuff = ([(-0.546, 1.500), (-0.544, 1.572), (-0.388, 1.572), (-0.384, 1.500)], [])
    buttons = [(_circle(0.0, 2.105, 0.022), []), (_circle(0.0, 1.955, 0.022), [])]
    return ([Detail("hair", 1, *hair), Detail("trim", 1, *shirt), Detail("detail", 2, *tie)] +
            [Detail("shade", 3, *lp) for lp in lapels] +
            [Detail("trim", 4, *collar), Detail("trim", 1, *pocket), Detail("trim", 1, *cuff),
             Detail("hair", 1, shoe, []), Detail("hair", 1, _mirror(shoe), [])] +
            [Detail("hair", 1, *b) for b in buttons])


def _father_hand_details():
    """The shirt cuff, and the thumb pressed across the curled fingers under the index (a darker paper)."""
    thumb = [(-0.108, 0.598), (-0.070, 0.596), (-0.020, 0.612), (0.030, 0.628), (0.062, 0.640), (0.074, 0.656),
             (0.062, 0.673), (0.020, 0.669), (-0.030, 0.661), (-0.080, 0.651), (-0.108, 0.640)]
    return [Detail("trim", 1, _at_elbow([(-0.088, 0.448), (0.088, 0.448), (0.084, 0.528), (-0.084, 0.528)]), []),
            Detail("shade", 1, _at_elbow(thumb), [])]


# ===================================================================================================================
# the mother: an A-line shift, a bouffant with a flip and a bow; arms akimbo, each a piece on a brad at the shoulder
# ===================================================================================================================
def _mother_half():
    """The body's right half (+x) from the leg gap at the hem round the shoe, up the dress, the shoulder, the neck and the
    hair to the top, without the end points on x = 0."""
    leg_in = [(0.062, 1.00), (0.070, 0.80), (0.066, 0.62), (0.076, 0.42), (0.098, 0.20), (0.106, 0.13)]
    shoe = [(0.088, 0.085), (0.082, 0.035), (0.106, 0.0), (0.195, -0.004), (0.247, 0.010), (0.265, 0.045),
            (0.243, 0.085), (0.187, 0.112)]
    leg_out = [(0.172, 0.16), (0.190, 0.32), (0.212, 0.50), (0.202, 0.70), (0.192, 0.82), (0.204, 0.95), (0.210, 1.005)]
    dress = [(0.415, 1.010), (0.442, 1.030), (0.432, 1.10), (0.400, 1.32), (0.358, 1.60), (0.312, 1.90),
             (0.276, 2.15), (0.272, 2.28), (0.292, 2.36), (0.318, 2.418), (0.306, 2.445), (0.200, 2.462),
             (0.125, 2.470)]
    neck = [(0.070, 2.495), (0.066, 2.600)]
    hair = [(0.100, 2.622), (0.160, 2.636), (0.210, 2.652), (0.226, 2.672), (0.212, 2.690), (0.186, 2.706),
            (0.172, 2.780), (0.174, 2.880), (0.172, 2.960), (0.150, 3.025), (0.090, 3.072)]
    return leg_in + shoe + leg_out + dress + neck + hair


def _mother_body():
    half = _mother_half()
    pts = [(0.0, 1.002)] + half + [(0.0, 3.085)] + _mirror(half)
    return pts, [0]


SHOULDER_BRAD = (0.288, 2.398)        # the brad at her left shoulder (mirrored for the right)


def _mother_arm(side):
    """A stroke: the arm from the brad at the shoulder out to the elbow and back in to the hand on the hip (half widths
    along it); `side` +1 her left (+x), -1 her right."""
    pts = [SHOULDER_BRAD, (0.462, 2.165), (0.615, 1.962), (0.512, 1.825), (0.398, 1.692), (0.345, 1.632), (0.300, 1.585)]
    widths = [0.072, 0.064, 0.056, 0.050, 0.042, 0.056, 0.058]
    return [(side * x, z) for x, z in pts], widths


def _mother_details():
    hair_out = [(0.098, 2.616), (0.160, 2.630), (0.214, 2.646), (0.232, 2.672), (0.216, 2.696), (0.192, 2.712),
                (0.178, 2.782), (0.180, 2.882), (0.178, 2.964), (0.154, 3.031), (0.092, 3.079)]   # the flip, up the side
    hairline = [(0.114, 2.650), (0.120, 2.760), (0.104, 2.860), (0.050, 2.912), (-0.030, 2.905), (-0.096, 2.872),
                (-0.122, 2.800), (-0.116, 2.660)]                                                 # round the face
    hair = hair_out + [(0.0, 3.092)] + _mirror(hair_out) + hairline[::-1]
    bow = ([(0.060, 3.040), (0.112, 3.098), (0.128, 3.066), (0.150, 3.020), (0.122, 3.012), (0.108, 3.028),
            (0.088, 2.980), (0.066, 2.990)], [0, 3, 6])
    dress = [(0.125, 2.476), (0.204, 2.468), (0.312, 2.452), (0.326, 2.420), (0.298, 2.360), (0.278, 2.280),
             (0.282, 2.150), (0.318, 1.900), (0.364, 1.600), (0.406, 1.320), (0.438, 1.100), (0.448, 1.028),
             (0.420, 1.004)]                                       # the bateau neckline out, the A-line down to the hem
    collar = ([(0.0, 2.452), (0.060, 2.470), (0.128, 2.470), (0.140, 2.430), (0.100, 2.390), (0.040, 2.392),
               (0.0, 2.420), (-0.040, 2.392), (-0.100, 2.390), (-0.140, 2.430), (-0.128, 2.470), (-0.060, 2.470)],
              [0, 6])
    band = ([(0.446, 1.070), (0.441, 1.130), (0.0, 1.130), (-0.441, 1.130), (-0.446, 1.070), (0.0, 1.070)], [0, 1, 3, 4])
    shoe = [(0.090, 0.098), (0.076, 0.035), (0.102, -0.006), (0.195, -0.010), (0.253, 0.006), (0.273, 0.045),
            (0.249, 0.092), (0.191, 0.120), (0.135, 0.120)]
    buttons = [_circle(0.0, z, 0.019) for z in (2.300, 2.170, 2.040)]
    return ([Detail("hair", 1, hair, []), Detail("accent", 2, *bow),
             Detail("detail", 1, [(0.0, 2.452)] + dress + [(0.0, 0.998)] + _mirror(dress), []),
             Detail("trim", 2, *collar), Detail("accent", 2, *band),
             Detail("hair", 1, shoe, []), Detail("hair", 1, _mirror(shoe), [])] +
            [Detail("trim", 2, b, []) for b in buttons])


# ===================================================================================================================
# the figures
# ===================================================================================================================
def _father():
    body, corners = _father_body()
    return Figure({"body": Piece(body, corners, None, _father_details(), None),
                   "hand": Piece(_father_hand(), [], None, _father_hand_details(), ELBOW)},
                  (0.0, 2.975), Hand(ELBOW, FATHER_TIP))


def _mother():
    body, corners = _mother_body()
    right = (-SHOULDER_BRAD[0], SHOULDER_BRAD[1])
    return Figure({"body": Piece(body, corners, None, _mother_details(), None),
                   "arm_L": Piece(None, [], _mother_arm(1.0), [], SHOULDER_BRAD),
                   "arm_R": Piece(None, [], _mother_arm(-1.0), [], right)},
                  (0.0, 2.780), None)


FIGURES = {"father": _father(), "mother": _mother()}
