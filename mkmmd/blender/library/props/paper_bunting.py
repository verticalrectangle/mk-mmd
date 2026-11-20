"""The paper bunting (`card = "library:paper_bunting"`): a string of paper pennants hung on a wall. Objects
`<name>_flags` (every flag, one sheet each, coloured per flag by the colour attribute `mk_paper`; form-exempt: paper is
flat by nature), `<name>_cord` (the twine and its loose ends) and `<name>_mounts` (two map pins, or two strips of
masking tape: form-exempt), under the root, which lies on the wall midway between the ends.

    [[prop]]
    name = "bunting"
    card = "library:paper_bunting"
    at = [0.0, 1.8, 2.15]                 # on the wall plane (y = 1.8), facing the room (-Y)
    slots = { span = 1.8, sag = 0.18, shape = "triangle", colors = ["love", "gold", "rose", "pine", "foam", "iris"] }

Numbers, options, roles and the card: paper_bunting_layout.py; the paper, tape and twine: papercraft_mats.py."""
import numpy as np

from .. import shell as SH
from . import paper_bunting_layout as LAY
from . import papercraft as PC
from . import papercraft_mats as PM
from . import register
from . import showkit as SK


@register("paper_bunting")
def paper_bunting(name, coll, root, slots=None):
    slots = dict(slots or {})
    O = LAY.options(slots)
    roles = SK.roles(slots, LAY.ROLES, "paper_bunting")
    mats = {"paper": PM.paper(f"{name}_paper"),
            "cord": PM.twine(f"{name}_cord", SK.rgb(roles["cord"], slots, "paper_bunting: cord")),
            "pin": PM.plastic(f"{name}_pin", SK.rgb(roles["pin"], slots, "paper_bunting: pin")),
            "tape": PM.tape(f"{name}_tape", SK.rgb(roles["tape"], slots, "paper_bunting: tape"))}
    mat_of = lambda i: mats[LAY.MATS[i]]                                                       # noqa: E731
    flags, idx = LAY.flags_mesh(O)
    cols = PC.tones(O.colors, int(idx.max()) + 1, slots, np.random.default_rng(int(O.seed) + 101), O.tone,
                    "paper_bunting: colors")
    o = SH.mesh_object(f"{name}_flags", flags, coll, root, mat_of)
    PM.paint(o, cols[idx])
    SH.exempt(o, "paper flags: flat by nature")
    SH.mesh_object(f"{name}_cord", LAY.cord_mesh(O), coll, root, mat_of)
    m = SH.mesh_object(f"{name}_mounts", LAY.mounts_mesh(O), coll, root, mat_of)
    if O.mount == "tape":
        SH.exempt(m, "masking tape: flat by nature")
    return LAY.card(name, O, roles)
