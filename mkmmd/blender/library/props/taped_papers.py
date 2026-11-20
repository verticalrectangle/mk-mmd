"""The taped papers (`card = "library:taped_papers"`): a loose cluster of flat paper things taped to a wall. Objects
`<name>_<kind><i>` (piece i: a printed poster or instant photo with its picture in a material of its own, or a moon, star
or card in coloured paper, coloured by the colour attribute `mk_paper`) and `<name>_tape` (every strip of masking tape),
under the root, which lies on the wall; every one form-exempt (paper is flat by nature).

    [[prop]]
    name = "papers"
    card = "library:taped_papers"
    at = [0.05, 1.8, 1.30]                # the gap (the card's `use.look` `gap`) behind her face, on the wall plane

    [prop.slots]
    pieces = [{ kind = "poster", design = "sun", at = [-0.53, 0.03], rot = -2.5 },
              { kind = "polaroid", photo = "heart", color = "foam", ink = "love", at = [-0.25, 0.27], rot = 7 },
              { kind = "moon", color = "gold", at = [0.04, 0.31], rot = -14 }]

Numbers, kinds, options, roles and the card: taped_papers_layout.py; the paper and the tape: papercraft_mats.py."""
import numpy as np

from .. import shell as SH
from . import papercraft as PC
from . import papercraft_mats as PM
from . import register
from . import showkit as SK
from . import taped_papers_layout as LAY


@register("taped_papers")
def taped_papers(name, coll, root, slots=None):
    slots = dict(slots or {})
    pieces, seed, tone = LAY.options(slots)
    roles = SK.roles(slots, LAY.ROLES, "taped_papers")
    paper = PM.paper(f"{name}_paper")
    tape = PM.tape(f"{name}_tape", SK.rgb(roles["tape"], slots, "taped_papers: tape"))
    pics = LAY.pictures(pieces, slots, roles, seed)
    rng = np.random.default_rng(int(seed) + 5)
    for p in pieces:
        oname = f"{name}_{p.kind}{p.i}"
        m = LAY.piece_mesh(p)
        if p.i in pics:
            win = (*LAY.photo_window(p), 0.24) if p.kind == "polaroid" else None
            mat = PM.printed(oname, PM.image(f"{oname}_print", pics[p.i]), gloss=win)
            o = SH.mesh_object(oname, m, coll, root, lambda _i, mat=mat: mat)
        else:
            o = SH.mesh_object(oname, m, coll, root, lambda _i: paper)
            rgb = PC.tones([p.color], 1, slots, rng, tone, f"taped_papers: pieces[{p.i}].color")[0]
            PM.paint(o, np.repeat(rgb[None], len(m.V), 0))
        SH.exempt(o, "paper: flat by nature")
    t = SH.mesh_object(f"{name}_tape", LAY.tapes_mesh(pieces, seed), coll, root, lambda _i: tape)
    SH.exempt(t, "masking tape: flat by nature")
    return LAY.card(name, pieces, roles)
