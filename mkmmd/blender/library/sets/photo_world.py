"""photo_world: the scene's world made of one photo (the maths is mkmmd/core/photoworld.py, which these nodes follow
step by step). Nothing spatial: the set root is only an empty, and the photo is in WORLD directions."""
import math

from ....core import photoworld as PW
from . import register
from .night_sky import FAR, to_root


def _tree(K, world, cfg, img, fill):
    """The world's nodes: a ray direction to the photo's texture coordinates (core.photoworld.uv), the photo there or
    `fill` outside it, at `strength` for camera rays and `light` for every other ray (lighting, reflections)."""
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    nb = K.NB(nt)
    f, r, u = PW.frame(cfg["toward"], cfg["roll"])
    w, h = (float(v) for v in img.size)
    L = max(w, h)
    d = nb.texcoord("Generated")                                     # in a world shader: the ray's direction
    a, b, c = (nb.vdot(d, tuple(float(x) for x in v)) for v in (r, u, f))
    lateral = nb.math("SQRT", nb.math("ADD", nb.math("MULTIPLY", a, a), nb.math("MULTIPLY", b, b)))
    alpha = nb.math("MULTIPLY", nb.math("ARCTAN2", lateral, c), float(cfg["fov"]) / float(cfg["spread"]))
    seen = nb.math("LESS_THAN", alpha, 0.5 * math.pi)
    rho = nb.math("DIVIDE", nb.math("TANGENT", nb.math("MINIMUM", alpha, 0.5 * math.pi - PW.EDGE)),
                  math.tan(math.radians(float(cfg["fov"])) / 2.0))
    g = nb.math("DIVIDE", rho, nb.math("MAXIMUM", lateral, 1e-12))
    s = nb.math("MULTIPLY_ADD", nb.math("MULTIPLY", a, g), 0.5 * L / w, 0.5)
    t = nb.math("MULTIPLY_ADD", nb.math("MULTIPLY", b, g), 0.5 * L / h, 0.5)
    tex = nb.node("ShaderNodeTexImage", {0: nb.comb(s, t, 0.0)}, image=img, interpolation="Smart", extension="EXTEND")
    mask = nb.math("MULTIPLY", seen, tex.outputs["Alpha"])
    for v in (s, t):
        mask = nb.math("MULTIPLY", mask, nb.math("LESS_THAN", nb.math("ABSOLUTE", nb.math("SUBTRACT", v, 0.5)), 0.5))
    col = nb.mixc(mask, tuple(fill) + (1.0,), tex.outputs["Color"])
    if float(cfg["saturation"]) != 1.0:
        col = nb.node("ShaderNodeHueSaturation", {"Color": col, "Saturation": float(cfg["saturation"])}).outputs[0]
    cam = nb.node("ShaderNodeLightPath").outputs["Is Camera Ray"]
    bg = nb.node("ShaderNodeBackground", {0: col, 1: nb.mixf(cam, float(cfg["light"]), float(cfg["strength"]))})
    nt.links.new(bg.outputs[0], nb.node("ShaderNodeOutputWorld").inputs[0])


@register("photo_world")
def photo_world(name, coll, root, spec, palette):
    """photo_world: the scene's world, i.e. the background, the light of the sky and what glossy things reflect, made of
    ONE photo projected round the scene in world directions (the set's `at` and `yaw` do not move or turn it). Replaces
    the scene's world with a fresh one named after the set.

      image = "street.jpg"   the photo (required): a file, project-relative, absolute or ~/..., or the slug of an asset
                             registry entry whose path is an image (jpg, jpeg, png, webp, exr, hdr). Its alpha, if it
                             has one, lets `fill` through
      toward = [0, 1, 0]     the world direction of the photo's centre (default: behind a cast member who faces -y,
                             what a camera in front of her sees)
      roll = 0               degrees: the photo turned clockwise about `toward`, as a camera looking along it sees it
      fov                    degrees across the photo's long side (required): its true field of view
      spread = fov           degrees its long side spans in the world: fov is the true perspective; more stretches it
                             angularly and uniformly about its centre (a modest photo fills a fisheye's 180 degrees)
      fill = "#080808"       palette slot or '#hex': the world where the photo does not reach
      strength = 1           what the camera sees (1: the photo's own pixels)
      light = strength       the photo's brightness for every other ray: the light it casts and what glossy things
                             reflect (daylight on the cast may want well over 1 while the background stays the photo)
      saturation = 1         the photo's saturation (0 grey)

    EEVEE: the world probe is 2048 px (sharp reflections) and the world's sun extraction is off. Card: kind, paths {},
    use.look [{name: "centre", point}] (100 km out along `toward`, in the set root's frame), use.surface [],
    colliders [], lights []."""
    import bpy

    from ...runtime import CTX
    from . import nightkit as K

    cfg = PW.parse(spec)
    path = PW.resolve(cfg["image"], (CTX.get("project") or {}).get("root"), (CTX.get("config") or {}).get("assets"))
    try:
        img = bpy.data.images.load(str(path), check_existing=True)
    except RuntimeError as e:
        raise PW.PhotoWorldError(f"photo_world {name!r}: Blender cannot read {path}: {e}") from None
    if min(img.size) < 1:
        raise PW.PhotoWorldError(f"photo_world {name!r}: {path} has no pixels Blender can read")
    try:
        fill = K.rgb(palette, cfg["fill"])
    except (KeyError, ValueError) as e:
        raise PW.PhotoWorldError(f"photo_world {name!r}: fill {cfg['fill']!r}: {e.args[0]}") from None
    old = bpy.data.worlds.get(name)
    if old is not None:
        bpy.data.worlds.remove(old)
    world = bpy.data.worlds.new(name)
    bpy.context.scene.world = world
    _tree(K, world, cfg, img, fill)
    world.sun_threshold = 0.0
    world.probe_resolution = "2048"
    look = [{"name": "centre", "point": to_root(PW.frame(cfg["toward"])[0] * FAR, root.rotation_euler.z).tolist()}]
    return {"kind": "photo_world", "paths": {}, "use": {"look": look, "surface": []}, "colliders": [], "lights": []}
