// needs fields.wgsl
// spiral (mkmmd.fields.pattern; FieldFx._spiral): an op-art pattern laid on the picture by its blend, mixed in by the
// strength and kept to its region; `pattern_fill` paints one (a collage's pattern back, mkmmd.fields.Painter) over a
// plane of P.w x P.h.

struct Params {
    w: u32,
    h: u32,
    blend: u32,         // fields.wgsl's field_blend modes
    mask: u32,          // 0 none, 1 the figure, 2 the ground
    cx: f32,            // the pattern's centre (px)
    cy: f32,
    strength: f32,
    pad: f32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> pat: Pattern;
@group(0) @binding(6) var<storage, read> region: array<vec4<f32>>;

fn spiral_colour(id: vec3<u32>) -> vec3<f32> {
    let pl = field_polar(id.x, id.y, vec2<f32>(P.cx, P.cy), pat.unit);
    return pattern_colour(pat, pl.x, pl.y, pl.z, pl.w);
}

@compute @workgroup_size(16, 16)
fn spiral(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let i = id.y * P.w + id.x;
    var a = P.strength;
    if (P.mask == 1u) {
        a = region[i].x * a;
    } else if (P.mask == 2u) {
        a = (1.0 - region[i].x) * a;
    }
    let img = src[i];
    dst[i] = vec4<f32>(field_blend(img.xyz, spiral_colour(id), a, P.blend), img.w);
}

@compute @workgroup_size(16, 16)
fn pattern_fill(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    dst[id.y * P.w + id.x] = vec4<f32>(spiral_colour(id), 1.0);
}
