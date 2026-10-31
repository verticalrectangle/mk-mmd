// needs fields.wgsl
// flood (mkmmd.fields.flood_front, flood; FieldFx._flood): ink dropped from a point. Per drop, in the order they
// fell: `front` works out how far its front has to run to each pixel (rho) and its mottle, and the largest rho
// (`maxes`, an atomic over the float's bits: rho is never below 0); `drop` lays the drop's colour into the premultiplied
// accumulation `acc` (rgb, coverage in .w); then `lay` lays the flood on the picture by its coverage and the strength,
// kept to its region.

struct Params {
    w: u32,
    h: u32,
    drop: u32,          // this drop's place in the calls (three each: the lobes, the ragged edge, the mottle)
    first: u32,         // 1: the first drop (the accumulation starts from nothing)
    mask: u32,          // 0 none, 1 the figure, 2 the ground
    pad0: u32,
    pad1: u32,
    pad2: u32,
    cx: f32,            // the centre (px)
    cy: f32,
    unit: f32,          // the short side (px)
    wob: f32,           // wobble
    soft: f32,          // float32 of max(soft x px, 1), and 3, 0.8 and 1.4 times it as float64 then float32
    soft3: f32,
    soft08: f32,
    soft14: f32,
    share: f32,
    alpha: f32,
    wob2: f32,          // float32(2 x wobble)
    strength: f32,
    colour: vec4<f32>,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> front: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> acc: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> maxes: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> region: array<vec4<f32>>;

fn flood_here(id: vec3<u32>) -> i32 {
    if (id.x >= P.w || id.y >= P.h) {
        return -1;
    }
    return i32(id.y * P.w + id.x);
}

@compute @workgroup_size(16, 16)
fn flood_front(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = flood_here(id);
    if (i < 0) {
        return;
    }
    let pl = field_polar(id.x, id.y, vec2<f32>(P.cx, P.cy), P.unit);
    let c = 3u * P.drop;
    let n1 = field_noise(pl.x / 0.33, pl.y / 0.33, c);
    let n2 = field_noise(pl.x / 0.065, pl.y / 0.065, c + 1u);
    let stretch = 1.0 + P.wob * (1.3 * (n1 - 0.5) + 0.35 * (n2 - 0.5));
    let rho = pl.z * max(stretch, 0.2) * P.unit;
    let mottle = field_noise(pl.x / 0.12, pl.y / 0.12, c + 2u);
    front[i] = vec4<f32>(rho, mottle, 0.0, 0.0);
    atomicMax(&maxes[P.drop], bitcast<u32>(max(rho, 0.0)));
}

@compute @workgroup_size(16, 16)
fn flood_drop(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = flood_here(id);
    if (i < 0) {
        return;
    }
    let most = bitcast<f32>(atomicLoad(&maxes[P.drop]));
    let r = P.share * (most + 4.0 * P.soft);
    let f = front[i];
    let ahead = r - f.x;
    var a = clamp(ahead / P.soft + 0.5, 0.0, 1.0);
    let wash = 0.22 * clamp(ahead / P.soft3 + 1.0, 0.0, 1.0);
    a = max(a, wash) * P.alpha;
    let shade = 1.0 - 0.07 * (f.y - 0.5) * P.wob2 - 0.16 * field_bump((ahead - P.soft08) / P.soft14);
    var prev = vec4<f32>(0.0);
    if (P.first == 0u) {
        prev = acc[i];
    }
    let pre = prev.xyz * (1.0 - a) + (P.colour.xyz * shade) * a;
    acc[i] = vec4<f32>(pre, prev.w + a * (1.0 - prev.w));
}

@compute @workgroup_size(16, 16)
fn flood_lay(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = flood_here(id);
    if (i < 0) {
        return;
    }
    let f = acc[i];
    let pic = f.xyz / max(f.w, 1e-6);
    var a = f.w * P.strength;
    if (P.mask == 1u) {
        a = region[i].x * a;
    } else if (P.mask == 2u) {
        a = (1.0 - region[i].x) * a;
    }
    let img = src[i];
    dst[i] = vec4<f32>(img.xyz + (pic - img.xyz) * a, img.w);
}
