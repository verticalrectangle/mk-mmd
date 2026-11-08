// An effect limited to a `rect` (mkmmd.framefx.FrameFx._apply): the picture the effect left, kept inside the rect, the
// frame outside it as it was before the effect, `before + (after - before) * m` with m = rows[y] * cols[x] (the rect's
// ramps, mkmmd.picturefx.rect_ramps). `ramps` holds h row ramps then w column ramps, float32.

struct Params {
    w: u32,
    h: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> after: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> before: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> ramps: array<f32>;

@compute @workgroup_size(16, 16)
fn rect(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let i = id.y * P.w + id.x;
    let m = ramps[id.y] * ramps[P.h + id.x];
    let a = before[i];
    let b = after[i];
    dst[i] = vec4<f32>(a.xyz + (b.xyz - a.xyz) * m, b.w);
}
