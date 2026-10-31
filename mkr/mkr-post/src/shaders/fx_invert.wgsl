// invert (mkmmd.eralook.invert; EraFx._invert): the picture inverted, its hues then turned (through OpenCV's float HSV,
// as cv2.cvtColor converts it), mixed in by the level and the region.

struct Params {
    w: u32,
    h: u32,
    turn: u32,          // 1: the hues turned
    mask: u32,          // 0 none, 1 the figure, 2 the ground
    hue: f32,           // float32 degrees
    v: f32,             // float32 of the level times the strength
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> region: array<vec4<f32>>;

const FLT_EPSILON: f32 = 1.1920929e-7;

// cv2.COLOR_RGB2HSV of a float pixel: (h degrees, s, v).
fn rgb2hsv(c: vec3<f32>) -> vec3<f32> {
    let r = c.x;
    let g = c.y;
    let b = c.z;
    var v = r;
    var vmin = r;
    if (v < g) { v = g; }
    if (v < b) { v = b; }
    if (vmin > g) { vmin = g; }
    if (vmin > b) { vmin = b; }
    var diff = v - vmin;
    let s = diff / (abs(v) + FLT_EPSILON);
    diff = 60.0 / (diff + FLT_EPSILON);
    var h: f32;
    if (v == r) {
        h = (g - b) * diff;
    } else if (v == g) {
        h = (b - r) * diff + 120.0;
    } else {
        h = (r - g) * diff + 240.0;
    }
    if (h < 0.0) {
        h = h + 360.0;
    }
    return vec3<f32>(h, s, v);
}

// cv2.COLOR_HSV2RGB of a float pixel.
fn hsv2rgb(hsv: vec3<f32>) -> vec3<f32> {
    let s = hsv.y;
    let v = hsv.z;
    if (s == 0.0) {
        return vec3<f32>(v);
    }
    var h = hsv.x * 0.016666668;
    h = h - 6.0 * trunc(h / 6.0);
    if (h < 0.0) {
        h = h + 6.0;
    }
    var sector = i32(floor(h));
    h = h - f32(sector);
    if (sector < 0 || sector >= 6) {
        sector = 0;
        h = 0.0;
    }
    let t0 = v;
    let t1 = v * (1.0 - s);
    let t2 = v * (1.0 - s * h);
    let t3 = v * (1.0 - s * (1.0 - h));
    // sector_data {b, g, r}: {1,3,0}, {1,0,2}, {3,0,1}, {0,2,1}, {0,1,3}, {2,1,0}
    switch sector {
        case 0: { return vec3<f32>(t0, t3, t1); }
        case 1: { return vec3<f32>(t2, t0, t1); }
        case 2: { return vec3<f32>(t1, t0, t3); }
        case 3: { return vec3<f32>(t1, t2, t0); }
        case 4: { return vec3<f32>(t3, t1, t0); }
        default: { return vec3<f32>(t0, t1, t2); }
    }
}

@compute @workgroup_size(16, 16)
fn invert(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= P.w || id.y >= P.h) {
        return;
    }
    let i = id.y * P.w + id.x;
    let img = src[i];
    var out = 1.0 - clamp(img.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    if (P.turn == 1u) {
        var hsv = rgb2hsv(out);
        var hh = hsv.x + P.hue;
        hh = hh - 360.0 * floor(hh / 360.0);
        hsv.x = hh;
        out = hsv2rgb(hsv);
    }
    out = clamp(out, vec3<f32>(0.0), vec3<f32>(1.0));
    var a = P.v;
    if (P.mask == 1u) {
        a = region[i].x * a;
    } else if (P.mask == 2u) {
        a = (1.0 - region[i].x) * a;
    }
    dst[i] = vec4<f32>(img.xyz + (out - img.xyz) * a, img.w);
}
