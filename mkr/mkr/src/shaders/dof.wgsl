// EEVEE Next's depth of field as a post process (its default: the scene's `use_bokeh_jittered` off; dof.rs), run once
// on the film when a frame's samples are in. Each pixel's circle of confusion (CoC: signed radius in full-resolution
// pixels, positive beyond the focus plane) from the depth; a half-resolution copy that favours, of each 2 x 2 pixels,
// the widest blur and dark colours over bright ones (EEVEE's bilateral downsample: a small bright or in-front detail
// beside a wider blur drops out of the blurred layers, as it does on EEVEE's frames), and its mips; per tile the widest
// blur that may reach it; the background and foreground layers gathered at half resolution from rings of samples whose
// own blur covers the pixel within the aperture's shape; and the resolve at full resolution: the nearly focused pixels
// scattered over their small circles, laid over the background layer, the foreground layer over both, written back
// into the film (weight 1).

struct Dof {
    size: vec2<u32>,                // full resolution W, H
    half: vec2<u32>,                // half resolution W, H (rounded up)
    tiles: vec2<u32>,               // tiles across and down (TILE full-resolution pixels each)
    level: u32,                     // the mip pass's level (written; level - 1 is read)
    blades: u32,                    // the aperture's corners (fewer than 3: round)
    coc: vec4<f32>,                 // CoC = x + y * depth-buffer value (reversed), clamped to +-z
    shape: vec4<f32>,               // the blades' rotation (radians), the anisotropic scale across and up, rings
    weights: vec4<f32>,             // colour weight 1 / (luma + x), CoC weight sharpness y, slight focus up to z px,
                                    // w: energy floor (a sample spreads its light over at least w pixels)
}

const TILE: f32 = 16.0;
const PI: f32 = 3.14159265;

@group(0) @binding(0) var<storage, read_write> film: array<vec4<f32>>;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var full_w: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var full_r: texture_2d<f32>;
@group(0) @binding(4) var half_w: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var half_r: texture_2d<f32>;
@group(0) @binding(6) var<storage, read_write> tiles: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> dilated: array<vec4<f32>>;
@group(0) @binding(8) var bg_w: texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var fg_w: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var bg_r: texture_2d<f32>;
@group(0) @binding(11) var fg_r: texture_2d<f32>;
@group(0) @binding(12) var s_linear: sampler;

var<immediate> dof: Dof;

fn coc_of(depth: f32) -> f32 {
    return clamp(dof.coc.x + dof.coc.y * depth, -dof.coc.z, dof.coc.z);
}

// EEVEE's fast luma: twice the green plus the red and the blue.
fn luma4(c: vec3<f32>) -> f32 {
    return 2.0 * c.g + c.r + c.b;
}

// Four pixels as one: the widest CoC of them (its sign kept) chooses, a pixel whose blur ends sooner than the chosen
// one (nearer the camera) weighs less the shorter it is, bright colours less than dark ones.
fn reduce4(colors: array<vec3<f32>, 4>, cocs: array<f32, 4>) -> vec4<f32> {
    var chosen = cocs[0];
    for (var i = 1u; i < 4u; i = i + 1u) {
        if (abs(cocs[i]) > abs(chosen)) {
            chosen = cocs[i];
        }
    }
    var sum = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let w = saturate(1.0 - (chosen - cocs[i]) * dof.weights.y) / (luma4(colors[i]) + dof.weights.x);
        sum = sum + vec4<f32>(colors[i], cocs[i]) * w;
        wsum = wsum + w;
    }
    return sum / max(wsum, 1e-12);
}

// How far offset `o` (full-resolution pixels, the picture's axes: y down) lies across a bokeh of radius 1 (1 on its
// edge; linear in `o`): the aperture's polygon (corners on the unit circle at the rotation + k 2 pi / blades, the
// camera's axes: y up) squeezed by the anisotropic scale; `front` turns it half round (a point before the focus plane
// images the aperture upside down).
fn bokeh_dist(o: vec2<f32>, front: bool) -> f32 {
    var q = vec2<f32>(o.x, -o.y);
    if (front) {
        q = -q;
    }
    q = q / dof.shape.yz;
    let r = length(q);
    if (dof.blades < 3u || r == 0.0) {
        return r;
    }
    let sector = 2.0 * PI / f32(dof.blades);
    let a = atan2(q.y, q.x) - dof.shape.x;
    let t = a - sector * floor(a / sector) - 0.5 * sector;
    return r * cos(t) / cos(0.5 * sector);
}

fn tile_of(p: vec2<f32>) -> u32 {
    let t = min(vec2<u32>(p / TILE), dof.tiles - 1u);
    return t.y * dof.tiles.x + t.x;
}

// --------------------------------------------------------------------------------------------- full to half resolution
@compute @workgroup_size(8, 8)
fn setup(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= dof.half.x || gid.y >= dof.half.y) {
        return;
    }
    var colors: array<vec3<f32>, 4>;
    var cocs: array<f32, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let p = vec2<u32>(gid.x * 2u + (i & 1u), gid.y * 2u + (i >> 1u));
        let q = min(p, dof.size - 1u);
        let f = film[q.y * dof.size.x + q.x];
        var c = vec3<f32>(0.0);
        if (f.a > 0.0) {
            c = f.rgb / f.a;
        }
        let k = coc_of(textureLoad(depth_tex, vec2<i32>(q), 0));
        colors[i] = c;
        cocs[i] = k;
        if (p.x < dof.size.x && p.y < dof.size.y) {
            textureStore(full_w, vec2<i32>(p), vec4<f32>(c, k));
        }
    }
    textureStore(half_w, vec2<i32>(gid.xy), reduce4(colors, cocs));
}

// A mip of the half-resolution copy from the level above, by the same rule.
@compute @workgroup_size(8, 8)
fn mip(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst = textureDimensions(half_w);
    if (gid.x >= dst.x || gid.y >= dst.y) {
        return;
    }
    let src = textureDimensions(half_r);
    var colors: array<vec3<f32>, 4>;
    var cocs: array<f32, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let q = min(vec2<u32>(gid.x * 2u + (i & 1u), gid.y * 2u + (i >> 1u)), src - 1u);
        let v = textureLoad(half_r, vec2<i32>(q), 0);
        colors[i] = v.rgb;
        cocs[i] = v.a;
    }
    textureStore(half_w, vec2<i32>(gid.xy), reduce4(colors, cocs));
}

// ----------------------------------------------------------------------------------------------------------- tiles
// Per tile: the widest foreground and background blurs, and the widest nearly focused one (up to the slight focus's
// reach).
@compute @workgroup_size(8, 8)
fn tile(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= dof.tiles.x || gid.y >= dof.tiles.y) {
        return;
    }
    var fg = 0.0;
    var bg = 0.0;
    var slight = 0.0;
    let base = gid.xy * u32(TILE);
    for (var y = 0u; y < u32(TILE); y = y + 1u) {
        for (var x = 0u; x < u32(TILE); x = x + 1u) {
            let p = base + vec2<u32>(x, y);
            if (p.x >= dof.size.x || p.y >= dof.size.y) {
                continue;
            }
            let k = textureLoad(full_r, vec2<i32>(p), 0).a;
            fg = max(fg, -k);
            bg = max(bg, k);
            if (abs(k) < dof.weights.z + 0.5) {
                slight = max(slight, abs(k));
            }
        }
    }
    tiles[gid.y * dof.tiles.x + gid.x] = vec4<f32>(fg, bg, slight, 0.0);
}

// Per tile, the widest blurs of the tiles near enough for them to reach it.
@compute @workgroup_size(8, 8)
fn dilate(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= dof.tiles.x || gid.y >= dof.tiles.y) {
        return;
    }
    let reach = i32(ceil(dof.coc.z / TILE)) + 1;
    var out = vec3<f32>(0.0);
    for (var dy = -reach; dy <= reach; dy = dy + 1) {
        for (var dx = -reach; dx <= reach; dx = dx + 1) {
            let u = vec2<i32>(gid.xy) + vec2<i32>(dx, dy);
            if (u.x < 0 || u.y < 0 || u.x >= i32(dof.tiles.x) || u.y >= i32(dof.tiles.y)) {
                continue;
            }
            let rec = tiles[u32(u.y) * dof.tiles.x + u32(u.x)].xyz;
            // the gap between the two tiles' pixels
            let gap = length(vec2<f32>(max(abs(vec2<f32>(f32(dx), f32(dy))) - 1.0, vec2<f32>(0.0)))) * TILE;
            out = max(out, select(vec3<f32>(0.0), rec, rec >= vec3<f32>(gap)));
        }
    }
    dilated[gid.y * dof.tiles.x + gid.x] = vec4<f32>(out, 0.0);
}

// ---------------------------------------------------------------------------------------------------------- gather
// The layer's samples (`front`: the foreground's, CoC < 0; else the background's) whose blur covers the pixel at
// `centre` (full-resolution pixels), in rings out to `radius`, each weighted by its layer's share (the slight focus
// keeps the CoCs under weights.z), how much its blur covers the pixel and its energy (1 / CoC², its light spread over
// pi CoC²): rgb their colour, a the share of the pixel they cover.
fn gather_layer(centre: vec2<f32>, radius: f32, front: bool) -> vec4<f32> {
    if (radius < dof.weights.z - 0.5) {
        return vec4<f32>(0.0);
    }
    let rings = max(u32(dof.shape.w), 1u);
    let spacing = radius / f32(rings);
    let level = max(log2(spacing * 0.5), 0.0);
    let to_uv = 1.0 / (2.0 * vec2<f32>(dof.half));
    var acc = vec4<f32>(0.0);
    var count = 0.0;
    for (var ring = 0u; ring <= rings; ring = ring + 1u) {
        let n = max(8u * ring, 1u);
        let r = f32(ring) * spacing;
        for (var j = 0u; j < n; j = j + 1u) {
            let a = (f32(j) + 0.5 * f32(ring & 1u)) * 2.0 * PI / f32(n);
            let o = r * vec2<f32>(cos(a), sin(a));
            let s = textureSampleLevel(half_r, s_linear, (centre + o) * to_uv, level);
            let k = select(s.a, -s.a, front);
            let layer = saturate(k - dof.weights.z + 0.5);
            let cover = saturate(k - bokeh_dist(-o, front) + 0.5);
            let w = layer * cover / max(k * k, dof.weights.w);
            acc = acc + vec4<f32>(s.rgb * w, w);
            count = count + 1.0;
        }
    }
    // each sample stands for pi radius² / count pixels and spreads 1 / (pi CoC²) of its light on each
    return vec4<f32>(acc.rgb / max(acc.w, 1e-12), acc.w * radius * radius / count);
}

@compute @workgroup_size(8, 8)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= dof.half.x || gid.y >= dof.half.y) {
        return;
    }
    let centre = vec2<f32>(gid.xy) * 2.0 + 1.0;
    let t = dilated[tile_of(centre)];
    textureStore(bg_w, vec2<i32>(gid.xy), gather_layer(centre, t.y, false));
    textureStore(fg_w, vec2<i32>(gid.xy), gather_layer(centre, t.x, true));
}

// --------------------------------------------------------------------------------------------------------- resolve
@compute @workgroup_size(8, 8)
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= dof.size.x || gid.y >= dof.size.y) {
        return;
    }
    let p = vec2<i32>(gid.xy);
    let me = textureLoad(full_r, p, 0);
    let centre = vec2<f32>(gid.xy) + 0.5;
    let uv = centre / (2.0 * vec2<f32>(dof.half));
    let fg = textureSampleLevel(fg_r, s_linear, uv, 0.0);
    let bg = textureSampleLevel(bg_r, s_linear, uv, 0.0);
    // the nearly focused pixels whose small circles cover this one, each spreading its light over pi CoC² (at least a
    // pixel) and kept by the slight focus's share
    let reach = min(dilated[tile_of(centre)].z, dof.weights.z + 0.5);
    let r = i32(ceil(reach));
    var slight = vec4<f32>(0.0);
    for (var dy = -r; dy <= r; dy = dy + 1) {
        for (var dx = -r; dx <= r; dx = dx + 1) {
            let q = p + vec2<i32>(dx, dy);
            if (q.x < 0 || q.y < 0 || q.x >= i32(dof.size.x) || q.y >= i32(dof.size.y)) {
                continue;
            }
            let s = textureLoad(full_r, q, 0);
            let k = abs(s.a);
            let share = saturate(dof.weights.z + 0.5 - k);
            if (share <= 0.0) {
                continue;
            }
            let d = bokeh_dist(vec2<f32>(f32(-dx), f32(-dy)), s.a < 0.0);
            let w = share * saturate(max(k, 0.5) - d + 0.5) / max(1.0, PI * k * k);
            slight = slight + vec4<f32>(s.rgb * w, w);
        }
    }
    let a_s = min(slight.a, 1.0);
    let c_s = slight.rgb / max(slight.a, 1e-12);
    let c_b = select(me.rgb, bg.rgb, bg.a > 1e-4);
    let a_f = saturate(fg.a);
    let c_f = fg.rgb;
    let out = c_f * a_f + (1.0 - a_f) * (c_s * a_s + (1.0 - a_s) * c_b);
    film[gid.y * dof.size.x + gid.x] = vec4<f32>(out, 1.0);
}
