// The looks' passes (looks_gpu.rs), appended to the drawing module after its materials: Workbench's flat colours,
// material ids and depth, its studio light and the stencil shadow volumes of the vector look's light, its texture
// colour. Group 1 holds what the looks add to the scene's bindings.

struct LookDraw {
    color: vec4<f32>,               // flat: the object's colour (linear); tex: the viewport colour and alpha
    info: vec4<u32>,                // x: material id; y: image (the scene's index, 0xffffffff: none); z: its UV slot in
                                    // the mesh's attributes; w: flags (1 premultiplied, 2 alpha cut at 0.1, 4 clipped,
                                    // 8 closest; extension in bits 4-5: 0 repeat, 1 extend, 2 mirror)
}

// Workbench's studio light (get_world_lighting, no specular) and its shadow (get_shadow), in the world.
struct LookLight {
    dirs: array<vec4<f32>, 4>,      // xyz the light's direction (glTF world), w its wrap
    diffuse: array<vec4<f32>, 4>,   // x its diffuse (red: the pass is white), 0 for a light not in use
    ambient: vec4<f32>,             // x the studio light's ambient (red)
    shadow: vec4<f32>,              // xyz the look's light's direction of travel (glTF world), w shadow_shift
    params: vec4<f32>,              // x the shadow focus edge, y the lit share (1), z the shadowed share (0), w far plane
    plane: vec4<f32>,               // planar depth: dot(xyz, p) + w (metres along the view axis)
}

struct VolEdge {
    v: vec4<u32>,                   // the edge's faces' far vertices and its ends, as Workbench orders them (v1 v2 the edge)
    m: vec4<u32>,                   // x its mesh
}

struct VolCap {
    v: vec4<u32>,                   // a face's three vertices, w its mesh
}

@group(1) @binding(0) var<storage, read> look_draws: array<LookDraw>;
@group(1) @binding(1) var<storage, read> look_light: LookLight;
@group(1) @binding(2) var<storage, read> vol_edges: array<VolEdge>;
@group(1) @binding(3) var<storage, read> vol_caps: array<VolCap>;
@group(1) @binding(4) var look_normals: texture_2d<f32>;

struct LookOut {
    @builtin(position) @invariant pos: vec4<f32>,
    @location(0) n: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) depth: f32,
    @location(3) @interpolate(flat) draw: u32,
}

// A surface of the look on the frame's own time (no motion blur: styles.Looks._film turns it off).
@vertex
fn vs_look(@builtin(vertex_index) v: u32, @builtin(instance_index) di: u32) -> LookOut {
    let d = draws[di];
    let m = meshes[d.mesh];
    let p = world_pos(m, v);
    var o: LookOut;
    o.pos = samples[imm.sample].vp * vec4<f32>(p, 1.0);
    o.n = world_nrm(m, v);
    o.uv = attr_of(m, v, look_draws[di].info.z).xy;
    o.depth = dot(look_light.plane.xyz, p) + look_light.plane.w;
    o.draw = di;
    return o;
}

// The flat pass: the object's colour, opaque (Workbench's flat light with its Object colour).
@fragment
fn fs_look_flat(in: LookOut) -> @location(0) vec4<f32> {
    return vec4<f32>(look_draws[in.draw].color.rgb, 1.0);
}

// The material ids, no antialiasing, and the planar depth of the same draw.
struct IdOut {
    @location(0) id: u32,
    @location(1) depth: f32,
}

@fragment
fn fs_look_id(in: LookOut) -> IdOut {
    return IdOut(look_draws[in.draw].info.x, in.depth);
}

// The light pass's normals: toward the viewer on back faces, alpha the coverage.
@fragment
fn fs_look_normal(in: LookOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let n = normalize(in.n);
    return vec4<f32>(select(-n, n, front), 1.0);
}

// The texture colour: the image (sampled as Workbench samples it: no mipmaps), cut where its alpha is under 0.1, or the
// viewport colour; the material's alpha. Linear: the film turns it to display values.
@fragment
fn fs_look_tex(in: LookOut) -> @location(0) vec4<f32> {
    let ld = look_draws[in.draw];
    var c = ld.color.rgb;
    if (ld.info.y != 0xffffffffu) {
        let flags = ld.info.w;
        var st = vec2<f32>(in.uv.x, 1.0 - in.uv.y);
        let ext = (flags >> 4u) & 3u;
        var t: vec4<f32>;
        let img = images[ld.info.y];
        if ((flags & 8u) != 0u) {
            if (ext == 0u) {
                t = textureSampleLevel(img, s_nearest_repeat, st, 0.0);
            } else if (ext == 2u) {
                t = textureSampleLevel(img, s_nearest_mirror, st, 0.0);
            } else {
                t = textureSampleLevel(img, s_nearest_extend, st, 0.0);
            }
        } else {
            if (ext == 0u) {
                t = textureSampleLevel(img, s_linear_repeat, st, 0.0);
            } else if (ext == 2u) {
                t = textureSampleLevel(img, s_linear_mirror, st, 0.0);
            } else {
                t = textureSampleLevel(img, s_linear_extend, st, 0.0);
            }
        }
        if ((flags & 4u) != 0u && (any(in.uv < vec2<f32>(0.0)) || any(in.uv > vec2<f32>(1.0)))) {
            t = vec4<f32>(0.0);
        }
        if ((flags & 1u) != 0u && t.a > 0.0 && t.a < 1.0) {
            t = vec4<f32>(t.rgb / t.a, t.a);
        }
        if ((flags & 2u) != 0u && t.a < 0.1) {
            discard;
        }
        c = t.rgb;
    }
    return vec4<f32>(c, ld.color.a);
}

// ------------------------------------------------------------------------------------------------- shadow volumes
// Extruded toward the far plane (Workbench's extrude_offset), along the light's direction of travel.
fn vol_back(p: vec3<f32>) -> vec3<f32> {
    let L = look_light.shadow.xyz;
    var d = 1e5;
    let lf = dot(look_light.plane.xyz, L);
    if (lf > 0.0) {
        d = (look_light.params.w - dot(look_light.plane.xyz, p) - look_light.plane.w) / lf - 1e-3;
    }
    return p + L * d;
}

fn vol_pos(mesh: u32, v: u32) -> vec3<f32> {
    return world_pos(meshes[mesh], v);
}

// An edge: a quad along the light where one of its faces faces the light and the other does not (six vertices).
@vertex
fn vs_vol_edge(@builtin(vertex_index) vi: u32, @builtin(instance_index) ei: u32) -> @builtin(position) vec4<f32> {
    let e = vol_edges[ei];
    let mesh = e.m.x;
    let v0 = vol_pos(mesh, e.v.x);
    let v1 = vol_pos(mesh, e.v.y);
    let v2 = vol_pos(mesh, e.v.z);
    let v3 = vol_pos(mesh, e.v.w);
    let L = look_light.shadow.xyz;
    let n1 = cross(v2 - v1, v0 - v1);
    let n2 = cross(v3 - v1, v2 - v1);
    let degen = vec2<bool>(dot(n1, n1) < 4e-17, dot(n2, n2) < 4e-17);
    var bf = vec2<bool>(dot(n1, L) > 0.0, dot(n2, L) > 0.0);
    if (degen.x) {
        bf.x = !bf.y;
    }
    if (degen.y) {
        bf.y = !bf.x;
    }
    if ((degen.x && degen.y) || bf.x == bf.y) {
        return vec4<f32>(0.0);
    }
    let a = select(v2, v1, bf.x);
    let b = select(v1, v2, bf.x);
    var s = 3u;
    if (vi < 3u) {
        s = vi;
    } else if (vi == 3u) {
        s = 2u;
    } else if (vi == 4u) {
        s = 1u;
    }
    var w = vol_back(b);
    if (s == 0u) {
        w = a;
    } else if (s == 1u) {
        w = b;
    } else if (s == 2u) {
        w = vol_back(a);
    }
    return samples[imm.sample].vp * vec4<f32>(w, 1.0);
}

// A face: its front cap where it is and its back cap at the far end, both turned to face the light.
@vertex
fn vs_vol_cap(@builtin(vertex_index) vi: u32, @builtin(instance_index) ci: u32) -> @builtin(position) vec4<f32> {
    let c = vol_caps[ci];
    let v0 = vol_pos(c.v.w, c.v.x);
    let v1 = vol_pos(c.v.w, c.v.y);
    let v2 = vol_pos(c.v.w, c.v.z);
    let inv = dot(cross(v2 - v1, v0 - v1), look_light.shadow.xyz) > 0.0;
    let front = vi < 3u;
    let t = vi % 3u;
    var idx = vec2<u32>(1u, 2u);
    if (inv) {
        idx = vec2<u32>(2u, 1u);
    }
    var k = 0u;
    if (t == 1u) {
        k = select(idx.y, idx.x, front);
    } else if (t == 2u) {
        k = select(idx.x, idx.y, front);
    }
    var p = v2;
    if (k == 0u) {
        p = v0;
    } else if (k == 1u) {
        p = v1;
    }
    if (!front) {
        p = vol_back(p);
    }
    var clip = samples[imm.sample].vp * vec4<f32>(p, 1.0);
    // pushed a little away from the viewer, as Workbench's caps are (WebGL's +0.00005 over [-1, 1]; here depth runs
    // over [0, 1], reversed)
    clip.z = clip.z - 0.000025 * clip.w;
    return clip;
}

@fragment
fn fs_vol() {
}

// ----------------------------------------------------------------------------------------------- the studio light
@vertex
fn vs_screen(@builtin(vertex_index) v: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((v << 1u) & 2u), f32(v & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

fn fast_rcp(v: f32) -> f32 {
    return bitcast<f32>(0x7eef370b - bitcast<i32>(v));
}

fn wrapped(nl: f32, w: f32) -> f32 {
    let w1 = w + 1.0;
    return clamp((nl + w) * fast_rcp(w1 * w1), 0.0, 1.0);
}

// One sample's shade, premultiplied: the studio light on white times Workbench's shadow factor; inside the stencil's
// shadow volumes (imm.draw 1) only the shadowed share.
@fragment
fn fs_look_light(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let n4 = textureLoad(look_normals, vec2<i32>(pos.xy), 0);
    if (n4.a < 0.5) {
        discard;
    }
    let n = normalize(n4.xyz);
    var d = look_light.ambient.x;
    for (var i = 0u; i < 4u; i = i + 1u) {
        d = d + wrapped(dot(look_light.dirs[i].xyz, n), look_light.dirs[i].w) * look_light.diffuse[i].x;
    }
    var lit = look_light.params.y;
    if (imm.draw == 1u) {
        lit = 0.0;
    }
    let mixv = smoothstep(look_light.shadow.w, look_light.params.x, -dot(n, look_light.shadow.xyz)) * lit;
    return vec4<f32>(d * (mixv + look_light.params.z), 0.0, 0.0, 1.0);
}
