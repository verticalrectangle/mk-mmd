// Normals that bend with a deforming mesh as Blender's do, rebuilt from the frame's posed vertices after the skin pass
// (mkr/mkr/src/bend.rs; mkmmd/core/bend.py `normals` is the reference, the bake's MESH `bend` the data), over every
// bending mesh at once (vertices are the posed buffer's, faces and fans numbered across the meshes):
//   faces: per face n_f = the unit sum of its triangles' cross products;
//   fans:  per fan lnor = the unit sum over its corners of n_f times the face's angle at the corner, ra = the mean angle
//          between lnor and its edges, ref = its reference edge flattened onto lnor's plane, ortho = lnor x ref, span =
//          the angle round lnor from ref to its other edge flattened alike (a whole turn when the two are one edge);
//          then per vertex of the fan a = alpha ra (alpha < 0: 2 pi + alpha (2 pi - ra)), b = beta span and
//          normal = cos(a) lnor + sin(a) (cos(b) ref + sin(b) ortho).
// f32 where bend.py has f64, evaluated where f32 keeps it exact: a triangle's cross product (p1 - p0) x (p2 - p0) is
// taken at its corner opposite its longest edge (the same vector, from its two shortest edges: a needle's, one of
// whose edges is next to nothing, would otherwise lose its direction to rounding); an angle between two vectors is
// atan2(|u x v|, u . v) of their unit vectors (exact near 0 and pi, where bend.py's arccos of the dot product is not),
// and pi / 2 when one of them is zero (bend.py's arccos(0)).
// A mirrored mesh (MIRRORED: its matrix has a negative determinant, so every cross product of its posed vertices points
// the other way) turns its lnor and its ortho axis back: its normals are the mirror image of the object-space ones.

const TURN: f32 = 6.2831855;
const HALF_PI: f32 = 1.5707964;
const GROUP: u32 = 64u;          // @workgroup_size
const MIRRORED: u32 = 1u;        // Fan.flags

struct Posed {
    pos: vec4<f32>,
    nrm: vec4<f32>,
}

struct Fan {
    center: u32,                 // a vertex at the fan's point
    refv: u32,                   // a vertex at the far end of its reference edge
    other: u32,                  // a vertex at the far end of its other edge
    flags: u32,
    corner0: u32,                // its corners: [corner0, corner1) of `corners`
    corner1: u32,
    edge0: u32,                  // its edges: [edge0, edge1) of `edges`
    edge1: u32,
    vert0: u32,                  // its vertices: [vert0, vert1) of `verts`
    vert1: u32,
    pad0: u32,
    pad1: u32,
}

struct Vert {
    vertex: u32,
    alpha: f32,
    beta: f32,
}

struct Sizes {
    faces: u32,
    fans: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> posed: array<Posed>;
@group(0) @binding(1) var<storage, read> face_first: array<u32>;      // per face its first triangle; one more at the end
@group(0) @binding(2) var<storage, read> tri_verts: array<u32>;       // 3 vertices per triangle, by face
@group(0) @binding(3) var<storage, read_write> face_nrm: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> fan_recs: array<Fan>;
@group(0) @binding(5) var<storage, read> corners: array<u32>;         // per corner: face, previous vertex, next vertex
@group(0) @binding(6) var<storage, read> edges: array<u32>;           // per edge its far-end vertex
@group(0) @binding(7) var<storage, read> verts: array<Vert>;
@group(0) @binding(8) var<uniform> sizes: Sizes;

fn unit(v: vec3<f32>) -> vec3<f32> {
    return v / max(length(v), 1e-30);
}

fn angle(u: vec3<f32>, v: vec3<f32>) -> f32 {
    let a = unit(u);
    let b = unit(v);
    let s = length(cross(a, b));
    let c = dot(a, b);
    if (s == 0.0 && c == 0.0) {
        return HALF_PI;
    }
    return atan2(s, c);
}

fn at(v: u32) -> vec3<f32> {
    return posed[v].pos.xyz;
}

// The item of an invocation of a dispatch that may spread its workgroups over y (bend.rs `groups`).
fn item(gid: vec3<u32>, groups: vec3<u32>) -> u32 {
    return gid.x + gid.y * groups.x * GROUP;
}

// A triangle's (p1 - p0) x (p2 - p0), twice its vector area, taken at its corner opposite its longest edge.
fn tri_cross(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>) -> vec3<f32> {
    let e0 = p2 - p1;
    let e1 = p0 - p2;
    let e2 = p1 - p0;
    let l0 = dot(e0, e0);
    let l1 = dot(e1, e1);
    let l2 = dot(e2, e2);
    if (l0 >= l1 && l0 >= l2) {
        return cross(e2, -e1);          // at p0: (p1 - p0) x (p2 - p0)
    }
    if (l1 >= l2) {
        return cross(e0, -e2);          // at p1: (p2 - p1) x (p0 - p1)
    }
    return cross(e1, -e0);              // at p2: (p0 - p2) x (p1 - p2)
}

@compute @workgroup_size(64)
fn faces(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let f = item(gid, groups);
    if (f >= sizes.faces) {
        return;
    }
    var sum = vec3<f32>(0.0);
    for (var t = face_first[f]; t < face_first[f + 1u]; t = t + 1u) {
        sum = sum + tri_cross(at(tri_verts[3u * t]), at(tri_verts[3u * t + 1u]), at(tri_verts[3u * t + 2u]));
    }
    face_nrm[f] = vec4<f32>(unit(sum), 0.0);
}

// The edge from the fan's point c to vertex w, flattened onto the plane square to ln, as a unit vector.
fn flat(w: u32, c: vec3<f32>, ln: vec3<f32>) -> vec3<f32> {
    let d = at(w) - c;
    return unit(d - ln * dot(d, ln));
}

@compute @workgroup_size(64)
fn fans(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = item(gid, groups);
    if (i >= sizes.fans) {
        return;
    }
    let fan = fan_recs[i];
    let c = at(fan.center);
    var sum = vec3<f32>(0.0);
    for (var k = fan.corner0; k < fan.corner1; k = k + 1u) {
        let face = corners[3u * k];
        sum = sum + face_nrm[face].xyz * angle(at(corners[3u * k + 1u]) - c, at(corners[3u * k + 2u]) - c);
    }
    let sgn: f32 = select(1.0, -1.0, (fan.flags & MIRRORED) != 0u);
    let ln = sgn * unit(sum);
    var ra: f32 = 0.0;
    for (var k = fan.edge0; k < fan.edge1; k = k + 1u) {
        ra = ra + angle(at(edges[k]) - c, ln);
    }
    ra = ra / f32(max(fan.edge1 - fan.edge0, 1u));
    let r = flat(fan.refv, c, ln);
    let oth = flat(fan.other, c, ln);
    let o = sgn * cross(ln, r);
    var span = TURN;
    if (dot(r, oth) < 1.0 - 1e-6) {
        let theta = angle(r, oth);
        span = select(theta, TURN - theta, dot(o, oth) < 0.0);
    }
    for (var k = fan.vert0; k < fan.vert1; k = k + 1u) {
        let v = verts[k];
        let a = select(TURN + v.alpha * (TURN - ra), v.alpha * ra, v.alpha >= 0.0);
        let b = v.beta * span;
        posed[v.vertex].nrm = vec4<f32>(unit(cos(a) * ln + sin(a) * (cos(b) * r + sin(b) * o)), 0.0);
    }
}
