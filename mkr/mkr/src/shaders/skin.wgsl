// Posing: each mesh's vertices into the world on one time of the frame (glTF axes), shape keys first, then the
// object's matrix, then its armature: posed = sum_i w_i J_i (M (v + sum_k m_k d_k)); a vertex whose weights sum to 0
// is not deformed by the armature. Normals go through the same matrices (no non-uniform scale in MMD rigs) and are
// renormalised.

struct Rest {
    pos: vec4<f32>,
    nrm: vec4<f32>,
}

struct Skin {
    joints: vec4<u32>,      // 8 bone indices, two u16 per u32 (low half first)
    w0: vec4<f32>,
    w1: vec4<f32>,
}

struct Posed {
    pos: vec4<f32>,
    nrm: vec4<f32>,
}

struct Job {
    m0: vec4<f32>,          // the object's 3x4 world matrix, rows
    m1: vec4<f32>,
    m2: vec4<f32>,
    vertex_base: u32,
    vertex_count: u32,
    skin_base: u32,         // 0xffffffff: not skinned
    joint_base: u32,        // first bone of its armature in `joints`
    morph_base: u32,        // f32 index of its deltas (key-major: key, vertex, xyz)
    morph_first: u32,       // first entry of its active keys in `morphs`
    morph_count: u32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read> rest: array<Rest>;
@group(0) @binding(1) var<storage, read> skin: array<Skin>;
@group(0) @binding(2) var<storage, read> deltas: array<f32>;
@group(0) @binding(3) var<storage, read> joints: array<vec4<f32>>;     // 3 rows per bone
@group(0) @binding(4) var<storage, read> morphs: array<vec2<f32>>;     // (key index as f32 bits, weight)
@group(0) @binding(5) var<storage, read_write> posed: array<Posed>;

var<immediate> job: Job;

fn apply(r0: vec4<f32>, r1: vec4<f32>, r2: vec4<f32>, p: vec4<f32>) -> vec3<f32> {
    return vec3<f32>(dot(r0, p), dot(r1, p), dot(r2, p));
}

fn bone(i: u32, which: u32) -> vec4<f32> {
    return joints[(job.joint_base + i) * 3u + which];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local = gid.x;
    if (local >= job.vertex_count) {
        return;
    }
    let v = job.vertex_base + local;
    var p = rest[v].pos.xyz;
    let n = rest[v].nrm.xyz;
    for (var k = 0u; k < job.morph_count; k = k + 1u) {
        let m = morphs[job.morph_first + k];
        let key = bitcast<u32>(m.x);
        let at = job.morph_base + (key * job.vertex_count + local) * 3u;
        p = p + m.y * vec3<f32>(deltas[at], deltas[at + 1u], deltas[at + 2u]);
    }
    var wp = apply(job.m0, job.m1, job.m2, vec4<f32>(p, 1.0));
    var wn = apply(job.m0, job.m1, job.m2, vec4<f32>(n, 0.0));
    if (job.skin_base != 0xffffffffu) {
        let s = skin[job.skin_base + local];
        let total = dot(s.w0, vec4<f32>(1.0)) + dot(s.w1, vec4<f32>(1.0));
        if (total > 0.0) {
            var r0 = vec4<f32>(0.0);
            var r1 = vec4<f32>(0.0);
            var r2 = vec4<f32>(0.0);
            for (var i = 0u; i < 8u; i = i + 1u) {
                let w = select(s.w1[i - 4u], s.w0[min(i, 3u)], i < 4u);
                if (w != 0.0) {
                    let pair = s.joints[i / 2u];
                    let j = select(pair >> 16u, pair & 0xffffu, (i & 1u) == 0u);
                    r0 = r0 + w * bone(j, 0u);
                    r1 = r1 + w * bone(j, 1u);
                    r2 = r2 + w * bone(j, 2u);
                }
            }
            wp = apply(r0, r1, r2, vec4<f32>(wp, 1.0));
            wn = apply(r0, r1, r2, vec4<f32>(wn, 0.0));
        }
    }
    posed[v] = Posed(vec4<f32>(wp, 1.0), vec4<f32>(normalize(wn), 0.0));
}
