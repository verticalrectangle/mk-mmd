// The world probe's mips and spherical harmonics (world.rs), after its faces are drawn (world.wgsl): each mip the
// 2 x 2 average of the one above, face by face (glossy lobes read the probe at a level by their roughness:
// world_light.wgsl), and the L1 harmonics of the radiance (diffuse light), projected from one small mip. cube.wgsl
// comes first.

struct ProbeImm {
    size: u32,                      // the written mip's (downsample) or the read mip's (harmonics) texels across
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var src: texture_2d_array<f32>;
@group(0) @binding(1) var dst: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(2) var<storage, read_write> harmonics: array<vec4<f32>, 4>;

var<immediate> pimm: ProbeImm;

@compute @workgroup_size(8, 8, 1)
fn downsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= pimm.size || gid.y >= pimm.size) {
        return;
    }
    let p = vec2<i32>(gid.xy) * 2;
    let f = i32(gid.z);
    let c = textureLoad(src, p, f, 0) + textureLoad(src, p + vec2<i32>(1, 0), f, 0)
          + textureLoad(src, p + vec2<i32>(0, 1), f, 0) + textureLoad(src, p + vec2<i32>(1, 1), f, 0);
    textureStore(dst, vec2<i32>(gid.xy), f, c * 0.25);
}

const SH_THREADS: u32 = 256u;
var<workgroup> partial: array<array<vec3<f32>, 4>, 256>;

// L1 harmonics of the radiance: per coefficient (Y00, Y1-1 (y), Y10 (z), Y11 (x), real, orthonormal) the integral of
// the radiance times it over the sphere, each texel weighted by its solid angle. Axes: glTF's.
@compute @workgroup_size(256, 1, 1)
fn project(@builtin(local_invocation_index) li: u32) {
    let n = pimm.size;
    let count = 6u * n * n;
    var acc = array<vec3<f32>, 4>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    let texel = 2.0 / f32(n);
    for (var i = li; i < count; i = i + SH_THREADS) {
        let face = i / (n * n);
        let xy = vec2<u32>(i % n, (i / n) % n);
        let st = (vec2<f32>(xy) + 0.5) * texel - 1.0;
        let d = cube_dir(face, st);
        let r2 = 1.0 + dot(st, st);
        let dw = texel * texel / (r2 * sqrt(r2));
        let l = textureLoad(src, vec2<i32>(xy), i32(face), 0).rgb * dw;
        acc[0] = acc[0] + l * 0.28209479;
        acc[1] = acc[1] + l * (0.48860251 * d.y);
        acc[2] = acc[2] + l * (0.48860251 * d.z);
        acc[3] = acc[3] + l * (0.48860251 * d.x);
    }
    partial[li] = acc;
    workgroupBarrier();
    for (var step = SH_THREADS / 2u; step > 0u; step = step / 2u) {
        if (li < step) {
            for (var k = 0u; k < 4u; k = k + 1u) {
                partial[li][k] = partial[li][k] + partial[li + step][k];
            }
        }
        workgroupBarrier();
    }
    if (li == 0u) {
        for (var k = 0u; k < 4u; k = k + 1u) {
            harmonics[k] = vec4<f32>(partial[0][k], 0.0);
        }
    }
}
