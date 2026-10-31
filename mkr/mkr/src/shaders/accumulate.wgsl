// The film, part 1: samples' pictures spread over the pixels around them with the filter's weights, up to four samples a
// pass (the film is read and written once for all of them).

struct Accumulate {
    width: u32,
    height: u32,
    first_sample: u32,              // the samples' taps in `taps` from here
    count: u32,                     // how many of the four pictures hold this pass's samples
    first: u32,                     // 1 on the frame's first pass: the film starts from it
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var sample0: texture_2d<f32>;
@group(0) @binding(1) var sample1: texture_2d<f32>;
@group(0) @binding(2) var sample2: texture_2d<f32>;
@group(0) @binding(3) var sample3: texture_2d<f32>;
@group(0) @binding(4) var<storage, read_write> film: array<vec4<f32>>;
// per sample 25 weights, entry (dy + 2) * 5 + dx + 2 for the sample drawn on p + (dx, dy); 28 floats per sample
@group(0) @binding(5) var<storage, read> taps: array<f32>;

var<immediate> acc: Accumulate;

fn load(k: u32, q: vec2<i32>) -> vec3<f32> {
    switch k {
        case 0u: { return textureLoad(sample0, q, 0).rgb; }
        case 1u: { return textureLoad(sample1, q, 0).rgb; }
        case 2u: { return textureLoad(sample2, q, 0).rgb; }
        default: { return textureLoad(sample3, q, 0).rgb; }
    }
}

@compute @workgroup_size(8, 8)
fn accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= acc.width || gid.y >= acc.height) {
        return;
    }
    let p = vec2<i32>(gid.xy);
    var sum = vec4<f32>(0.0);
    for (var k = 0u; k < acc.count; k = k + 1u) {
        let base = (acc.first_sample + k) * 28u;
        for (var dy = -2; dy <= 2; dy = dy + 1) {
            for (var dx = -2; dx <= 2; dx = dx + 1) {
                let q = p + vec2<i32>(dx, dy);
                let w = taps[base + u32((dy + 2) * 5 + dx + 2)];
                if (w > 0.0 && q.x >= 0 && q.y >= 0 && q.x < i32(acc.width) && q.y < i32(acc.height)) {
                    // EEVEE's film drops negative and NaN radiance (measured by ShaderGraph)
                    let c = load(k, q);
                    let v = select(vec3<f32>(0.0), max(c, vec3<f32>(0.0)), c == c);
                    sum = sum + vec4<f32>(v * w, w);
                }
            }
        }
    }
    let i = gid.y * acc.width + gid.x;
    if (acc.first == 1u) {
        film[i] = sum;
    } else {
        film[i] = film[i] + sum;
    }
}
