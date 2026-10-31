// The layers' flat surfaces (layers_gpu.rs), appended to the drawing module after its materials. Every flat pass
// writes premultiplied RGBA over a transparent clear: the film keeps colour and coverage apart.

// A figure's coverage: every triangle of its objects, opaque (Workbench's object colour ignores the materials).
@fragment
fn fl_cover() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}

// The keep mask's flat emission of a surface the depth prepass left in front: a kept material white, the rest black.
@fragment
fn fl_white() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}

@fragment
fn fl_black() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
