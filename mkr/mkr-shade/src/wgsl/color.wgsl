// mkr-shade: colour maths (Mix / MixRGB blend modes, HSV / HSL, Hue Saturation Value, Invert, Gamma,
// Bright Contrast, RGB to BW, implicit colour -> float conversion).

// Luminance weights of scene-linear Rec.709 (the Y row of its RGB -> XYZ matrix), as Blender's colour management.
const MKS_LUMA: vec3f = vec3f(0.2126729, 0.7151522, 0.0721750);

fn mks_luma(c: vec3f) -> f32 {
  return dot(c, MKS_LUMA);
}

fn mks_rgb_to_hsv(rgb: vec3f) -> vec3f {
  let cmax = max(rgb.x, max(rgb.y, rgb.z));
  let cmin = min(rgb.x, min(rgb.y, rgb.z));
  let cdelta = cmax - cmin;
  let v = cmax;
  var s = 0.0;
  var h = 0.0;
  if (cmax != 0.0) {
    s = cdelta / cmax;
  }
  if (s != 0.0) {
    let c = (vec3f(cmax) - rgb) / cdelta;
    if (rgb.x == cmax) {
      h = c.z - c.y;
    } else if (rgb.y == cmax) {
      h = 2.0 + c.x - c.z;
    } else {
      h = 4.0 + c.y - c.x;
    }
    h = h / 6.0;
    if (h < 0.0) {
      h = h + 1.0;
    }
  }
  return vec3f(h, s, v);
}

fn mks_hsv_to_rgb(hsv: vec3f) -> vec3f {
  var h = hsv.x;
  let s = hsv.y;
  let v = hsv.z;
  if (s == 0.0) {
    return vec3f(v);
  }
  if (h == 1.0) {
    h = 0.0;
  }
  h = h * 6.0;
  let i = floor(h);
  let f = h - i;
  let p = v * (1.0 - s);
  let q = v * (1.0 - (s * f));
  let t = v * (1.0 - (s * (1.0 - f)));
  if (i == 0.0) {
    return vec3f(v, t, p);
  } else if (i == 1.0) {
    return vec3f(q, v, p);
  } else if (i == 2.0) {
    return vec3f(p, v, t);
  } else if (i == 3.0) {
    return vec3f(p, q, v);
  } else if (i == 4.0) {
    return vec3f(t, p, v);
  }
  return vec3f(v, p, q);
}

fn mks_rgb_to_hsl(rgb: vec3f) -> vec3f {
  let cmax = max(rgb.x, max(rgb.y, rgb.z));
  let cmin = min(rgb.x, min(rgb.y, rgb.z));
  let l = min(1.0, (cmax + cmin) / 2.0);
  var h = 0.0;
  var s = 0.0;
  if (cmax != cmin) {
    let d = cmax - cmin;
    s = select(d / (cmax + cmin), d / (2.0 - cmax - cmin), l > 0.5);
    if (cmax == rgb.x) {
      h = (rgb.y - rgb.z) / d + select(0.0, 6.0, rgb.y < rgb.z);
    } else if (cmax == rgb.y) {
      h = (rgb.z - rgb.x) / d + 2.0;
    } else {
      h = (rgb.x - rgb.y) / d + 4.0;
    }
  }
  h = h / 6.0;
  return vec3f(h, s, l);
}

fn mks_hue_to_rgb_c(nr: vec3f) -> vec3f {
  return clamp(nr, vec3f(0.0), vec3f(1.0));
}

fn mks_hsl_to_rgb(hsl: vec3f) -> vec3f {
  let h = hsl.x;
  let s = hsl.y;
  let l = hsl.z;
  let nr = abs(h * 6.0 - 3.0) - 1.0;
  let ng = 2.0 - abs(h * 6.0 - 2.0);
  let nb = 2.0 - abs(h * 6.0 - 4.0);
  let c = clamp(vec3f(nr, ng, nb), vec3f(0.0), vec3f(1.0));
  let chroma = (1.0 - abs(2.0 * l - 1.0)) * s;
  return (c - 0.5) * chroma + l;
}

// MixRGB / Mix (colour) blend modes. fac is used as given (callers clamp it when the node does).
fn mks_blend_mix(f: f32, a: vec4f, b: vec4f) -> vec4f {
  return vec4f(mix(a.rgb, b.rgb, f), a.a);
}

fn mks_blend_add(f: f32, a: vec4f, b: vec4f) -> vec4f {
  return vec4f(mix(a.rgb, a.rgb + b.rgb, f), a.a);
}

fn mks_blend_mult(f: f32, a: vec4f, b: vec4f) -> vec4f {
  return vec4f(mix(a.rgb, a.rgb * b.rgb, f), a.a);
}

fn mks_blend_screen(f: f32, a: vec4f, b: vec4f) -> vec4f {
  let fm = 1.0 - f;
  return vec4f(vec3f(1.0) - (vec3f(fm) + f * (vec3f(1.0) - b.rgb)) * (vec3f(1.0) - a.rgb), a.a);
}

fn mks_overlay1(f: f32, a: f32, b: f32) -> f32 {
  let fm = 1.0 - f;
  if (a < 0.5) {
    return a * (fm + 2.0 * f * b);
  }
  return 1.0 - (fm + 2.0 * f * (1.0 - b)) * (1.0 - a);
}

fn mks_blend_overlay(f: f32, a: vec4f, b: vec4f) -> vec4f {
  return vec4f(mks_overlay1(f, a.r, b.r), mks_overlay1(f, a.g, b.g), mks_overlay1(f, a.b, b.b), a.a);
}

fn mks_blend_sub(f: f32, a: vec4f, b: vec4f) -> vec4f {
  return vec4f(mix(a.rgb, a.rgb - b.rgb, f), a.a);
}

fn mks_div1(f: f32, a: f32, b: f32) -> f32 {
  let fm = 1.0 - f;
  return select(a, fm * a + f * a / b, b != 0.0);
}

fn mks_blend_div(f: f32, a: vec4f, b: vec4f) -> vec4f {
  return vec4f(mks_div1(f, a.r, b.r), mks_div1(f, a.g, b.g), mks_div1(f, a.b, b.b), a.a);
}

fn mks_blend_diff(f: f32, a: vec4f, b: vec4f) -> vec4f {
  return vec4f(mix(a.rgb, abs(a.rgb - b.rgb), f), a.a);
}

fn mks_blend_exclusion(f: f32, a: vec4f, b: vec4f) -> vec4f {
  return vec4f(max(mix(a.rgb, a.rgb + b.rgb - 2.0 * a.rgb * b.rgb, f), vec3f(0.0)), a.a);
}

fn mks_blend_dark(f: f32, a: vec4f, b: vec4f) -> vec4f {
  return vec4f(mix(a.rgb, min(a.rgb, b.rgb), f), a.a);
}

fn mks_blend_light(f: f32, a: vec4f, b: vec4f) -> vec4f {
  return vec4f(mix(a.rgb, max(a.rgb, b.rgb), f), a.a);
}

fn mks_dodge1(f: f32, a: f32, b: f32) -> f32 {
  if (a != 0.0) {
    let tmp = 1.0 - f * b;
    if (tmp <= 0.0) {
      return 1.0;
    }
    let r = a / tmp;
    return select(r, 1.0, r > 1.0);
  }
  return a;
}

fn mks_blend_dodge(f: f32, a: vec4f, b: vec4f) -> vec4f {
  return vec4f(mks_dodge1(f, a.r, b.r), mks_dodge1(f, a.g, b.g), mks_dodge1(f, a.b, b.b), a.a);
}

fn mks_burn1(f: f32, a: f32, b: f32) -> f32 {
  let fm = 1.0 - f;
  let tmp = fm + f * b;
  if (tmp <= 0.0) {
    return 0.0;
  }
  let r = 1.0 - (1.0 - a) / tmp;
  if (r < 0.0) {
    return 0.0;
  }
  if (r > 1.0) {
    return 1.0;
  }
  return r;
}

fn mks_blend_burn(f: f32, a: vec4f, b: vec4f) -> vec4f {
  return vec4f(mks_burn1(f, a.r, b.r), mks_burn1(f, a.g, b.g), mks_burn1(f, a.b, b.b), a.a);
}

fn mks_blend_hue(f: f32, a: vec4f, b: vec4f) -> vec4f {
  let fm = 1.0 - f;
  var r = a.rgb;
  let hsv2 = mks_rgb_to_hsv(b.rgb);
  if (hsv2.y != 0.0) {
    var hsv = mks_rgb_to_hsv(a.rgb);
    hsv.x = hsv2.x;
    let tmp = mks_hsv_to_rgb(hsv);
    r = mix(a.rgb, tmp, f);
  }
  return vec4f(r, a.a);
}

fn mks_blend_sat(f: f32, a: vec4f, b: vec4f) -> vec4f {
  let fm = 1.0 - f;
  var r = a.rgb;
  var hsv = mks_rgb_to_hsv(a.rgb);
  if (hsv.y != 0.0) {
    let hsv2 = mks_rgb_to_hsv(b.rgb);
    hsv.y = fm * hsv.y + f * hsv2.y;
    r = mks_hsv_to_rgb(hsv);
  }
  return vec4f(r, a.a);
}

fn mks_blend_val(f: f32, a: vec4f, b: vec4f) -> vec4f {
  let fm = 1.0 - f;
  var hsv = mks_rgb_to_hsv(a.rgb);
  let hsv2 = mks_rgb_to_hsv(b.rgb);
  hsv.z = fm * hsv.z + f * hsv2.z;
  return vec4f(mks_hsv_to_rgb(hsv), a.a);
}

fn mks_blend_color(f: f32, a: vec4f, b: vec4f) -> vec4f {
  var r = a.rgb;
  let hsv2 = mks_rgb_to_hsv(b.rgb);
  if (hsv2.y != 0.0) {
    var hsv = mks_rgb_to_hsv(a.rgb);
    hsv.x = hsv2.x;
    hsv.y = hsv2.y;
    let tmp = mks_hsv_to_rgb(hsv);
    r = mix(a.rgb, tmp, f);
  }
  return vec4f(r, a.a);
}

fn mks_blend_soft(f: f32, a: vec4f, b: vec4f) -> vec4f {
  let fm = 1.0 - f;
  let one = vec3f(1.0);
  let scr = one - (one - b.rgb) * (one - a.rgb);
  return vec4f(fm * a.rgb + f * ((one - a.rgb) * b.rgb * a.rgb + a.rgb * scr), a.a);
}

fn mks_blend_linear(f: f32, a: vec4f, b: vec4f) -> vec4f {
  return vec4f(a.rgb + f * (2.0 * b.rgb - vec3f(1.0)), a.a);
}

fn mks_hue_sat(hue: f32, sat: f32, value: f32, fac: f32, col: vec4f) -> vec4f {
  var hsv = mks_rgb_to_hsv(col.rgb);
  hsv.x = fract(hsv.x + hue + 0.5);
  hsv.y = clamp(hsv.y * sat, 0.0, 1.0);
  hsv.z = hsv.z * value;
  let rgb = mks_hsv_to_rgb(hsv);
  return vec4f(mix(col.rgb, rgb, fac), col.a);
}

fn mks_invert(fac: f32, col: vec4f) -> vec4f {
  return vec4f(mix(col.rgb, vec3f(1.0) - col.rgb, fac), col.a);
}

fn mks_gamma(col: vec4f, gamma: f32) -> vec4f {
  var c = col;
  if (c.r > 0.0) {
    c.r = mks_compatible_pow(c.r, gamma);
  }
  if (c.g > 0.0) {
    c.g = mks_compatible_pow(c.g, gamma);
  }
  if (c.b > 0.0) {
    c.b = mks_compatible_pow(c.b, gamma);
  }
  return c;
}

fn mks_bright_contrast(col: vec4f, bright: f32, contrast: f32) -> vec4f {
  let a = 1.0 + contrast;
  let b = bright - contrast * 0.5;
  return vec4f(max(a * col.rgb + vec3f(b), vec3f(0.0)), col.a);
}

fn mks_srgb_to_linear1(c: f32) -> f32 {
  return select(pow((c + 0.055) * (1.0 / 1.055), 2.4), select(0.0, c * (1.0 / 12.92), c >= 0.0), c < 0.04045);
}

fn mks_linear_to_srgb1(c: f32) -> f32 {
  return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, select(0.0, c * 12.92, c >= 0.0), c < 0.0031308);
}

// Image Texture colour output by Blender's alpha rules. `stored_premul`: the texture holds premultiplied colour.
// `clear`: alpha must not touch colour (channel-packed / ignored alpha, or non-colour data).
// `alpha_used`: the node's Alpha output is linked (then Color is straight, else premultiplied).
fn mks_image_color(t: vec4f, stored_premul: bool, clear: bool, alpha_used: bool) -> vec4f {
  if (clear) {
    return vec4f(t.rgb, 1.0);
  }
  if (stored_premul) {
    if (alpha_used) {
      let inv = select(0.0, 1.0 / t.a, t.a != 0.0);
      return vec4f(t.rgb * inv, 1.0);
    }
    return vec4f(t.rgb, 1.0);
  }
  if (alpha_used) {
    return vec4f(t.rgb, 1.0);
  }
  return vec4f(t.rgb * t.a, 1.0);
}
