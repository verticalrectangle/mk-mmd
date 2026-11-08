//! The scene bake `mk build` writes beside the .blend (mkmmd/blender/build/bake.py; v5 adds `lit`): bake.json, bake.bin
//! (float32, offsets marked FLOAT), lit.bin (bytes, offsets marked BYTE) and the view LUT. Everything positional is in
//! glTF's axes: Blender's (x, y, z) is glTF's (x, z, -y).

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::Value;

/// The oldest bake mkr reads: v5 carries the lit scene (lights, materials, meshes, the view LUT).
pub const LIT_VERSION: u32 = 5;

pub struct Bake {
    pub dir: PathBuf,
    pub doc: Doc,
    /// bake.json as JSON (mkr-shade reads the node graphs from it).
    pub raw: Value,
    bin: Option<memmap2::Mmap>,
    lit_bin: Option<memmap2::Mmap>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Doc {
    pub version: u32,
    pub fps: f64,
    pub frame0: i64,
    pub frames: usize,
    pub outputs: Vec<OutputRec>,
    #[serde(default)]
    pub shots: Vec<Shot>,
    #[serde(default)]
    pub armatures: Vec<ArmatureRec>,
    #[serde(default)]
    pub cameras: Vec<CameraRec>,
    #[serde(default)]
    pub objects: Vec<ObjectRec>,
    #[serde(default)]
    pub blocks: Vec<BlockRec>,
    #[serde(default)]
    pub medium: Vec<MediumWin>,
    /// How many screen type objects the scene has (v4; their records are `lit.screen`'s).
    #[serde(default, deserialize_with = "null_default")]
    pub screen: usize,
    #[serde(default)]
    pub lit: Option<Lit>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct OutputRec {
    pub name: String,
    pub size: [u32; 2],
}

/// A shot-table entry: Blender frames `from`..`to`, the camera per output, the render-time look per output.
#[derive(Deserialize, Debug, Clone)]
pub struct Shot {
    pub name: String,
    pub from: i64,
    #[serde(default)]
    pub to: i64,
    #[serde(default)]
    pub cameras: BTreeMap<String, String>,
    #[serde(default)]
    pub styles: Option<BTreeMap<String, Value>>,
    #[serde(default)]
    pub plate: bool,
    /// The Blender frames [first, last] its camera is keyed over (a plate's, a transition's): beyond them it is not.
    #[serde(default)]
    pub keyed: Option<[i64; 2]>,
}

impl Shot {
    /// The kind of render-time look (silhouette, vector, reflection) the shot has for `output`, if any.
    pub fn look(&self, output: &str) -> Option<String> {
        let spec = self.styles.as_ref()?.get(output)?;
        let obj = spec.as_object()?;
        obj.keys().next().cloned()
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct ArmatureRec {
    pub name: String,
    pub bones: Vec<String>,
    pub offset: usize,
}

/// A shot camera: per frame at `offset + 15 k` its 3x4 world matrix (column by column), lens (mm), shift x, shift y;
/// at `clip_at + 2 k` its clip range.
#[derive(Deserialize, Debug, Clone)]
pub struct CameraRec {
    pub name: String,
    pub sensor: f64,
    #[serde(default)]
    pub sensor_height: f64,
    #[serde(default = "auto")]
    pub fit: String,
    pub offset: usize,
    pub clip_at: usize,
    #[serde(default)]
    pub fisheye: Option<Value>,
}

fn auto() -> String {
    "AUTO".into()
}

#[derive(Deserialize, Debug, Clone)]
pub struct ObjectRec {
    pub name: String,
    #[serde(rename = "type", default)]
    pub kind: String,
    #[serde(default)]
    pub baked: Option<String>,
    #[serde(default)]
    pub why: Option<String>,
    #[serde(default)]
    pub shows: Vec<[i64; 2]>,
    #[serde(default)]
    pub text: bool,
}

#[derive(Deserialize, Debug, Clone)]
pub struct BlockRec {
    pub name: String,
    #[serde(default)]
    pub why: String,
    #[serde(default)]
    pub shows: Vec<[i64; 2]>,
}

/// A [[medium]] window: Blender frames from <= f < to, the look per output ({} or absent: none).
#[derive(Deserialize, Debug, Clone)]
pub struct MediumWin {
    #[serde(default)]
    pub index: usize,
    pub from: i64,
    pub to: i64,
    #[serde(default)]
    pub styles: BTreeMap<String, Value>,
}

impl MediumWin {
    pub fn look(&self, output: &str) -> Option<String> {
        self.styles.get(output)?.as_object()?.keys().next().cloned()
    }
}

// ============================================================================================================ v5: lit
#[derive(Deserialize, Debug, Clone)]
pub struct Lit {
    #[serde(default = "lit_bin")]
    pub file: String,
    pub view: View,
    #[serde(default)]
    pub film: Film,
    #[serde(default)]
    pub eevee: Value,
    /// Every scene.render property by identifier (motion_blur_position, ...).
    #[serde(default)]
    pub render: Value,
    #[serde(default)]
    pub world: Option<Value>,
    #[serde(default)]
    pub lights: Vec<LightRec>,
    #[serde(default)]
    pub materials: BTreeMap<String, MaterialRec>,
    #[serde(default)]
    pub groups: BTreeMap<String, Value>,
    #[serde(default)]
    pub images: BTreeMap<String, ImageRec>,
    #[serde(default)]
    pub cameras: BTreeMap<String, LitCamera>,
    #[serde(default)]
    pub meshes: Vec<MeshRec>,
    /// Objects Blender renders that the lit bake cannot hold, with the bake frames [k0, k1) they show on.
    #[serde(default)]
    pub left_out: Vec<BlockRec>,
    /// The screen type layer: its meshes, cameras, pass and view (null: the scene has none).
    #[serde(default)]
    pub screen: Option<ScreenRec>,
    /// The layers' points: per key its world position on the frames a layer asks for it.
    #[serde(default, deserialize_with = "null_default")]
    pub points: BTreeMap<String, PointRec>,
}

/// The screen type pass (styles.Looks._screen_pass): its meshes (frame by frame, in the world: the plane z = 0, one unit a
/// frame height), an orthographic camera per output, the samples, and its view transform (Standard) as a LUT.
#[derive(Deserialize, Debug, Clone)]
pub struct ScreenRec {
    #[serde(default)]
    pub meshes: Vec<MeshRec>,
    #[serde(default)]
    pub cameras: BTreeMap<String, ScreenCamera>,
    #[serde(rename = "pass", default)]
    pub pass: ScreenPass,
    pub lut: LutRec,
    #[serde(default)]
    pub zero: Option<[f32; 3]>,
    /// Screen type the bake could not hold: the frames it is on screen keep their layer Blender's.
    #[serde(default, deserialize_with = "null_default")]
    pub left_out: Vec<ScreenLeftOut>,
}

/// A screen type object the bake left out, with the output it is made for (null: every output) and the Blender frames
/// [on, off) it is on screen (null: always).
#[derive(Deserialize, Debug, Clone)]
pub struct ScreenLeftOut {
    pub name: String,
    #[serde(default)]
    pub why: String,
    #[serde(default)]
    pub aspect: Option<String>,
    #[serde(default)]
    pub show: Option<[i64; 2]>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct ScreenCamera {
    pub ortho_scale: f32,
    #[serde(default = "auto")]
    pub sensor_fit: String,
    pub clip: [f32; 2],
    /// The 3x4 camera-to-world matrix, column by column, glTF axes.
    pub matrix: [f32; 12],
}

#[derive(Deserialize, Debug, Clone, Default)]
pub struct ScreenPass {
    #[serde(default)]
    pub samples: ScreenSamples,
}

/// EEVEE samples of the pass: `default`, or a look's own ({shot or "medium <i>": {output: n}}).
#[derive(Deserialize, Debug, Clone)]
pub struct ScreenSamples {
    #[serde(default = "sixteen")]
    pub default: u32,
    #[serde(default)]
    pub looks: BTreeMap<String, BTreeMap<String, u32>>,
}

impl Default for ScreenSamples {
    fn default() -> Self {
        ScreenSamples { default: 16, looks: BTreeMap::new() }
    }
}

fn sixteen() -> u32 {
    16
}

/// A layer's world point: x y z per bake frame at FLOAT `offset + 3 k` (NaN where no layer asks for it or the expression
/// failed), the runs of bake frames [k0, k1) and the shot each asks it through, the expression's failures.
#[derive(Deserialize, Debug, Clone)]
pub struct PointRec {
    #[serde(default)]
    pub expr: String,
    pub offset: usize,
    #[serde(default)]
    pub frames: Vec<(i64, i64, String)>,
    #[serde(default)]
    pub errors: Vec<Value>,
}

fn lit_bin() -> String {
    "lit.bin".into()
}

#[derive(Deserialize, Debug, Clone)]
pub struct View {
    #[serde(default)]
    pub transform: String,
    #[serde(default)]
    pub look: String,
    #[serde(default)]
    pub exposure: f64,
    #[serde(default = "one")]
    pub gamma: f64,
    #[serde(default)]
    pub curve: bool,
    #[serde(default)]
    pub display: String,
    #[serde(default = "one")]
    pub dither: f64,
    pub lut: LutRec,
    #[serde(default)]
    pub zero: Option<[f32; 3]>,
}

fn one() -> f64 {
    1.0
}

#[derive(Deserialize, Debug, Clone)]
pub struct LutRec {
    pub file: String,
    pub n: usize,
    pub lo: f64,
    pub hi: f64,
    #[serde(default)]
    pub zero: Option<[f32; 3]>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Film {
    #[serde(default = "filter_size")]
    pub filter_size: f64,
    #[serde(default)]
    pub transparent: bool,
}

fn filter_size() -> f64 {
    1.5
}

impl Default for Film {
    fn default() -> Self {
        Film { filter_size: 1.5, transparent: false }
    }
}

/// A light: per frame at `offset + 22 k` (or k = 0 when `frames` is 1): position, forward, right, up, colour, energy,
/// size_x, size_y, spot_size, spot_blend, radius, shown.
#[derive(Deserialize, Debug, Clone)]
pub struct LightRec {
    pub name: String,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub shape: Option<String>,
    #[serde(default)]
    pub use_shadow: bool,
    #[serde(default)]
    pub visible: BTreeMap<String, bool>,
    #[serde(default)]
    pub props: Value,
    #[serde(default = "one_usize")]
    pub frames: usize,
    pub offset: usize,
}

fn one_usize() -> usize {
    1
}

pub const LIGHT_FLOATS: usize = 22;

#[derive(Deserialize, Debug, Clone)]
pub struct MaterialRec {
    #[serde(default)]
    pub settings: Value,
    #[serde(default)]
    pub tree: Option<Value>,
    #[serde(default)]
    pub anim: Vec<AnimRec>,
}

/// An animated node socket: `size` floats per frame at `offset + size k` (k = 0 when `frames` is 1).
#[derive(Deserialize, Debug, Clone)]
pub struct AnimRec {
    pub node: String,
    pub socket: String,
    #[serde(default)]
    pub io: String,
    #[serde(default = "one_usize")]
    pub size: usize,
    pub offset: usize,
    #[serde(default = "one_usize")]
    pub frames: usize,
}

#[derive(Deserialize, Debug, Clone)]
pub struct ImageRec {
    #[serde(default)]
    pub file: String,
    #[serde(default)]
    pub colorspace: String,
    #[serde(default)]
    pub alpha_mode: String,
    #[serde(default)]
    pub size: [u32; 2],
    #[serde(default)]
    pub missing: bool,
}

/// A camera's per-frame lens data at `offset + 8 k`: use_dof, focus_distance, fstop, blades, rotation, ratio,
/// ortho_scale, sensor_width.
#[derive(Deserialize, Debug, Clone)]
pub struct LitCamera {
    #[serde(rename = "type", default = "persp")]
    pub kind: String,
    pub offset: usize,
}

fn persp() -> String {
    "PERSP".into()
}

pub const LIT_CAMERA_FLOATS: usize = 8;

#[derive(Deserialize, Debug, Clone)]
pub struct MeshRec {
    pub name: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default, deserialize_with = "null_default")]
    pub materials: Vec<Option<String>>,
    #[serde(default, deserialize_with = "null_default")]
    pub visible: MeshVisible,
    /// The only output the object renders for (mk's per-output objects).
    #[serde(default)]
    pub aspect: Option<String>,
    #[serde(default)]
    pub object: Option<ObjectProps>,
    /// What its materials' Attribute nodes of type Object, Instancer or View Layer read off it, by "KIND:name":
    /// RGBA (as Blender converts the property; zeros where it has none) on every frame.
    #[serde(default, deserialize_with = "null_default")]
    pub object_attrs: BTreeMap<String, ObjectAttrRec>,
    #[serde(default)]
    pub shown: Option<usize>,
    #[serde(default)]
    pub matrix: Option<MatrixRec>,
    #[serde(default)]
    pub armature: Option<String>,
    #[serde(default, deserialize_with = "null_default")]
    pub dqs: bool,
    #[serde(default, deserialize_with = "null_default")]
    pub vertices: usize,
    #[serde(default, deserialize_with = "null_default")]
    pub attributes: BTreeMap<String, AttrRec>,
    #[serde(default)]
    pub triangles: Option<TriRec>,
    #[serde(default, deserialize_with = "null_default")]
    pub slots: Vec<[usize; 3]>,
    #[serde(default)]
    pub morphs: Option<MorphRec>,
    /// A mesh that bends (skinned or shape-keyed): how its normals are rebuilt from its posed vertices on every frame.
    /// Kept as JSON so a bake written before the format settled still opens; `crate::bend` reads it (and refuses an
    /// older record with a clear error).
    #[serde(default)]
    pub bend: Option<Value>,
    #[serde(default)]
    pub by_frame: Option<ByFrame>,
    /// The object whose look record selects it (an instance: its instancer); older bakes leave it out.
    #[serde(default)]
    pub record: Option<String>,
    #[serde(default)]
    pub instance: Option<InstanceRec>,
    /// Screen type: the Blender frames [on, off) it is on screen (null: always), and whether a silhouette reverses it
    /// out of its figure.
    #[serde(default)]
    pub show: Option<[i64; 2]>,
    #[serde(default, deserialize_with = "null_default")]
    pub knockout: bool,
    /// Negative screen type: not laid over the picture but inverting it under its coverage, a layer of its own
    /// (`negative/<f>.png`, crate::layers); never in the screen layer. Never knock-out type too (the text stage refuses
    /// both).
    #[serde(default, deserialize_with = "null_default")]
    pub negative: bool,
}

impl MeshRec {
    /// The object an object pattern matches for this mesh: its `record`, else its instancer, else itself.
    pub fn record_name(&self) -> &str {
        match (&self.record, &self.instance) {
            (Some(r), _) => r,
            (None, Some(i)) => &i.by,
            (None, None) => &self.name,
        }
    }
}

/// A mesh drawn as one instance of `of` by the instancer `by`.
#[derive(Deserialize, Debug, Clone)]
pub struct InstanceRec {
    pub by: String,
    #[serde(default)]
    pub of: String,
}

/// Object properties materials read (Object Info).
#[derive(Deserialize, Debug, Clone)]
pub struct ObjectProps {
    #[serde(default)]
    pub color: Option<[f32; 4]>,
    #[serde(default)]
    pub pass_index: f64,
}

/// An object attribute of a mesh: 4 floats per frame at `offset + 4 k` (k = 0 when `frames` is 1).
#[derive(Deserialize, Debug, Clone)]
pub struct ObjectAttrRec {
    pub offset: usize,
    #[serde(default = "one_usize")]
    pub frames: usize,
}

impl ObjectAttrRec {
    /// Its value on bake frame `k`.
    pub fn at(&self, bake: &Bake, k: usize) -> Result<[f32; 4]> {
        let v = bake.f(self.offset + 4 * if self.frames == 1 { 0 } else { k }, 4)?;
        Ok([v[0], v[1], v[2], v[3]])
    }
}

/// A JSON null as the type's default (the bake writes null for fields a record kind does not use).
fn null_default<'de, D, T>(d: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    Ok(Option::<T>::deserialize(d)?.unwrap_or_default())
}

#[derive(Deserialize, Debug, Clone)]
pub struct MeshVisible {
    #[serde(default = "yes")]
    pub camera: bool,
    #[serde(default = "yes")]
    pub shadow: bool,
    #[serde(default)]
    pub holdout: bool,
}

fn yes() -> bool {
    true
}

impl Default for MeshVisible {
    fn default() -> Self {
        MeshVisible { camera: true, shadow: true, holdout: false }
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct MatrixRec {
    pub offset: usize,
    #[serde(default = "one_usize")]
    pub frames: usize,
}

#[derive(Deserialize, Debug, Clone)]
pub struct AttrRec {
    pub offset: usize,
    pub comp: usize,
    #[serde(rename = "type", default = "f32_name")]
    pub ty: String,
}

fn f32_name() -> String {
    "f32".into()
}

#[derive(Deserialize, Debug, Clone)]
pub struct TriRec {
    pub offset: usize,
    pub count: usize,
}

#[derive(Deserialize, Debug, Clone)]
pub struct MorphRec {
    pub names: Vec<String>,
    pub deltas: usize,
    pub values: usize,
}

#[derive(Deserialize, Debug, Clone)]
pub struct ByFrame {
    pub geoms: Vec<Geom>,
    pub frames: Vec<i64>,
    /// Screen type, fitted (bake_lit.py SCREEN; crate::screen_geo decodes it): per bake frame the BYTE offset of its
    /// pose in lit.bin, -1 hidden.
    #[serde(default)]
    pub poses: Option<Vec<i64>>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Geom {
    pub vertices: usize,
    #[serde(default)]
    pub attributes: BTreeMap<String, AttrRec>,
    pub triangles: TriRec,
    #[serde(default)]
    pub slots: Vec<[usize; 3]>,
    /// A fitted geometry (screen type): its prototypes and islands; `attributes` is then empty, `vertices`, `slots` and
    /// `triangles.count` are the frames' it makes and `triangles.offset` only names it.
    #[serde(default)]
    pub fit: Option<FitRec>,
}

/// A fitted geometry's prototype shapes, its islands (each a prototype through its map) and the attributes its poses
/// hold one value of per island.
#[derive(Deserialize, Debug, Clone)]
pub struct FitRec {
    pub protos: ProtoRec,
    pub islands: IslandsRec,
    /// (attribute key, components)
    #[serde(default)]
    pub attrs: Vec<(String, usize)>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct ProtoRec {
    pub count: usize,
    /// BYTE: u32 x 4 per prototype: first vertex, vertices, first triangle, triangles.
    pub table: usize,
    pub vertices: usize,
    /// Per prototype vertex: position and normal (glTF world), and the attributes that vary over an island.
    pub attributes: BTreeMap<String, AttrRec>,
    /// u32, three per triangle, its prototype's own vertex numbers.
    pub triangles: TriRec,
}

#[derive(Deserialize, Debug, Clone)]
pub struct IslandsRec {
    pub count: usize,
    /// BYTE: u32 x 4 per island: prototype, slot, first vertex, first triangle.
    pub table: usize,
}

impl Bake {
    /// Read bake.json and map bake.bin (and lit.bin when the bake has the lit scene).
    pub fn open(dir: &Path) -> Result<Bake> {
        let doc_path = dir.join("bake.json");
        let text = std::fs::read_to_string(&doc_path)
            .with_context(|| format!("no bake in {}: run `mk build --bake`", dir.display()))?;
        let raw: Value = serde_json::from_str(&text).with_context(|| format!("{} is not JSON", doc_path.display()))?;
        let doc = Doc::deserialize(&raw).with_context(|| format!("{} is not a bake", doc_path.display()))?;
        let bin = map(&dir.join("bake.bin"))?;
        let lit_bin = match &doc.lit {
            Some(lit) => map(&dir.join(&lit.file))?,
            None => None,
        };
        if let Some(b) = &bin {
            if b.len() % 4 != 0 {
                bail!("bake.bin is not a float32 array ({} bytes)", b.len());
            }
        }
        Ok(Bake { dir: dir.to_path_buf(), doc, raw, bin, lit_bin })
    }

    /// The bake refuses to be drawn lit without v5's `lit` (`mk build --bake` writes it).
    pub fn lit(&self) -> Result<&Lit> {
        match &self.doc.lit {
            Some(l) if self.doc.version >= LIT_VERSION => Ok(l),
            _ => bail!(
                "the bake in {} (version {}) has no lit scene: mkr needs version {LIT_VERSION} (`mk build --bake`)",
                self.dir.display(),
                self.doc.version
            ),
        }
    }

    /// bake.bin as float32.
    pub fn floats(&self) -> &[f32] {
        match &self.bin {
            Some(m) => bytemuck::cast_slice(&m[..]),
            None => &[],
        }
    }

    /// `n` floats of bake.bin from `offset`.
    pub fn f(&self, offset: usize, n: usize) -> Result<&[f32]> {
        let all = self.floats();
        all.get(offset..offset + n)
            .with_context(|| format!("bake.bin is too short: floats {offset}..{} of {}", offset + n, all.len()))
    }

    /// lit.bin's bytes.
    pub fn lit_bytes(&self) -> &[u8] {
        match &self.lit_bin {
            Some(m) => &m[..],
            None => &[],
        }
    }

    /// `n` bytes of lit.bin from `offset`.
    pub fn bytes(&self, offset: usize, n: usize) -> Result<&[u8]> {
        let all = self.lit_bytes();
        all.get(offset..offset + n)
            .with_context(|| format!("lit.bin is too short: bytes {offset}..{} of {}", offset + n, all.len()))
    }

    pub fn camera(&self, name: &str) -> Option<&CameraRec> {
        self.doc.cameras.iter().find(|c| c.name == name)
    }

    pub fn output(&self, name: &str) -> Option<&OutputRec> {
        self.doc.outputs.iter().find(|o| o.name == name)
    }

    /// Bake frame index of a Blender frame, if the clip has it.
    pub fn k(&self, frame: i64) -> Option<usize> {
        let k = frame - self.doc.frame0;
        (k >= 0 && (k as usize) < self.doc.frames).then_some(k as usize)
    }
}

fn map(path: &Path) -> Result<Option<memmap2::Mmap>> {
    let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    if file.metadata()?.len() == 0 {
        return Ok(None);
    }
    // SAFETY: the bake is written once by `mk build` and only read while mkr runs.
    let m = unsafe { memmap2::Mmap::map(&file) }.with_context(|| format!("cannot map {}", path.display()))?;
    Ok(Some(m))
}
