//! Which frames to draw and through which camera: the frame spec of the command line, the cut (the shot table's entry
//! in force at a frame, as mkmmd.core.shotstyle.shot_at finds it) and what leaves a frame to Blender.

use anyhow::{Result, bail};

use crate::bake::{MediumWin, Shot};

/// Blender frame numbers from `a`, `a:b` (inclusive), `a:b:step` items joined by commas; sorted, each once.
pub fn parse_spec(spec: &str) -> Result<Vec<i64>> {
    let mut out = Vec::new();
    for item in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let parts: Vec<&str> = item.split(':').map(str::trim).collect();
        let num = |s: &str| -> Result<i64> {
            s.parse::<i64>().map_err(|_| anyhow::anyhow!("--frames: {item:?} is not a frame, a:b or a:b:step"))
        };
        match parts.as_slice() {
            [a] => out.push(num(a)?),
            [a, b] | [a, b, _] => {
                let (a, b) = (num(a)?, num(b)?);
                let step = if parts.len() == 3 { num(parts[2])? } else { 1 };
                if step <= 0 {
                    bail!("--frames: {item:?} needs a positive step");
                }
                if b < a {
                    bail!("--frames: {item:?} ends before it starts");
                }
                let mut f = a;
                while f <= b {
                    out.push(f);
                    f += step;
                }
            }
            _ => bail!("--frames: {item:?} is not a frame, a:b or a:b:step"),
        }
    }
    if out.is_empty() {
        bail!("--frames: no frames in {spec:?}");
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

/// The shot-table entry in force at `frame`: the last one that starts at or before it (a cut belongs to the incoming
/// shot), the first one before the first cut; plate shots are not in the cut.
pub fn shot_at(shots: &[Shot], frame: i64) -> Option<&Shot> {
    let cut = || shots.iter().filter(|s| !s.plate);
    let mut best: Option<&Shot> = None;
    for e in cut() {
        if e.from <= frame && best.is_none_or(|b| e.from >= b.from) {
            best = Some(e);
        }
    }
    best.or_else(|| cut().min_by_key(|e| e.from))
}

/// The [[medium]] window holding `frame` (from <= frame < to).
pub fn medium_at(wins: &[MediumWin], frame: i64) -> Option<&MediumWin> {
    wins.iter().find(|w| w.from <= frame && frame < w.to)
}

/// Whether bake frame `k` falls in one of the half-open ranges [k0, k1).
pub fn shows(ranges: &[[i64; 2]], k: usize) -> bool {
    let k = k as i64;
    ranges.iter().any(|r| r[0] <= k && k < r[1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn shot(name: &str, from: i64, plate: bool) -> Shot {
        Shot { name: name.into(), from, to: from + 10, cameras: BTreeMap::new(), styles: None, plate, keyed: None }
    }

    #[test]
    fn spec_items() {
        assert_eq!(parse_spec("5").unwrap(), vec![5]);
        assert_eq!(parse_spec("3:6").unwrap(), vec![3, 4, 5, 6]);
        assert_eq!(parse_spec("10:20:5, 1 ,3:4").unwrap(), vec![1, 3, 4, 10, 15, 20]);
        assert_eq!(parse_spec("4:4,4").unwrap(), vec![4]);
        assert!(parse_spec("6:3").is_err());
        assert!(parse_spec("1:5:0").is_err());
        assert!(parse_spec("x").is_err());
        assert!(parse_spec("").is_err());
    }

    #[test]
    fn cut_belongs_to_the_incoming_shot() {
        let shots = vec![shot("a", 91, false), shot("b", 128, false), shot("p", 120, true), shot("c", 132, false)];
        assert_eq!(shot_at(&shots, 91).unwrap().name, "a");
        assert_eq!(shot_at(&shots, 127).unwrap().name, "a");
        assert_eq!(shot_at(&shots, 128).unwrap().name, "b");
        assert_eq!(shot_at(&shots, 125).unwrap().name, "a", "plates are not in the cut");
        assert_eq!(shot_at(&shots, 50).unwrap().name, "a", "before the first cut: the first shot");
        assert_eq!(shot_at(&shots, 5000).unwrap().name, "c");
        assert!(shot_at(&[], 5).is_none());
    }

    #[test]
    fn ranges_are_half_open() {
        assert!(shows(&[[0, 3]], 0));
        assert!(shows(&[[0, 3]], 2));
        assert!(!shows(&[[0, 3]], 3));
        assert!(shows(&[[0, 1], [5, 9]], 5));
    }
}
