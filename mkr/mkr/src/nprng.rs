//! NumPy's random numbers, bit for bit: `np.random.default_rng(entropy)` (SeedSequence into PCG64) and the draws the
//! cut-out look takes from it (`uniform`, `integers(0, n)`), so the GPU cut-out (cutout.rs) cuts the paper with the
//! scissors and the fibres mkmmd.core.cutout draws on the CPU.

/// SeedSequence's hash constants (numpy/random/bit_generator.pyx).
const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const POOL: usize = 4;
/// PCG64's multiplier (PCG_DEFAULT_MULTIPLIER_128).
const PCG_MULT: u128 = (2549297995355413924u128 << 64) | 4865540595714422341u128;

fn hashmix(value: u32, hash: &mut u32) -> u32 {
    let mut v = value ^ *hash;
    *hash = hash.wrapping_mul(MULT_A);
    v = v.wrapping_mul(*hash);
    v ^ (v >> 16)
}

fn mix(x: u32, y: u32) -> u32 {
    let r = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    r ^ (r >> 16)
}

/// SeedSequence(entropy).generate_state(4, uint64): each entropy integer as its 32-bit words (least significant first,
/// 0 as one word), mixed into a pool of four.
fn seed_state(entropy: &[u64]) -> [u64; 4] {
    let mut words: Vec<u32> = Vec::new();
    for &e in entropy {
        if e == 0 {
            words.push(0);
        }
        let mut n = e;
        while n > 0 {
            words.push(n as u32);
            n >>= 32;
        }
    }
    let mut hash = INIT_A;
    let mut pool = [0u32; POOL];
    for (i, p) in pool.iter_mut().enumerate() {
        *p = hashmix(words.get(i).copied().unwrap_or(0), &mut hash);
    }
    for src in 0..POOL {
        for dst in 0..POOL {
            if src != dst {
                let h = hashmix(pool[src], &mut hash);
                pool[dst] = mix(pool[dst], h);
            }
        }
    }
    for &w in words.iter().skip(POOL) {
        for p in pool.iter_mut() {
            let h = hashmix(w, &mut hash);
            *p = mix(*p, h);
        }
    }
    let mut hash = INIT_B;
    let mut out = [0u32; 8];
    for (i, o) in out.iter_mut().enumerate() {
        let mut v = pool[i % POOL] ^ hash;
        hash = hash.wrapping_mul(MULT_B);
        v = v.wrapping_mul(hash);
        *o = v ^ (v >> 16);
    }
    [0, 1, 2, 3].map(|k| out[2 * k] as u64 | ((out[2 * k + 1] as u64) << 32))
}

/// `np.random.default_rng(entropy)`: PCG64 (XSL-RR) seeded through SeedSequence.
pub struct Rng {
    state: u128,
    inc: u128,
    /// The upper half of the last 64-bit draw, kept for the next 32-bit one (as numpy's PCG64 does).
    half: Option<u32>,
}

impl Rng {
    pub fn new(entropy: &[u64]) -> Rng {
        let v = seed_state(entropy);
        let seed = ((v[0] as u128) << 64) | v[1] as u128;
        let inc = ((v[2] as u128) << 64) | v[3] as u128;
        let mut r = Rng { state: 0, inc: (inc << 1) | 1, half: None };
        r.step();
        r.state = r.state.wrapping_add(seed);
        r.step();
        r
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let s = self.state;
        ((s >> 64) as u64 ^ s as u64).rotate_right((s >> 122) as u32)
    }

    fn next_u32(&mut self) -> u32 {
        if let Some(h) = self.half.take() {
            return h;
        }
        let v = self.next_u64();
        self.half = Some((v >> 32) as u32);
        v as u32
    }

    /// `random()`: a double in [0, 1).
    pub fn next_double(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    /// `uniform(low, high)`.
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_double()
    }

    /// `integers(0, n)` for 1 <= n <= 2^32: Lemire's bounded draw on 32-bit words.
    pub fn below(&mut self, n: u64) -> u64 {
        assert!((1..=1u64 << 32).contains(&n), "integers(0, {n})");
        let rng = n - 1;
        if rng == 0 {
            return 0;
        }
        if rng == 0xFFFF_FFFF {
            return self.next_u32() as u64;
        }
        let excl = rng + 1;
        let mut m = self.next_u32() as u64 * excl;
        let mut left = m & 0xFFFF_FFFF;
        if left < excl {
            let threshold = (0xFFFF_FFFFu64 - rng) % excl;
            while left < threshold {
                m = self.next_u32() as u64 * excl;
                left = m & 0xFFFF_FFFF;
            }
        }
        m >> 32
    }

    /// `uniform(-1, 1, (rows, cols)).astype(float32)`, row by row.
    pub fn grid(&mut self, rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols).map(|_| self.uniform(-1.0, 1.0) as f32).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // the values numpy 2.5 gives for the same seeds (np.random.default_rng(...)) and draws
    #[test]
    fn matches_numpy() {
        let mut r = Rng::new(&[1, 104, 4242]);
        assert_eq!([r.next_u64(), r.next_u64(), r.next_u64()], [8546538499255272736, 15883809108386625365, 17337320590008800686]);
        let mut r = Rng::new(&[1, 104, 4242]);
        assert_eq!(r.grid(1, 3), vec![-0.0733824372291565f32, 0.7221260070800781, 0.8797160387039185]);
        let mut r = Rng::new(&[7, 0, 99]);
        assert_eq!([r.below(128), r.below(128), r.below(128)], [89, 1, 35]);
        let mut r = Rng::new(&[3, 12, 31]);
        assert_eq!(r.uniform(0.0, 2.0 * std::f64::consts::PI), 1.2792397567486913);
        let mut r = Rng::new(&[7920, 1515]);
        assert_eq!([r.next_u64(), r.next_u64()], [8549163748157030130, 6602662613976384880]);
        let mut r = Rng::new(&[0]);
        assert_eq!([r.next_u64(), r.next_u64()], [11749869230777074271, 4976686463289251617]);
        // an entropy word past 32 bits and more words than the pool holds
        let mut r = Rng::new(&[(1 << 40) + 5, 3, 4, 5, 6]);
        assert_eq!([r.next_u64(), r.next_u64()], [11996807908867296993, 11851064134985068456]);
        // a bound that rejects some draws
        let mut r = Rng::new(&[5, 1, 99]);
        assert_eq!((0..5).map(|_| r.below(100)).collect::<Vec<_>>(), vec![42, 85, 68, 74, 81]);
    }
}
