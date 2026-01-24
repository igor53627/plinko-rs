use clap::Parser;
use aes::Aes128;
use ctr::cipher::{KeyIvInit, StreamCipher};
use plinko::binomial::binomial_sample;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about = "Binomial sampler benchmark: BTPE+inverse vs current")]
struct Args {
    /// Number of trials (n)
    #[arg(long)]
    n: u64,

    /// Numerator of p
    #[arg(long)]
    num: u64,

    /// Denominator of p
    #[arg(long)]
    denom: u64,

    /// Number of samples to draw
    #[arg(long, default_value = "100000")]
    samples: u64,

    /// PRF seed for deterministic sampling
    #[arg(long, default_value = "123456789")]
    seed: u64,

    /// RNG used to generate PRF outputs (r values)
    #[arg(long, default_value = "aes", value_parser = ["splitmix", "aes"])]
    prf_rng: String,

    /// Stream seed for PRF RNG (defaults to --seed)
    #[arg(long)]
    prf_stream_seed: Option<u64>,

    /// RNG for BTPE (splitmix or aes)
    #[arg(long, default_value = "splitmix", value_parser = ["splitmix", "aes"])]
    btpe_rng: String,

    /// BTPE seeding mode (per-sample or stream)
    #[arg(long, default_value = "per-sample", value_parser = ["per-sample", "stream"])]
    btpe_seed_mode: String,

    /// Stream seed for BTPE when using stream mode (defaults to --seed)
    #[arg(long)]
    btpe_stream_seed: Option<u64>,
}

fn main() -> eyre::Result<()> {
    let args = Args::parse();
    if args.denom == 0 {
        eyre::bail!("denom must be > 0");
    }
    if args.num > args.denom {
        eyre::bail!("num must be <= denom");
    }

    let p = args.num as f64 / args.denom as f64;
    let n = args.n;
    let samples = args.samples as usize;

    let btpe_rng = BtpeRng::parse(&args.btpe_rng)?;
    let btpe_seed_mode = BtpeSeedMode::parse(&args.btpe_seed_mode)?;
    let btpe_stream_seed = args.btpe_stream_seed.unwrap_or(args.seed);
    let prf_rng = BtpeRng::parse(&args.prf_rng)?;
    let prf_stream_seed = args.prf_stream_seed.unwrap_or(args.seed);

    println!("Binomial sampler benchmark (BTPE+inverse hybrid)");
    println!("n = {n}, p = {}/{} ({p:.8})", args.num, args.denom);
    println!("samples = {}", args.samples);
    println!(
        "prf_rng = {:?}, btpe_rng = {:?}, btpe_seed_mode = {:?}",
        prf_rng, btpe_rng, btpe_seed_mode
    );

    let mut prfs = Vec::with_capacity(samples);
    match prf_rng {
        BtpeRng::SplitMix => {
            let mut rng = SplitMix64::new(prf_stream_seed);
            for _ in 0..samples {
                prfs.push(rng.next_u64());
            }
        }
        BtpeRng::Aes => {
            let mut rng = AesCtrRng::new_from_seed(prf_stream_seed);
            for _ in 0..samples {
                prfs.push(rng.next_u64());
            }
        }
    }

    let btpe = BtpecSampler::new(n, p);
    let (btpe_stats, btpe_time) =
        run_btpe(&btpe, &prfs, btpe_rng, btpe_seed_mode, btpe_stream_seed);
    let (cur_stats, cur_time) = run_current(n, args.num, args.denom, &prfs);

    let expected_mean = n as f64 * p;
    let expected_var = expected_mean * (1.0 - p);

    println!("\nExpected:");
    println!("  mean = {:.6}", expected_mean);
    println!("  var  = {:.6}", expected_var);

    println!("\nBTPE+inverse:");
    btpe_stats.print();
    println!("  time = {:.3} ms", btpe_time);

    println!("\nCurrent (inverse/beta CDF):");
    cur_stats.print();
    println!("  time = {:.3} ms", cur_time);

    Ok(())
}

fn run_btpe(
    btpe: &BtpecSampler,
    prfs: &[u64],
    btpe_rng: BtpeRng,
    btpe_seed_mode: BtpeSeedMode,
    stream_seed: u64,
) -> (Stats, f64) {
    let mut stats = Stats::new();
    let start = Instant::now();
    match (btpe_seed_mode, btpe_rng) {
        (BtpeSeedMode::PerSample, BtpeRng::SplitMix) => {
            for &prf in prfs {
                stats.add(btpe.sample_from_splitmix(prf));
            }
        }
        (BtpeSeedMode::PerSample, BtpeRng::Aes) => {
            for &prf in prfs {
                stats.add(btpe.sample_from_aes(prf));
            }
        }
        (BtpeSeedMode::Stream, BtpeRng::SplitMix) => {
            let mut rng = SplitMix64::new(stream_seed);
            for _ in prfs {
                stats.add(btpe.sample_with_rng(&mut rng));
            }
        }
        (BtpeSeedMode::Stream, BtpeRng::Aes) => {
            let mut rng = AesCtrRng::new_from_seed(stream_seed);
            for _ in prfs {
                stats.add(btpe.sample_with_rng(&mut rng));
            }
        }
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
    (stats, elapsed_ms)
}

fn run_current(n: u64, num: u64, denom: u64, prfs: &[u64]) -> (Stats, f64) {
    let mut stats = Stats::new();
    let start = Instant::now();
    for &prf in prfs {
        stats.add(binomial_sample(n, num, denom, prf));
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
    (stats, elapsed_ms)
}

#[derive(Default)]
struct Stats {
    min: u64,
    max: u64,
    sum: f64,
    sum_sq: f64,
    count: u64,
}

impl Stats {
    fn new() -> Self {
        Self {
            min: u64::MAX,
            max: 0,
            sum: 0.0,
            sum_sq: 0.0,
            count: 0,
        }
    }

    fn add(&mut self, v: u64) {
        self.min = self.min.min(v);
        self.max = self.max.max(v);
        let fv = v as f64;
        self.sum += fv;
        self.sum_sq += fv * fv;
        self.count += 1;
    }

    fn mean(&self) -> f64 {
        self.sum / self.count as f64
    }

    fn var(&self) -> f64 {
        let mean = self.mean();
        self.sum_sq / self.count as f64 - mean * mean
    }

    fn print(&self) {
        println!("  mean = {:.6}", self.mean());
        println!("  var  = {:.6}", self.var());
        println!("  min  = {}", self.min);
        println!("  max  = {}", self.max);
    }
}

#[derive(Clone, Copy, Debug)]
enum BtpeRng {
    SplitMix,
    Aes,
}

impl BtpeRng {
    fn parse(s: &str) -> eyre::Result<Self> {
        match s {
            "splitmix" => Ok(Self::SplitMix),
            "aes" => Ok(Self::Aes),
            _ => eyre::bail!("unknown btpe_rng: {s}"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum BtpeSeedMode {
    PerSample,
    Stream,
}

impl BtpeSeedMode {
    fn parse(s: &str) -> eyre::Result<Self> {
        match s {
            "per-sample" => Ok(Self::PerSample),
            "stream" => Ok(Self::Stream),
            _ => eyre::bail!("unknown btpe_seed_mode: {s}"),
        }
    }
}

trait Rng64 {
    fn next_u64(&mut self) -> u64;

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64 + 0.5) / (u64::MAX as f64 + 1.0)
    }
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

impl Rng64 for SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        SplitMix64::next_u64(self)
    }
}

struct AesCtrRng {
    cipher: ctr::Ctr128BE<Aes128>,
    buf: [u8; 64],
    idx: usize,
}

impl AesCtrRng {
    fn new_from_seed(seed: u64) -> Self {
        let mut key = [0u8; 16];
        key[..8].copy_from_slice(&seed.to_le_bytes());
        key[8..].copy_from_slice(&seed.to_le_bytes());
        let iv = [0u8; 16];
        let cipher = ctr::Ctr128BE::<Aes128>::new(&key.into(), &iv.into());
        Self {
            cipher,
            buf: [0u8; 64],
            idx: 64,
        }
    }

    fn refill(&mut self) {
        self.buf = [0u8; 64];
        self.cipher.apply_keystream(&mut self.buf);
        self.idx = 0;
    }
}

impl Rng64 for AesCtrRng {
    fn next_u64(&mut self) -> u64 {
        if self.idx + 8 > self.buf.len() {
            self.refill();
        }
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&self.buf[self.idx..self.idx + 8]);
        self.idx += 8;
        u64::from_le_bytes(bytes)
    }
}

struct BtpecSampler {
    n: i64,
    p: f64,
    q: f64,
    xnp: f64,
    use_complement: bool,
    m: i64,
    fm: f64,
    xnpq: f64,
    p1: f64,
    xm: f64,
    xl: f64,
    xr: f64,
    c: f64,
    xll: f64,
    xlr: f64,
    p2: f64,
    p3: f64,
    p4: f64,
}

impl BtpecSampler {
    fn new(n: u64, p_raw: f64) -> Self {
        let mut p = p_raw;
        let mut use_complement = false;
        if p > 0.5 {
            p = 1.0 - p;
            use_complement = true;
        }
        let q = 1.0 - p;
        let xnp = n as f64 * p;

        if xnp < 30.0 {
            return Self {
                n: n as i64,
                p,
                q,
                xnp,
                use_complement,
                m: 0,
                fm: 0.0,
                xnpq: 0.0,
                p1: 0.0,
                xm: 0.0,
                xl: 0.0,
                xr: 0.0,
                c: 0.0,
                xll: 0.0,
                xlr: 0.0,
                p2: 0.0,
                p3: 0.0,
                p4: 0.0,
            };
        }

        let ffm = xnp + p;
        let m = ffm as i64;
        let fm = m as f64;
        let xnpq = xnp * q;
        let p1 = (2.195 * xnpq.sqrt() - 4.6 * q).floor() + 0.5;
        let xm = fm + 0.5;
        let xl = xm - p1;
        let xr = xm + p1;
        let c = 0.134 + 20.5 / (15.3 + fm);
        let mut al = (ffm - xl) / (ffm - xl * p);
        let xll = al * (1.0 + 0.5 * al);
        al = (xr - ffm) / (xr * q);
        let xlr = al * (1.0 + 0.5 * al);
        let p2 = p1 * (1.0 + 2.0 * c);
        let p3 = p2 + c / xll;
        let p4 = p3 + c / xlr;

        Self {
            n: n as i64,
            p,
            q,
            xnp,
            use_complement,
            m,
            fm,
            xnpq,
            p1,
            xm,
            xl,
            xr,
            c,
            xll,
            xlr,
            p2,
            p3,
            p4,
        }
    }

    fn sample_from_splitmix(&self, seed: u64) -> u64 {
        let mut rng = SplitMix64::new(seed);
        self.sample_with_rng(&mut rng)
    }

    fn sample_from_aes(&self, seed: u64) -> u64 {
        let mut rng = AesCtrRng::new_from_seed(seed);
        self.sample_with_rng(&mut rng)
    }

    fn sample_with_rng(&self, rng: &mut dyn Rng64) -> u64 {
        let k = if self.xnp < 30.0 {
            self.sample_inverse_small(rng)
        } else {
            self.sample_btpe(rng)
        };

        let k = if self.use_complement {
            self.n - k
        } else {
            k
        };
        k as u64
    }

    fn sample_inverse_small(&self, rng: &mut dyn Rng64) -> i64 {
        let qn = self.q.powi(self.n as i32);
        let r = self.p / self.q;
        let g = r * (self.n as f64 + 1.0);

        loop {
            let mut ix: i64 = 0;
            let mut f = qn;
            let mut u = rng.next_f64();
            loop {
                if u < f {
                    return ix;
                }
                if ix > 110 {
                    break;
                }
                u -= f;
                ix += 1;
                f *= g / ix as f64 - r;
            }
        }
    }

    fn sample_btpe(&self, rng: &mut dyn Rng64) -> i64 {
        loop {
            let u = rng.next_f64() * self.p4;
            let mut v = rng.next_f64();

            if u <= self.p1 {
                let ix = (self.xm - self.p1 * v + u) as i64;
                return ix;
            }

            if u <= self.p2 {
                let x = self.xl + (u - self.p1) / self.c;
                v = v * self.c + 1.0 - (self.xm - x).abs() / self.p1;
                if v > 1.0 || v <= 0.0 {
                    continue;
                }
                let ix = x as i64;
                if self.accept(ix, v) {
                    return ix;
                }
                continue;
            }

            if u <= self.p3 {
                let ix = (self.xl + v.ln() / self.xll) as i64;
                if ix < 0 {
                    continue;
                }
                v = v * (u - self.p2) * self.xll;
                if self.accept(ix, v) {
                    return ix;
                }
                continue;
            }

            let ix = (self.xr - v.ln() / self.xlr) as i64;
            if ix > self.n {
                continue;
            }
            v = v * (u - self.p3) * self.xlr;
            if self.accept(ix, v) {
                return ix;
            }
        }
    }

    fn accept(&self, ix: i64, v: f64) -> bool {
        let k = (ix - self.m).abs() as f64;
        if k > 20.0 && k < self.xnpq / 2.0 - 1.0 {
            let amaxp =
                (k / self.xnpq) * ((k * (k / 3.0 + 0.625) + 1.0 / 6.0) / self.xnpq + 0.5);
            let ynorm = -k * k / (2.0 * self.xnpq);
            let alv = v.ln();
            if alv < ynorm - amaxp {
                return true;
            }
            if alv > ynorm + amaxp {
                return false;
            }

            let x1 = ix as f64 + 1.0;
            let f1 = self.fm + 1.0;
            let z = self.n as f64 + 1.0 - self.fm;
            let w = self.n as f64 - ix as f64 + 1.0;

            let t = self.xm * (f1 / x1).ln()
                + (self.n as f64 - self.m as f64 + 0.5) * (z / w).ln()
                + (ix as f64 - self.m as f64) * ((w * self.p) / (x1 * self.q)).ln()
                + stirling_correction(f1)
                + stirling_correction(z)
                + stirling_correction(x1)
                + stirling_correction(w);

            return alv <= t;
        }

        let mut f = 1.0;
        let r = self.p / self.q;
        let g = (self.n as f64 + 1.0) * r;
        if ix > self.m {
            for i in (self.m + 1)..=ix {
                f *= g / i as f64 - r;
            }
        } else if ix < self.m {
            for i in (ix + 1)..=self.m {
                f /= g / i as f64 - r;
            }
        }

        v <= f
    }
}

fn stirling_correction(x: f64) -> f64 {
    let x2 = x * x;
    (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / x2) / x2) / x2) / x2) / x / 166320.0
}
