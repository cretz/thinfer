//! Token sampling over a logits row: temperature, top-k, top-p, min-p,
//! presence / repetition penalties, seeded and deterministic.

use zeroize::{Zeroize, Zeroizing};

/// Sampling knobs. `temperature == 0` is greedy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    /// Keep the `top_k` most likely tokens (0 = off).
    pub top_k: u32,
    /// Nucleus mass (1.0 = off).
    pub top_p: f32,
    /// Drop tokens below `min_p * p_max` (0 = off).
    pub min_p: f32,
    /// Subtracted once from the logit of every token already generated.
    pub presence_penalty: f32,
    /// Divides positive / multiplies negative logits of seen tokens (1 = off).
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl SamplingParams {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            seed: 0,
        }
    }
}

/// Holds the generated history (job data): wiped on drop. Both buffers are
/// sized to the vocabulary on first use, so they never reallocate (a grown
/// `Vec` would leave an unwiped copy behind).
pub struct Sampler {
    p: SamplingParams,
    state: u64,
    /// Distinct generated tokens, in first-seen order.
    seen: Vec<u32>,
    /// `seen` membership, one flag per vocabulary entry.
    is_seen: Vec<bool>,
}

impl Drop for Sampler {
    fn drop(&mut self) {
        self.seen.zeroize();
        self.is_seen.zeroize();
    }
}

impl Sampler {
    pub fn new(p: SamplingParams) -> Self {
        Self {
            p,
            state: p.seed ^ 0x9e37_79b9_7f4a_7c15,
            seen: Vec::new(),
            is_seen: Vec::new(),
        }
    }

    /// Record a generated token (penalties apply to it from now on).
    pub fn accept(&mut self, token: u32) {
        let t = token as usize;
        if t >= self.is_seen.len() {
            // Sized before the first sample; a token past it means a caller
            // accepted without sampling: grow (only then can it reallocate).
            self.reserve(t + 1);
        }
        if !self.is_seen[t] {
            self.is_seen[t] = true;
            self.seen.push(token);
        }
    }

    fn reserve(&mut self, vocab: usize) {
        if vocab > self.is_seen.len() {
            let mut seen = Vec::with_capacity(vocab);
            seen.extend_from_slice(&self.seen);
            self.seen.zeroize();
            self.seen = seen;
            let mut is_seen = vec![false; vocab];
            is_seen[..self.is_seen.len()].copy_from_slice(&self.is_seen);
            self.is_seen.zeroize();
            self.is_seen = is_seen;
        }
    }

    fn uniform(&mut self) -> f32 {
        // splitmix64
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Pick the next token from `logits` (consumed as scratch).
    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        let p = self.p;
        self.reserve(logits.len());
        for &t in &self.seen {
            let l = &mut logits[t as usize];
            if p.repetition_penalty != 1.0 {
                *l = if *l > 0.0 {
                    *l / p.repetition_penalty
                } else {
                    *l * p.repetition_penalty
                };
            }
            *l -= p.presence_penalty;
        }
        if p.temperature <= 0.0 {
            return argmax(logits);
        }
        // Candidates as token ids ranked by logit; the ranking is job data too.
        let mut cand: Zeroizing<Vec<u32>> = Zeroizing::new((0..logits.len() as u32).collect());
        let by_logit = |a: &u32, b: &u32| logits[*b as usize].total_cmp(&logits[*a as usize]);
        let k = if p.top_k == 0 {
            cand.len()
        } else {
            (p.top_k as usize).min(cand.len())
        };
        if k < cand.len() {
            cand.select_nth_unstable_by(k - 1, by_logit);
            cand.truncate(k);
        }
        cand.sort_unstable_by(by_logit);
        let max = logits[cand[0] as usize] / p.temperature;
        let mut probs: Zeroizing<Vec<f32>> = Zeroizing::new(
            cand.iter()
                .map(|&i| (logits[i as usize] / p.temperature - max).exp())
                .collect(),
        );
        let sum: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|x| *x /= sum);
        let mut keep = probs.len();
        if p.min_p > 0.0 {
            let floor = probs[0] * p.min_p;
            keep = probs.iter().take_while(|&&x| x >= floor).count().max(1);
        }
        if p.top_p < 1.0 {
            let mut acc = 0.0;
            for (i, &x) in probs.iter().enumerate().take(keep) {
                acc += x;
                if acc >= p.top_p {
                    keep = i + 1;
                    break;
                }
            }
        }
        let total: f32 = probs[..keep].iter().sum();
        let mut r = self.uniform() * total;
        for i in 0..keep {
            r -= probs[i];
            if r <= 0.0 {
                return cand[i];
            }
        }
        cand[keep - 1]
    }
}

pub fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_max() {
        let mut s = Sampler::new(SamplingParams::greedy());
        assert_eq!(s.sample(&mut [0.1, 3.0, -1.0]), 1);
    }

    #[test]
    fn top_k_one_is_greedy() {
        let mut s = Sampler::new(SamplingParams {
            temperature: 1.0,
            top_k: 1,
            ..SamplingParams::greedy()
        });
        for _ in 0..10 {
            assert_eq!(s.sample(&mut [0.1, 3.0, 2.9]), 1);
        }
    }

    #[test]
    fn seeded_is_deterministic() {
        let p = SamplingParams {
            temperature: 1.0,
            seed: 7,
            ..SamplingParams::greedy()
        };
        let draw = |n| {
            let mut s = Sampler::new(p);
            (0..n)
                .map(|_| s.sample(&mut [1.0, 1.1, 0.9, 1.05]))
                .collect::<Vec<_>>()
        };
        assert_eq!(draw(20), draw(20));
    }

    #[test]
    fn repetition_penalty_discourages_seen() {
        let mut s = Sampler::new(SamplingParams {
            repetition_penalty: 10.0,
            ..SamplingParams::greedy()
        });
        s.accept(1);
        assert_eq!(s.sample(&mut [2.0, 3.0]), 0);
    }
}
