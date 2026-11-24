//! The talker's prompt as rows of summed embeddings, transcribed from the
//! reference `Qwen3TTSForConditionalGeneration.generate`: an optional style
//! instruction turn, the assistant role, the codec control prefix (think /
//! language tag, optional speaker vector, pad, bos) overlaid on text pads,
//! then the text itself, either all up front (non-streaming) or one token per
//! generated frame (streaming, `trailing`), and for in-context voice cloning
//! the reference transcript + codes.

use super::{codec_ids as c, text_ids as t};

/// One embedding summed into a prompt row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// A text token through the text embedding + projection.
    Text(u32),
    /// A talker codec / control id through the codec embedding.
    Codec(u32),
    /// The speaker vector (x-vector of a reference clip, or a preset row).
    Speaker,
    /// Frame `i` of the in-context reference: the sum of its 16 code
    /// embeddings.
    RefFrame(usize),
}

/// What the talker reads before its first frame and alongside each frame.
/// Job data (it spells the text): wiped on drop.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TalkerPrompt {
    pub rows: Vec<Vec<Part>>,
    /// Text ids added to the inputs of generated frames `0..` (then
    /// `TTS_PAD` for every later frame).
    pub trailing: Vec<u32>,
}

impl Drop for TalkerPrompt {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        for row in &mut self.rows {
            row.iter_mut().for_each(|p| *p = Part::Text(0));
        }
        self.trailing.zeroize();
    }
}

/// The in-context voice reference: its transcript's ids and its codes.
pub struct IclRef<'a> {
    /// Ids of `<|im_start|>assistant\n{transcript}<|im_end|>\n`.
    pub text_ids: &'a [u32],
    /// Reference frames.
    pub frames: usize,
}

pub struct PromptSpec<'a> {
    /// Ids of `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`.
    pub text_ids: &'a [u32],
    /// Ids of `<|im_start|>user\n{instruct}<|im_end|>\n` (style / voice
    /// description), if any.
    pub instruct_ids: Option<&'a [u32]>,
    /// Language tag (`None` = auto).
    pub language: Option<u32>,
    pub speaker: bool,
    pub icl: Option<IclRef<'a>>,
    pub non_streaming: bool,
}

/// Role `<|im_start|>assistant\n` at the front, `<|im_end|>\n<|im_start|>
/// assistant\n` (5 ids) at the back of the text turn.
const ROLE: usize = 3;
const TAIL: usize = 5;

pub fn build(spec: &PromptSpec<'_>) -> Result<TalkerPrompt, String> {
    let ids = spec.text_ids;
    if ids.len() < ROLE + TAIL + 1 {
        return Err("empty text".into());
    }
    let text = &ids[ROLE..ids.len() - TAIL];
    let mut rows: Vec<Vec<Part>> = Vec::new();
    if let Some(ins) = spec.instruct_ids {
        rows.extend(ins.iter().map(|&i| vec![Part::Text(i)]));
    }
    rows.extend(ids[..ROLE].iter().map(|&i| vec![Part::Text(i)]));
    // Codec prefix: think tags (+ language), speaker, pad, bos.
    let mut codec: Vec<Part> = match spec.language {
        None => vec![c::NOTHINK, c::THINK_BOS, c::THINK_EOS],
        Some(l) => vec![c::THINK, c::THINK_BOS, l, c::THINK_EOS],
    }
    .into_iter()
    .map(Part::Codec)
    .collect();
    if spec.speaker {
        codec.push(Part::Speaker);
    }
    codec.extend([Part::Codec(c::PAD), Part::Codec(c::BOS)]);
    let n = codec.len();
    // Text pads over every codec entry but the last two; bos over the
    // second to last; the last (codec bos) waits for the first text row.
    for (j, &cp) in codec[..n - 1].iter().enumerate() {
        let tp = if j < n - 2 { t::TTS_PAD } else { t::TTS_BOS };
        rows.push(vec![Part::Text(tp), cp]);
    }
    let codec_bos = codec[n - 1];
    let mut trailing = vec![t::TTS_PAD];
    if let Some(icl) = &spec.icl {
        let rt = &icl.text_ids;
        if rt.len() < ROLE + 2 {
            return Err("empty reference transcript".into());
        }
        // Reference transcript + text + eos against codec bos + ref frames.
        let mut txt: Vec<u32> = rt[ROLE..rt.len() - 2].to_vec();
        txt.extend_from_slice(text);
        txt.push(t::TTS_EOS);
        let mut cod = vec![codec_bos];
        cod.extend((0..icl.frames).map(Part::RefFrame));
        if spec.non_streaming {
            rows.extend(
                txt.iter()
                    .map(|&i| vec![Part::Text(i), Part::Codec(c::PAD)]),
            );
            rows.extend(cod.iter().map(|&p| vec![p, Part::Text(t::TTS_PAD)]));
        } else if txt.len() > cod.len() {
            rows.extend(cod.iter().zip(&txt).map(|(&p, &i)| vec![Part::Text(i), p]));
            trailing = txt[cod.len()..].to_vec();
        } else {
            rows.extend(
                cod.iter()
                    .enumerate()
                    .map(|(j, &p)| vec![Part::Text(txt.get(j).copied().unwrap_or(t::TTS_PAD)), p]),
            );
        }
    } else if spec.non_streaming {
        rows.extend(
            text.iter()
                .map(|&i| vec![Part::Text(i), Part::Codec(c::PAD)]),
        );
        rows.push(vec![Part::Text(t::TTS_EOS), Part::Codec(c::PAD)]);
        rows.push(vec![Part::Text(t::TTS_PAD), codec_bos]);
    } else {
        rows.push(vec![Part::Text(text[0]), codec_bos]);
        trailing = text[1..].to_vec();
        trailing.push(t::TTS_EOS);
    }
    Ok(TalkerPrompt { rows, trailing })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(text: &[u32]) -> Vec<u32> {
        let mut v = vec![t::IM_START, t::ASSISTANT, 198];
        v.extend_from_slice(text);
        v.extend([t::IM_END, 198, t::IM_START, t::ASSISTANT, 198]);
        v
    }

    #[test]
    fn streaming_feeds_text_per_frame() {
        let ids = turn(&[10, 11, 12]);
        let p = build(&PromptSpec {
            text_ids: &ids,
            instruct_ids: None,
            language: Some(2050),
            speaker: true,
            icl: None,
            non_streaming: false,
        })
        .unwrap();
        // role 3 + (think, think_bos, lang, think_eos, speaker, pad) + first text.
        assert_eq!(p.rows.len(), 3 + 6 + 1);
        assert_eq!(
            p.rows[3],
            vec![Part::Text(t::TTS_PAD), Part::Codec(c::THINK)]
        );
        assert_eq!(p.rows[7], vec![Part::Text(t::TTS_PAD), Part::Speaker]);
        assert_eq!(p.rows[8], vec![Part::Text(t::TTS_BOS), Part::Codec(c::PAD)]);
        assert_eq!(p.rows[9], vec![Part::Text(10), Part::Codec(c::BOS)]);
        assert_eq!(p.trailing, vec![11, 12, t::TTS_EOS]);
    }

    #[test]
    fn non_streaming_puts_text_up_front() {
        let ids = turn(&[10, 11]);
        let ins = [1u32, 2];
        let p = build(&PromptSpec {
            text_ids: &ids,
            instruct_ids: Some(&ins),
            language: None,
            speaker: false,
            icl: None,
            non_streaming: true,
        })
        .unwrap();
        // instruct 2 + role 3 + (nothink, think_bos, think_eos, pad) + 2 text + eos + bos.
        assert_eq!(p.rows.len(), 2 + 3 + 4 + 2 + 1 + 1);
        assert_eq!(p.rows[8], vec![Part::Text(t::TTS_BOS), Part::Codec(c::PAD)]);
        assert_eq!(
            p.rows[11],
            vec![Part::Text(t::TTS_EOS), Part::Codec(c::PAD)]
        );
        assert_eq!(
            p.rows[12],
            vec![Part::Text(t::TTS_PAD), Part::Codec(c::BOS)]
        );
        assert_eq!(p.trailing, vec![t::TTS_PAD]);
    }

    #[test]
    fn icl_streaming_overlays_ref_frames() {
        let ids = turn(&[10, 11, 12, 13, 14]);
        let rt = [t::IM_START, t::ASSISTANT, 198, 20, 21, t::IM_END, 198];
        let p = build(&PromptSpec {
            text_ids: &ids,
            instruct_ids: None,
            language: Some(2050),
            speaker: true,
            icl: Some(IclRef {
                text_ids: &rt,
                frames: 3,
            }),
            non_streaming: false,
        })
        .unwrap();
        // text = [20, 21, 10..14, eos] (8) vs codec = [bos, f0, f1, f2] (4).
        let head = 3 + 6;
        assert_eq!(p.rows.len(), head + 4);
        assert_eq!(p.rows[head], vec![Part::Text(20), Part::Codec(c::BOS)]);
        assert_eq!(p.rows[head + 3], vec![Part::Text(11), Part::RefFrame(2)]);
        assert_eq!(p.trailing, vec![12, 13, 14, t::TTS_EOS]);
    }
}
