//! Incremental Server-Sent Events framing for the job event stream.

use crate::Error;
use crate::wire::JobEvent;

/// One decoded frame: the event and its `id:` (the server's sequence number,
/// sent back as `Last-Event-ID` on a reconnect).
pub struct Frame {
    pub id: Option<u64>,
    pub event: JobEvent,
}

/// Buffers stream bytes and yields whole frames.
#[derive(Default)]
pub struct Decoder {
    buf: Vec<u8>,
}

impl Decoder {
    pub fn push(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    /// The next complete frame with data, if one is buffered. Frames without
    /// data (keep-alive comments) are skipped.
    pub fn next(&mut self) -> Result<Option<Frame>, Error> {
        while let Some((end, sep)) = frame_end(&self.buf) {
            let frame: Vec<u8> = self.buf.drain(..end + sep).take(end).collect();
            let text = String::from_utf8(frame)
                .map_err(|_| Error::Protocol("event stream is not UTF-8".into()))?;
            if let Some(f) = parse(&text)? {
                return Ok(Some(f));
            }
        }
        Ok(None)
    }
}

/// The first blank-line frame boundary: (offset, separator length).
fn frame_end(buf: &[u8]) -> Option<(usize, usize)> {
    let find = |pat: &[u8]| buf.windows(pat.len()).position(|w| w == pat);
    match (find(b"\n\n"), find(b"\r\n\r\n")) {
        (Some(a), Some(b)) if b < a => Some((b, 4)),
        (Some(a), _) => Some((a, 2)),
        (None, Some(b)) => Some((b, 4)),
        (None, None) => None,
    }
}

/// Decode one frame: its `id:` and its `data:` lines (joined) as a
/// [`JobEvent`]. `None` for a frame with no data.
fn parse(frame: &str) -> Result<Option<Frame>, Error> {
    let mut data = String::new();
    let mut id = None;
    for line in frame.lines() {
        let field = |name: &str| {
            line.strip_prefix(name)
                .map(|rest| rest.strip_prefix(' ').unwrap_or(rest))
        };
        if let Some(rest) = field("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(rest);
        } else if let Some(rest) = field("id:") {
            id = rest.trim().parse().ok();
        }
    }
    if data.is_empty() {
        return Ok(None);
    }
    let event = serde_json::from_str(&data)
        .map_err(|e| Error::Protocol(format!("undecodable event: {e}")))?;
    Ok(Some(Frame { id, event }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::progress::Stage;

    fn drain(d: &mut Decoder) -> Vec<Frame> {
        let mut out = Vec::new();
        while let Some(f) = d.next().unwrap() {
            out.push(f);
        }
        out
    }

    #[test]
    fn decodes_frames_with_ids_and_keeps_a_partial_tail() {
        let mut d = Decoder::default();
        d.push(b"id: 1\nevent: started\ndata: {\"type\":\"started\"}\n\n: keep-alive\n\n");
        d.push(b"id: 2\ndata: {\"type\":\"queued\",\"position\":1}\n\nid: 3\ndata: {\"type\":\"do");
        let frames = drain(&mut d);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].id, Some(1));
        assert!(matches!(frames[0].event, JobEvent::Started));
        assert!(matches!(frames[1].event, JobEvent::Queued { position: 1 }));
        d.push(b"ne\",\"result\":{\"width\":1,\"height\":1,\"frames\":1,\"fps\":null,\"seed\":null,\"resultUrl\":\"/r\"}}\r\n\r\n");
        let frames = drain(&mut d);
        assert_eq!(frames[0].id, Some(3));
        assert!(matches!(frames[0].event, JobEvent::Done { .. }));
    }

    #[test]
    fn progress_round_trips_to_a_stage() {
        let mut d = Decoder::default();
        d.push(b"data: {\"type\":\"progress\",\"stage\":{\"stage\":\"step\",\"i\":2,\"n\":3}}\n\n");
        match drain(&mut d).remove(0).event {
            JobEvent::Progress { stage } => {
                assert_eq!(Stage::from(stage), Stage::Step { i: 2, n: 3 })
            }
            other => panic!("expected progress, got {other:?}"),
        }
    }

    #[test]
    fn utf8_split_across_chunks_survives() {
        let mut d = Decoder::default();
        let frame = "data: {\"type\":\"log\",\"message\":\"caf\u{e9}\"}\n\n".as_bytes();
        let cut = frame.iter().position(|&b| b == 0xc3).unwrap() + 1;
        d.push(&frame[..cut]);
        assert!(d.next().unwrap().is_none());
        d.push(&frame[cut..]);
        match drain(&mut d).remove(0).event {
            JobEvent::Log { message } => assert_eq!(message, "caf\u{e9}"),
            other => panic!("expected log, got {other:?}"),
        }
    }
}
