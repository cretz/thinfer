//! The remote execution path: an HTTP client for a `thinfer-serve` box. It is
//! the mirror of [`crate::executor::LocalExecutor`] -- same job, same progress
//! vocabulary -- but the work runs on the server. A front end builds a
//! [`JobSpec`] (the client wire shape), the executor POSTs it, tails the SSE
//! event stream into the caller's [`ProgressSink`] (so the CLI renders the exact
//! same stderr lines as a local run), and downloads the finished artifact.
//!
//! Behind the `remote` feature so non-client consumers (thinfer-serve itself)
//! never pull reqwest.

use std::path::Path;

use futures_util::StreamExt;
use tokio::io::AsyncWriteExt;

use crate::progress::ProgressSink;
use crate::request::JobSummary;
use crate::wire::{CreateResponse, JobEvent, JobResult, JobSpec};

/// HTTP client to a `thinfer-serve` deployment. Construct once with the base URL
/// (and an optional bearer token); reuse across jobs.
pub struct RemoteExecutor {
    /// Base URL with no trailing slash (e.g. `http://box:8080`).
    base: String,
    client: reqwest::Client,
    token: Option<String>,
}

impl RemoteExecutor {
    pub fn new(base_url: &str, token: Option<String>) -> Result<Self, String> {
        let base = base_url.trim_end_matches('/').to_string();
        if !(base.starts_with("http://") || base.starts_with("https://")) {
            return Err(format!(
                "--remote must be an http(s) URL (got {base_url:?})"
            ));
        }
        Ok(Self {
            base,
            client: reqwest::Client::new(),
            token,
        })
    }

    fn authed(&self, rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.token {
            Some(t) => rb.bearer_auth(t),
            None => rb,
        }
    }

    /// Submit `spec`, stream progress into `sink`, and download the result to
    /// `output` on success. Returns the summary the server reported.
    pub async fn run(
        &self,
        spec: &JobSpec,
        output: &Path,
        sink: &dyn ProgressSink,
    ) -> Result<JobSummary, String> {
        let created: CreateResponse = self.post_json(&format!("{}/jobs", self.base), spec).await?;
        sink.note(&format!("Submitted job {}", created.id));

        let result = self.stream_events(&created.id, sink).await?;
        self.download(&result.result_url, output).await?;

        Ok(JobSummary {
            output: output.to_path_buf(),
            width: result.width,
            height: result.height,
            frames: result.frames,
            fps: result.fps,
            seed: result.seed,
        })
    }

    /// POST a JSON body and decode the JSON response, surfacing the server's
    /// error message on a non-success status.
    async fn post_json<B: serde::Serialize, R: serde::de::DeserializeOwned>(
        &self,
        url: &str,
        body: &B,
    ) -> Result<R, String> {
        let resp = self
            .authed(self.client.post(url).json(body))
            .send()
            .await
            .map_err(|e| format!("POST {url}: {e}"))?;
        let resp = error_for_status(resp).await?;
        resp.json::<R>()
            .await
            .map_err(|e| format!("decode {url} response: {e}"))
    }

    /// Tail `GET /jobs/{id}/events`, mapping each SSE payload onto `sink` until a
    /// terminal event. Returns the `done` result, or an error for a failed /
    /// cancelled job.
    async fn stream_events(&self, id: &str, sink: &dyn ProgressSink) -> Result<JobResult, String> {
        let url = format!("{}/jobs/{id}/events", self.base);
        let resp = self
            .authed(self.client.get(&url))
            .send()
            .await
            .map_err(|e| format!("GET {url}: {e}"))?;
        let resp = error_for_status(resp).await?;

        let mut stream = resp.bytes_stream();
        let mut buf = String::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("event stream: {e}"))?;
            buf.push_str(&String::from_utf8_lossy(&chunk));
            // SSE frames are separated by a blank line. Drain whole frames,
            // leaving any partial trailing frame in `buf`.
            while let Some(end) = find_frame_end(&buf) {
                let frame = buf[..end].to_string();
                buf.drain(..end + frame_sep_len(&buf, end));
                if let Some(event) = parse_frame(&frame)? {
                    match apply(event, sink) {
                        Flow::Continue => {}
                        Flow::Done(result) => return Ok(result),
                        Flow::Err(msg) => return Err(msg),
                    }
                }
            }
        }
        Err("event stream ended before the job finished".into())
    }

    /// Download the artifact at `result_url` (server-relative) to `output`.
    async fn download(&self, result_url: &str, output: &Path) -> Result<(), String> {
        let url = format!("{}{result_url}", self.base);
        let resp = self
            .authed(self.client.get(&url))
            .send()
            .await
            .map_err(|e| format!("GET {url}: {e}"))?;
        let resp = error_for_status(resp).await?;

        if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| format!("create {}: {e}", parent.display()))?;
        }
        let mut file = tokio::fs::File::create(output)
            .await
            .map_err(|e| format!("create {}: {e}", output.display()))?;
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("download {url}: {e}"))?;
            file.write_all(&chunk)
                .await
                .map_err(|e| format!("write {}: {e}", output.display()))?;
        }
        file.flush()
            .await
            .map_err(|e| format!("flush {}: {e}", output.display()))?;
        Ok(())
    }
}

enum Flow {
    Continue,
    Done(JobResult),
    Err(String),
}

/// Route one decoded event onto the sink, returning whether to keep streaming.
fn apply(event: JobEvent, sink: &dyn ProgressSink) -> Flow {
    match event {
        JobEvent::Queued { position } => {
            sink.note(&format!("Queued at position {position}"));
            Flow::Continue
        }
        JobEvent::Started => {
            sink.note("Started");
            Flow::Continue
        }
        JobEvent::Progress { stage } => {
            sink.stage(stage.into());
            Flow::Continue
        }
        JobEvent::Log { message } => {
            sink.note(&message);
            Flow::Continue
        }
        JobEvent::Done { result } => Flow::Done(result),
        JobEvent::Error { message } => Flow::Err(message),
        JobEvent::Cancelled => Flow::Err("job was cancelled".into()),
    }
}

/// Index of the first blank-line frame boundary in `buf` (`\n\n` or `\r\n\r\n`),
/// or `None` if no complete frame is buffered yet.
fn find_frame_end(buf: &str) -> Option<usize> {
    let lf = buf.find("\n\n");
    let crlf = buf.find("\r\n\r\n");
    match (lf, crlf) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// Length of the blank-line separator at `end` (so the drain skips it).
fn frame_sep_len(buf: &str, end: usize) -> usize {
    if buf[end..].starts_with("\r\n\r\n") {
        4
    } else {
        2
    }
}

/// Decode one SSE frame: concatenate its `data:` lines and parse the JSON as a
/// [`JobEvent`]. Frames with no data (keep-alive comments, lone `event:` lines)
/// yield `None`.
fn parse_frame(frame: &str) -> Result<Option<JobEvent>, String> {
    let mut data = String::new();
    for line in frame.lines() {
        if let Some(rest) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(rest.strip_prefix(' ').unwrap_or(rest));
        }
    }
    if data.is_empty() {
        return Ok(None);
    }
    serde_json::from_str(&data)
        .map(Some)
        .map_err(|e| format!("decode SSE event {data:?}: {e}"))
}

/// Turn a non-2xx response into an `Err` carrying the server's `{error}` message
/// (falling back to the raw body / status).
async fn error_for_status(resp: reqwest::Response) -> Result<reqwest::Response, String> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    let body = resp.text().await.unwrap_or_default();
    let message = serde_json::from_str::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(str::to_string))
        .unwrap_or_else(|| {
            if body.is_empty() {
                status.to_string()
            } else {
                body
            }
        });
    Err(format!("server returned {status}: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::progress::Stage;

    /// Drain every complete frame from `buf` (mirrors the streaming loop) and
    /// return the decoded events, leaving any partial trailing frame behind.
    fn drain(buf: &mut String) -> Vec<JobEvent> {
        let mut out = Vec::new();
        while let Some(end) = find_frame_end(buf) {
            let frame = buf[..end].to_string();
            buf.drain(..end + frame_sep_len(buf, end));
            if let Some(ev) = parse_frame(&frame).unwrap() {
                out.push(ev);
            }
        }
        out
    }

    #[test]
    fn parses_data_line_stripping_one_leading_space() {
        let ev = parse_frame("event: started\ndata: {\"type\":\"started\"}")
            .unwrap()
            .unwrap();
        assert!(matches!(ev, JobEvent::Started));
    }

    #[test]
    fn keepalive_comment_yields_no_event() {
        assert!(parse_frame(": keep-alive ping").unwrap().is_none());
        assert!(parse_frame("event: progress").unwrap().is_none());
    }

    #[test]
    fn progress_event_round_trips_to_a_stage() {
        let ev = parse_frame(
            "data: {\"type\":\"progress\",\"stage\":{\"stage\":\"step\",\"i\":2,\"n\":3}}",
        )
        .unwrap()
        .unwrap();
        match ev {
            JobEvent::Progress { stage } => {
                assert_eq!(Stage::from(stage), Stage::Step { i: 2, n: 3 });
            }
            other => panic!("expected progress, got {other:?}"),
        }
    }

    #[test]
    fn drains_multiple_frames_and_keeps_a_partial_tail() {
        let mut buf = String::from(
            "data: {\"type\":\"started\"}\n\n\
             data: {\"type\":\"queued\",\"position\":1}\n\n\
             data: {\"type\":\"do",
        );
        let events = drain(&mut buf);
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], JobEvent::Started));
        assert!(matches!(events[1], JobEvent::Queued { position: 1 }));
        // The incomplete third frame stays buffered for the next chunk.
        assert_eq!(buf, "data: {\"type\":\"do");
    }

    #[test]
    fn handles_crlf_frame_separators() {
        let mut buf = String::from("data: {\"type\":\"started\"}\r\n\r\n");
        let events = drain(&mut buf);
        assert_eq!(events.len(), 1);
        assert!(buf.is_empty());
    }

    #[test]
    fn rejects_non_http_base_url() {
        assert!(RemoteExecutor::new("box:8080", None).is_err());
        assert!(RemoteExecutor::new("http://box:8080/", None).is_ok());
    }
}
