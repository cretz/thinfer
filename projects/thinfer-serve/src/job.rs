//! In-memory job metadata + the per-job event log. Artifacts live on disk
//! (under the configured artifact dir); everything here is RAM and is lost on
//! restart (accepted for a single-user host -- orphaned artifacts are swept by
//! TTL). Each job has a replayable event log plus a broadcast channel so an SSE
//! client gets caught up (replay) then tails live.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use thinfer_app::request::JobRequest;
use thinfer_app::wire::{JobEvent, JobResult, JobStateKind, JobStatus, ProgressStage};
use tokio::sync::{Notify, broadcast};

/// An event with its monotonic sequence number (the SSE `id:`, for
/// `Last-Event-ID` reconnect).
#[derive(Clone, Debug)]
pub struct SeqEvent {
    pub seq: u64,
    pub event: JobEvent,
}

struct Inner {
    kind: JobStateKind,
    last_progress: Option<ProgressStage>,
    result: Option<JobResult>,
    error: Option<String>,
    log: Vec<SeqEvent>,
    seq: u64,
}

/// Live state for one job. The `request` and `output_path` are immutable; the
/// rest is behind a mutex. Cloned `Arc`s are shared between the API handlers and
/// the worker that runs it.
pub struct JobHandle {
    pub id: String,
    pub request: JobRequest,
    /// Absolute path of the produced artifact (file) or its directory (PNG
    /// frames).
    pub output_path: PathBuf,
    /// Client RSA public key (base64 SPKI) for result encryption, if requested.
    /// `Some` => the worker encrypts the artifact in place and the result is
    /// served as opaque ciphertext; `None` => plaintext.
    pub public_key: Option<String>,
    inner: Mutex<Inner>,
    tx: broadcast::Sender<SeqEvent>,
    cancel: AtomicBool,
}

impl JobHandle {
    fn new(
        id: String,
        request: JobRequest,
        output_path: PathBuf,
        public_key: Option<String>,
    ) -> Self {
        let (tx, _) = broadcast::channel(1024);
        Self {
            id,
            request,
            output_path,
            public_key,
            inner: Mutex::new(Inner {
                kind: JobStateKind::Queued,
                last_progress: None,
                result: None,
                error: None,
                log: Vec::new(),
                seq: 0,
            }),
            tx,
            cancel: AtomicBool::new(false),
        }
    }

    /// Append an event: assign a seq, update derived state, persist to the log,
    /// and broadcast to live subscribers.
    pub fn push(&self, event: JobEvent) {
        let mut inner = self.inner.lock().unwrap();
        inner.seq += 1;
        let seq = inner.seq;
        match &event {
            JobEvent::Queued { .. } => inner.kind = JobStateKind::Queued,
            JobEvent::Started => inner.kind = JobStateKind::Running,
            JobEvent::Progress { stage } => inner.last_progress = Some(stage.clone()),
            JobEvent::Log { .. } => {}
            JobEvent::Done { result } => {
                inner.kind = JobStateKind::Done;
                inner.result = Some(result.clone());
            }
            JobEvent::Error { message } => {
                inner.kind = JobStateKind::Failed;
                inner.error = Some(message.clone());
            }
            JobEvent::Cancelled => inner.kind = JobStateKind::Cancelled,
        }
        let se = SeqEvent { seq, event };
        inner.log.push(se.clone());
        // Err just means no live subscribers; the log replay covers them.
        let _ = self.tx.send(se);
    }

    /// (replay log snapshot after `after_seq`, live receiver). Subscribe first
    /// so no event slips between the snapshot and the live tail.
    pub fn subscribe(&self, after_seq: u64) -> (Vec<SeqEvent>, broadcast::Receiver<SeqEvent>) {
        let inner = self.inner.lock().unwrap();
        let rx = self.tx.subscribe();
        let replay = inner
            .log
            .iter()
            .filter(|e| e.seq > after_seq)
            .cloned()
            .collect();
        (replay, rx)
    }

    pub fn status(&self, queue_position: Option<usize>) -> JobStatus {
        let inner = self.inner.lock().unwrap();
        JobStatus {
            id: self.id.clone(),
            state: inner.kind,
            queue_position: matches!(inner.kind, JobStateKind::Queued)
                .then_some(queue_position)
                .flatten(),
            progress: inner.last_progress.clone(),
            result: inner.result.clone(),
            error: inner.error.clone(),
        }
    }

    pub fn state_kind(&self) -> JobStateKind {
        self.inner.lock().unwrap().kind
    }

    pub fn request_cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }

    pub fn is_cancel_requested(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
}

/// All jobs + the pending queue. Sequential ids (single-user host; not secrets).
pub struct JobStore {
    jobs: Mutex<HashMap<String, std::sync::Arc<JobHandle>>>,
    queue: Mutex<VecDeque<std::sync::Arc<JobHandle>>>,
    next_id: AtomicU64,
    active: AtomicUsize,
    /// Wakes idle workers when a job is enqueued.
    pub notify: Notify,
}

impl Default for JobStore {
    fn default() -> Self {
        Self {
            jobs: Mutex::new(HashMap::new()),
            queue: Mutex::new(VecDeque::new()),
            next_id: AtomicU64::new(1),
            active: AtomicUsize::new(0),
            notify: Notify::new(),
        }
    }
}

impl JobStore {
    /// Allocate a job id, build the request via `make` (which can use the id to
    /// place the artifact path), then register + enqueue. `make` runs before any
    /// lock is held, so it may touch the filesystem (create the job dir). Returns
    /// (handle, queue position).
    pub fn submit(
        &self,
        make: impl FnOnce(&str) -> Result<(JobRequest, PathBuf, Option<String>), String>,
    ) -> Result<(std::sync::Arc<JobHandle>, usize), String> {
        let n = self.next_id.fetch_add(1, Ordering::SeqCst);
        let id = format!("job-{n}");
        let (request, output_path, public_key) = make(&id)?;
        let handle =
            std::sync::Arc::new(JobHandle::new(id.clone(), request, output_path, public_key));
        self.jobs.lock().unwrap().insert(id, handle.clone());
        let position = {
            let mut q = self.queue.lock().unwrap();
            q.push_back(handle.clone());
            q.len() - 1
        };
        handle.push(JobEvent::Queued { position });
        self.notify.notify_one();
        Ok((handle, position))
    }

    pub fn get(&self, id: &str) -> Option<std::sync::Arc<JobHandle>> {
        self.jobs.lock().unwrap().get(id).cloned()
    }

    /// Index of `id` in the pending queue, if still queued.
    pub fn position(&self, id: &str) -> Option<usize> {
        self.queue.lock().unwrap().iter().position(|h| h.id == id)
    }

    /// True when a large-input job would have to wait (worker busy or others
    /// queued) -- such jobs are rejected rather than queued.
    pub fn is_busy(&self) -> bool {
        self.active.load(Ordering::SeqCst) > 0 || !self.queue.lock().unwrap().is_empty()
    }

    /// Pop the next job for a worker, skipping any that were cancelled while
    /// queued (those are marked cancelled and dropped). Increments the active
    /// count for the returned job.
    pub fn take_next(&self) -> Option<std::sync::Arc<JobHandle>> {
        let mut q = self.queue.lock().unwrap();
        while let Some(handle) = q.pop_front() {
            if handle.is_cancel_requested() {
                handle.push(JobEvent::Cancelled);
                continue;
            }
            self.active.fetch_add(1, Ordering::SeqCst);
            return Some(handle);
        }
        None
    }

    /// Mark a job finished (decrement active count).
    pub fn finish(&self) {
        self.active.fetch_sub(1, Ordering::SeqCst);
    }
}
