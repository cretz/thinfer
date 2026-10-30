//! In-memory job registry + the per-job event log. A job here is CIPHERTEXT
//! until a worker runs it: the client's sealed spec (see `thinfer_app::jobkey`)
//! is all the server holds while it queues. When a worker is ready it asks for
//! the key (`keyRequired` event), the client answers (`POST /jobs/{id}/key`),
//! and only then is anything decrypted, right before it feeds the model. The
//! artifact is sealed under the same key in RAM before it touches disk. After
//! the job, the key is dropped (zeroed); only its SHA-256 remains, so the
//! client can prove ownership when fetching the sealed result.
//!
//! Everything here is RAM and is lost on restart. Finished jobs and their
//! artifact dirs are reaped after `retention_secs` (see [`JobStore::reap`]).

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use thinfer_app::jobkey::JobKey;
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
    /// When the job reached a terminal state (reaping clock).
    finished: Option<Instant>,
}

/// Where a job is in the key handshake.
struct KeySlot {
    /// True only between `keyRequired` and the key's arrival.
    awaiting: bool,
    key: Option<JobKey>,
    /// SHA-256 of the delivered key: the fetch capability for the result.
    /// Kept after the key itself is dropped; reveals nothing about it.
    key_hash: Option<[u8; 32]>,
}

/// One job. Its spec is sealed; see the module docs.
pub struct JobHandle {
    pub id: String,
    /// This job's artifact dir (`<artifact_dir>/<id>`): uploads it consumes and
    /// its sealed result live here, and the dir is removed on fetch or reap.
    pub dir: PathBuf,
    /// The sealed `JobSpec`. Taken (once) by the worker that runs the job.
    sealed: Mutex<Option<Vec<u8>>>,
    keys: Mutex<KeySlot>,
    key_arrived: Notify,
    inner: Mutex<Inner>,
    tx: broadcast::Sender<SeqEvent>,
    cancel: AtomicBool,
}

impl JobHandle {
    fn new(id: String, dir: PathBuf, sealed: Vec<u8>) -> Self {
        let (tx, _) = broadcast::channel(1024);
        Self {
            id,
            dir,
            sealed: Mutex::new(Some(sealed)),
            keys: Mutex::new(KeySlot {
                awaiting: false,
                key: None,
                key_hash: None,
            }),
            key_arrived: Notify::new(),
            inner: Mutex::new(Inner {
                kind: JobStateKind::Queued,
                last_progress: None,
                result: None,
                error: None,
                log: Vec::new(),
                seq: 0,
                finished: None,
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
            JobEvent::KeyRequired | JobEvent::Log { .. } => {}
            JobEvent::Started => inner.kind = JobStateKind::Running,
            JobEvent::Progress { stage } => inner.last_progress = Some(stage.clone()),
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
        if event.is_terminal() {
            inner.finished = Some(Instant::now());
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
        // A worker blocked waiting for the key should give up now.
        self.key_arrived.notify_one();
    }

    pub fn is_cancel_requested(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }

    /// Take the sealed spec (once; the worker that runs the job).
    pub fn take_sealed(&self) -> Option<Vec<u8>> {
        self.sealed.lock().unwrap().take()
    }

    /// Ask the client for the key and wait up to `wait` for it. `None` on
    /// timeout or cancel. The key is accepted only during this window.
    pub async fn request_key(&self, wait: Duration) -> Option<JobKey> {
        self.keys.lock().unwrap().awaiting = true;
        self.push(JobEvent::KeyRequired);
        let deadline = tokio::time::Instant::now() + wait;
        loop {
            if let Some(key) = self.keys.lock().unwrap().key.take() {
                return Some(key);
            }
            if self.is_cancel_requested()
                || tokio::time::timeout_at(deadline, self.key_arrived.notified())
                    .await
                    .is_err()
            {
                self.keys.lock().unwrap().awaiting = false;
                return None;
            }
        }
    }

    /// Accept the client's key, but only while the worker is waiting for it.
    pub fn deliver_key(&self, key: JobKey) -> Result<(), &'static str> {
        let mut slot = self.keys.lock().unwrap();
        if !slot.awaiting {
            return Err("job is not waiting for its key");
        }
        slot.awaiting = false;
        slot.key_hash = Some(key.proof());
        slot.key = Some(key);
        drop(slot);
        self.key_arrived.notify_one();
        Ok(())
    }

    /// Whether `proof` (base64 SHA-256 of the job key) matches the delivered
    /// key. Constant-time compare.
    pub fn owns(&self, proof: &[u8]) -> bool {
        let slot = self.keys.lock().unwrap();
        slot.key_hash.is_some_and(|h| {
            h.len() == proof.len()
                && h.iter().zip(proof).fold(0u8, |acc, (a, b)| acc | (a ^ b)) == 0
        })
    }

    fn finished_before(&self, cutoff: Instant) -> bool {
        self.inner
            .lock()
            .unwrap()
            .finished
            .is_some_and(|t| t < cutoff)
    }
}

/// All jobs + the pending queue.
pub struct JobStore {
    jobs: Mutex<HashMap<String, Arc<JobHandle>>>,
    queue: Mutex<VecDeque<Arc<JobHandle>>>,
    active: AtomicUsize,
    /// Wakes idle workers when a job is enqueued.
    pub notify: Notify,
}

impl Default for JobStore {
    fn default() -> Self {
        Self {
            jobs: Mutex::new(HashMap::new()),
            queue: Mutex::new(VecDeque::new()),
            active: AtomicUsize::new(0),
            notify: Notify::new(),
        }
    }
}

/// 32 hex chars of randomness: an unguessable job id (and dir name).
fn random_id() -> String {
    let mut b = [0u8; 16];
    rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut b);
    b.iter().map(|x| format!("{x:02x}")).collect()
}

impl JobStore {
    /// Register + enqueue a sealed job whose dir is `<artifact_dir>/<id>`.
    /// Returns (handle, queue position).
    pub fn submit(
        &self,
        artifact_dir: &std::path::Path,
        sealed: Vec<u8>,
    ) -> (Arc<JobHandle>, usize) {
        let id = random_id();
        let handle = Arc::new(JobHandle::new(id.clone(), artifact_dir.join(&id), sealed));
        self.jobs.lock().unwrap().insert(id, handle.clone());
        let position = {
            let mut q = self.queue.lock().unwrap();
            q.push_back(handle.clone());
            q.len() - 1
        };
        handle.push(JobEvent::Queued { position });
        self.notify.notify_one();
        (handle, position)
    }

    pub fn get(&self, id: &str) -> Option<Arc<JobHandle>> {
        self.jobs.lock().unwrap().get(id).cloned()
    }

    /// Forget a job and delete its dir (after its result is fetched).
    pub fn remove(&self, id: &str) {
        if let Some(job) = self.jobs.lock().unwrap().remove(id) {
            let _ = std::fs::remove_dir_all(&job.dir);
        }
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
    pub fn take_next(&self) -> Option<Arc<JobHandle>> {
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

    /// Drop every job that finished more than `retention` ago, deleting its
    /// dir (an unfetched sealed result included). Returns the count reaped.
    pub fn reap(&self, retention: Duration) -> usize {
        let Some(cutoff) = Instant::now().checked_sub(retention) else {
            return 0;
        };
        let expired: Vec<Arc<JobHandle>> = {
            let mut jobs = self.jobs.lock().unwrap();
            let ids: Vec<String> = jobs
                .iter()
                .filter(|(_, j)| j.finished_before(cutoff))
                .map(|(id, _)| id.clone())
                .collect();
            ids.iter().filter_map(|id| jobs.remove(id)).collect()
        };
        for job in &expired {
            let _ = std::fs::remove_dir_all(&job.dir);
        }
        expired.len()
    }
}

/// Spawn the finished-job reaper (at most every 60s).
pub fn spawn_reaper(store: Arc<JobStore>, retention: Duration) {
    let period = retention
        .min(Duration::from_secs(60))
        .max(Duration::from_secs(1));
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(period);
        loop {
            tick.tick().await;
            let n = store.reap(retention);
            if n > 0 {
                tracing::info!(reaped = n, "job reaper removed finished jobs");
            }
        }
    });
}
