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
//!
//! The queue is a priority queue: higher `priority` runs first, ties in
//! arrival order, and waiting raises a job's effective priority by one level
//! per `aging` so low-priority work never starves. Jobs are picked between
//! runs; a running job is never preempted.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use thinfer_app::jobkey::JobKey;
use thinfer_app::wire::{
    JobEvent, JobListEntry, JobResult, JobStateKind, JobStatus, ProgressStage,
};
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
    /// Queue priority (higher first); changeable while queued.
    priority: AtomicI32,
    created: Instant,
    /// SHA-256 of the job key, declared at submit: when set, managing the job
    /// (cancel / reprioritize / delete) requires it, and the delivered key
    /// must match it.
    owner: Option<[u8; 32]>,
}

impl JobHandle {
    fn new(
        id: String,
        dir: PathBuf,
        sealed: Vec<u8>,
        priority: i32,
        owner: Option<[u8; 32]>,
    ) -> Self {
        let (tx, _) = broadcast::channel(1024);
        Self {
            id,
            dir,
            priority: AtomicI32::new(priority),
            created: Instant::now(),
            owner,
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
            JobEvent::KeyRequired | JobEvent::Log { .. } | JobEvent::Text { .. } => {}
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
            priority: self.priority(),
            queue_position: matches!(inner.kind, JobStateKind::Queued)
                .then_some(queue_position)
                .flatten(),
            progress: inner.last_progress.clone(),
            result: inner.result.clone(),
            error: inner.error.clone(),
        }
    }

    /// The structural listing row (no result, no error text).
    fn entry(&self, queue_position: Option<usize>) -> JobListEntry {
        let inner = self.inner.lock().unwrap();
        JobListEntry {
            id: self.id.clone(),
            state: inner.kind,
            priority: self.priority(),
            queue_position,
            progress: inner.last_progress.clone(),
            age_secs: self.created.elapsed().as_secs(),
        }
    }

    pub fn priority(&self) -> i32 {
        self.priority.load(Ordering::SeqCst)
    }

    /// Priority plus one level per `aging` waited.
    fn effective_priority(&self, aging: Duration) -> i64 {
        let waited = self.created.elapsed().as_secs() / aging.as_secs().max(1);
        self.priority() as i64 + waited.min(i32::MAX as u64) as i64
    }

    /// Whether `proof` may manage this job: anyone for a job that declared no
    /// owner, else only the declared key hash.
    pub fn manageable_by(&self, proof: &[u8]) -> bool {
        self.owner.is_none_or(|h| ct_eq(&h, proof))
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

    /// Whether this job may still consume its claimed uploads: not finished
    /// and not cancelled.
    pub fn may_consume(&self) -> bool {
        !self.is_cancel_requested() && self.inner.lock().unwrap().finished.is_none()
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

    /// Accept the client's key, but only while the worker is waiting for it
    /// and (when the job declared one) only the key matching its owner proof.
    pub fn deliver_key(&self, key: JobKey) -> Result<(), &'static str> {
        if !self.manageable_by(&key.proof()) {
            return Err("key does not match the job's declared key proof");
        }
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
        slot.key_hash.is_some_and(|h| ct_eq(&h, proof))
    }

    fn finished_before(&self, cutoff: Instant) -> bool {
        self.inner
            .lock()
            .unwrap()
            .finished
            .is_some_and(|t| t < cutoff)
    }
}

/// Constant-time compare (length is not secret).
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// All jobs + the pending queue.
pub struct JobStore {
    jobs: Mutex<HashMap<String, Arc<JobHandle>>>,
    /// Pending jobs in arrival order; picked by effective priority.
    queue: Mutex<Vec<Arc<JobHandle>>>,
    active: AtomicUsize,
    /// One priority level gained per this long waited.
    aging: Duration,
    /// Wakes idle workers when a job is enqueued.
    pub notify: Notify,
}

impl Default for JobStore {
    fn default() -> Self {
        Self::new(Duration::from_secs(300))
    }
}

/// 32 hex chars of randomness: an unguessable job id (and dir name).
fn random_id() -> String {
    let mut b = [0u8; 16];
    rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut b);
    b.iter().map(|x| format!("{x:02x}")).collect()
}

impl JobStore {
    pub fn new(aging: Duration) -> Self {
        Self {
            jobs: Mutex::new(HashMap::new()),
            queue: Mutex::new(Vec::new()),
            active: AtomicUsize::new(0),
            aging,
            notify: Notify::new(),
        }
    }

    /// A new sealed job whose dir is `<artifact_dir>/<id>`, not yet visible to
    /// workers or lookups until [`JobStore::enqueue`].
    pub fn create(
        &self,
        artifact_dir: &std::path::Path,
        sealed: Vec<u8>,
        priority: i32,
        owner: Option<[u8; 32]>,
    ) -> Arc<JobHandle> {
        let id = random_id();
        Arc::new(JobHandle::new(
            id.clone(),
            artifact_dir.join(&id),
            sealed,
            priority,
            owner,
        ))
    }

    /// The pending queue in run order: effective priority descending, then
    /// arrival (the stable sort keeps arrival order among ties).
    fn ordered(&self, q: &[Arc<JobHandle>]) -> Vec<Arc<JobHandle>> {
        let mut v = q.to_vec();
        v.sort_by_key(|h| std::cmp::Reverse(h.effective_priority(self.aging)));
        v
    }

    /// Register + enqueue a created job. Returns its queue position.
    pub fn enqueue(&self, handle: &Arc<JobHandle>) -> usize {
        self.jobs
            .lock()
            .unwrap()
            .insert(handle.id.clone(), handle.clone());
        let position = {
            let mut q = self.queue.lock().unwrap();
            q.push(handle.clone());
            let order = self.ordered(&q);
            order.iter().position(|h| h.id == handle.id).unwrap_or(0)
        };
        handle.push(JobEvent::Queued { position });
        self.notify.notify_one();
        position
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

    /// Run-order index of `id` in the pending queue, if still queued.
    pub fn position(&self, id: &str) -> Option<usize> {
        let q = self.queue.lock().unwrap();
        self.ordered(&q).iter().position(|h| h.id == id)
    }

    /// Every known job: pending ones first in run order, then the rest
    /// (running / finished) newest first. Structural fields only.
    pub fn list(&self) -> Vec<JobListEntry> {
        let order = self.ordered(&self.queue.lock().unwrap());
        let mut out: Vec<JobListEntry> = order
            .iter()
            .enumerate()
            .map(|(i, h)| h.entry(Some(i)))
            .collect();
        let mut rest: Vec<Arc<JobHandle>> = self
            .jobs
            .lock()
            .unwrap()
            .values()
            .filter(|h| !order.iter().any(|q| q.id == h.id))
            .cloned()
            .collect();
        rest.sort_by_key(|h| std::cmp::Reverse(h.created));
        out.extend(rest.iter().map(|h| h.entry(None)));
        out
    }

    /// Change a queued job's priority. `false` when it is no longer queued.
    pub fn set_priority(&self, id: &str, priority: i32) -> bool {
        let q = self.queue.lock().unwrap();
        match q.iter().find(|h| h.id == id) {
            Some(h) => {
                h.priority.store(priority, Ordering::SeqCst);
                true
            }
            None => false,
        }
    }

    /// Drop a job: a queued one leaves the queue (cancelled) and is forgotten,
    /// a running one is flagged to cancel (the reaper forgets it once it
    /// ends), a finished one is forgotten now (its dir, an unfetched result
    /// included, deleted).
    pub fn delete(&self, id: &str) {
        let Some(job) = self.get(id) else {
            return;
        };
        let queued = {
            let mut q = self.queue.lock().unwrap();
            let before = q.len();
            q.retain(|h| h.id != id);
            q.len() < before
        };
        if queued {
            job.request_cancel();
            job.push(JobEvent::Cancelled);
            self.remove(id);
        } else if job.inner.lock().unwrap().finished.is_some() {
            self.remove(id);
        } else {
            job.request_cancel();
        }
    }

    /// Take the next job for a worker (highest effective priority, then
    /// arrival), skipping any cancelled while queued (marked cancelled and
    /// dropped). Increments the active count for the returned job.
    pub fn take_next(&self) -> Option<Arc<JobHandle>> {
        let mut q = self.queue.lock().unwrap();
        loop {
            let next = self.ordered(&q).into_iter().next()?;
            q.retain(|h| h.id != next.id);
            if next.is_cancel_requested() {
                next.push(JobEvent::Cancelled);
                continue;
            }
            next.inner.lock().unwrap().kind = JobStateKind::Starting;
            self.active.fetch_add(1, Ordering::SeqCst);
            return Some(next);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn submit(store: &JobStore, priority: i32, owner: Option<[u8; 32]>) -> Arc<JobHandle> {
        let h = store.create(std::path::Path::new("unused"), vec![1], priority, owner);
        store.enqueue(&h);
        h
    }

    #[test]
    fn higher_priority_first_ties_in_arrival_order() {
        let store = JobStore::new(Duration::from_secs(3600));
        let a = submit(&store, 0, None);
        let b = submit(&store, 5, None);
        let c = submit(&store, 0, None);
        assert_eq!(store.position(&b.id), Some(0));
        assert_eq!(store.position(&a.id), Some(1));
        let order: Vec<String> = (0..3)
            .map(|_| store.take_next().unwrap().id.clone())
            .collect();
        assert_eq!(order, [b.id.clone(), a.id.clone(), c.id.clone()]);
        assert!(store.take_next().is_none());
    }

    #[test]
    fn reprioritize_and_delete_queued() {
        let store = JobStore::new(Duration::from_secs(3600));
        let a = submit(&store, 0, None);
        let b = submit(&store, 0, None);
        assert!(store.set_priority(&b.id, 1));
        assert_eq!(store.position(&b.id), Some(0));
        store.delete(&b.id);
        assert!(store.get(&b.id).is_none());
        assert_eq!(store.take_next().unwrap().id, a.id);
        // No longer queued: priority is fixed.
        assert!(!store.set_priority(&a.id, 3));
        let list = store.list();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].queue_position, None);
        assert_eq!(list[0].state, JobStateKind::Starting);
    }

    #[test]
    fn aging_lifts_waiting_jobs() {
        let store = JobStore::new(Duration::from_secs(1));
        let old = submit(&store, -2, None);
        std::thread::sleep(Duration::from_millis(3100));
        let new = submit(&store, 0, None);
        // -2 + 3 levels waited > 0.
        assert_eq!(store.take_next().unwrap().id, old.id);
        assert_eq!(store.take_next().unwrap().id, new.id);
    }

    #[test]
    fn owner_proof_gates_management_and_key() {
        let store = JobStore::default();
        let key = JobKey::generate();
        let h = submit(&store, 0, Some(key.proof()));
        assert!(!h.manageable_by(&[]));
        assert!(!h.manageable_by(&[0; 32]));
        assert!(h.manageable_by(&key.proof()));
        // A different key is refused even while the worker waits.
        h.keys.lock().unwrap().awaiting = true;
        assert!(h.deliver_key(JobKey::generate()).is_err());
        assert!(h.deliver_key(key).is_ok());
        assert!(submit(&store, 0, None).manageable_by(&[]));
    }
}
