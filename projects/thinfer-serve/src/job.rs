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
//! runs. A job submitted with `preempt` runs next, ahead of the queue, and
//! when no worker is free it pauses a running job that can (one it may
//! manage), which goes back in the queue.
//!
//! Pausing (see `thinfer_app::checkpoint`): a running job asked to pause
//! stops at a step boundary, hands its state to the worker, which seals it
//! under the job key into `<dir>/checkpoint.sealed`, and the job gives up its
//! worker: back in the queue, or held (out of it) until resumed. Its sealed
//! spec stays with it, and the resumed run asks for the key again, so the
//! state is ciphertext at rest and opened only to run.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use thinfer_app::jobkey::JobKey;
use thinfer_app::wire::{
    JobEvent, JobListEntry, JobPause, JobResult, JobStateKind, JobStatus, PauseMode, ProgressStage,
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
    /// Run next; pause a running job this way when no worker is free.
    preempt: Option<PauseMode>,
    /// A pause asked for and not yet acted on.
    pause: Mutex<Option<JobPause>>,
    /// While it runs a model that can pause.
    pausable: AtomicBool,
}

/// A paused job's state, sealed under its key, in its dir.
const CHECKPOINT_FILE: &str = "checkpoint.sealed";

impl JobHandle {
    fn new(
        id: String,
        dir: PathBuf,
        sealed: Vec<u8>,
        priority: i32,
        owner: Option<[u8; 32]>,
        preempt: Option<PauseMode>,
    ) -> Self {
        let (tx, _) = broadcast::channel(1024);
        Self {
            id,
            dir,
            priority: AtomicI32::new(priority),
            created: Instant::now(),
            owner,
            preempt,
            pause: Mutex::new(None),
            pausable: AtomicBool::new(false),
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
            JobEvent::Paused { held: true } => inner.kind = JobStateKind::Paused,
            JobEvent::Paused { held: false } => inner.kind = JobStateKind::Queued,
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
            pausable: self.can_pause(inner.kind),
        }
    }

    /// Running a model that can pause, and not already pausing.
    fn can_pause(&self, kind: JobStateKind) -> bool {
        kind == JobStateKind::Running
            && self.pausable.load(Ordering::SeqCst)
            && self.pause.lock().unwrap().is_none()
    }

    /// Whether the run may pause (set once its request is open, cleared when
    /// it ends).
    pub fn set_pausable(&self, pausable: bool) {
        self.pausable.store(pausable, Ordering::SeqCst);
    }

    /// How the owner asked the run to pause, if they did.
    pub fn pause_mode(&self) -> Option<PauseMode> {
        self.pause.lock().unwrap().as_ref().map(|p| p.mode)
    }

    /// The pause the worker acts on (taken once).
    pub fn take_pause(&self) -> Option<JobPause> {
        self.pause.lock().unwrap().take()
    }

    /// Where a paused run's sealed state lives.
    pub fn checkpoint_path(&self) -> PathBuf {
        self.dir.join(CHECKPOINT_FILE)
    }

    /// Give the sealed spec back (a paused job runs again).
    pub fn restore_sealed(&self, sealed: Vec<u8>) {
        *self.sealed.lock().unwrap() = Some(sealed);
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
            pausable: self.can_pause(inner.kind),
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
    /// Workers taking jobs (a preempting job pauses another only when all
    /// are busy).
    workers: usize,
    /// One priority level gained per this long waited.
    aging: Duration,
    /// Wakes idle workers when a job is enqueued.
    pub notify: Notify,
}

impl Default for JobStore {
    fn default() -> Self {
        Self::new(Duration::from_secs(300), 1)
    }
}

/// 32 hex chars of randomness: an unguessable job id (and dir name).
fn random_id() -> String {
    let mut b = [0u8; 16];
    rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut b);
    b.iter().map(|x| format!("{x:02x}")).collect()
}

impl JobStore {
    pub fn new(aging: Duration, workers: usize) -> Self {
        Self {
            jobs: Mutex::new(HashMap::new()),
            queue: Mutex::new(Vec::new()),
            active: AtomicUsize::new(0),
            workers: workers.max(1),
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
        preempt: Option<PauseMode>,
    ) -> Arc<JobHandle> {
        let id = random_id();
        Arc::new(JobHandle::new(
            id.clone(),
            artifact_dir.join(&id),
            sealed,
            priority,
            owner,
            preempt,
        ))
    }

    /// The pending queue in run order: preempting jobs first, then effective
    /// priority descending, then arrival (the stable sort keeps arrival
    /// order among ties).
    fn ordered(&self, q: &[Arc<JobHandle>]) -> Vec<Arc<JobHandle>> {
        let mut v = q.to_vec();
        v.sort_by_key(|h| {
            (
                h.preempt.is_none(),
                std::cmp::Reverse(h.effective_priority(self.aging)),
            )
        });
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
        if let Some(mode) = handle.preempt {
            self.preempt_for(handle, mode);
        }
        self.notify.notify_one();
        position
    }

    /// With every worker busy, pause the lowest-priority running job that can
    /// pause (and is not already pausing), so `job` gets its worker. Any job
    /// may: a pause loses no work, it only reorders, as priority does.
    fn preempt_for(&self, job: &JobHandle, mode: PauseMode) {
        if self.active.load(Ordering::SeqCst) < self.workers {
            return;
        }
        let victim = self
            .jobs
            .lock()
            .unwrap()
            .values()
            .filter(|h| h.id != job.id && h.can_pause(h.state_kind()))
            .min_by_key(|h| h.priority())
            .cloned();
        if let Some(victim) = victim {
            tracing::info!(id = %victim.id, by = %job.id, "pausing a running job for a preempting one");
            *victim.pause.lock().unwrap() = Some(JobPause {
                mode,
                hold: false,
                priority: None,
            });
        }
    }

    /// Ask a job to pause. Queued: held (`hold`) or just reprioritized.
    /// Taken but not yet running: given back before it runs. Running: it
    /// pauses at its next step boundary (or poll, for [`PauseMode::Now`]),
    /// when its model can.
    pub fn pause(&self, id: &str, pause: JobPause) -> Result<(), &'static str> {
        let job = self.get(id).ok_or("no such job")?;
        {
            let mut q = self.queue.lock().unwrap();
            if q.iter().any(|h| h.id == id) {
                if let Some(p) = pause.priority {
                    job.priority.store(p, Ordering::SeqCst);
                }
                if pause.hold {
                    q.retain(|h| h.id != id);
                    drop(q);
                    job.push(JobEvent::Paused { held: true });
                }
                return Ok(());
            }
        }
        match job.state_kind() {
            JobStateKind::Starting => {}
            JobStateKind::Running if job.pausable.load(Ordering::SeqCst) => {}
            JobStateKind::Running => return Err("this job's model cannot pause"),
            JobStateKind::Paused => return Err("the job is already paused"),
            _ => return Err("the job is not queued or running"),
        }
        *job.pause.lock().unwrap() = Some(pause);
        Ok(())
    }

    /// Queue a held job again.
    pub fn resume(&self, id: &str, priority: Option<i32>) -> Result<(), &'static str> {
        let job = self.get(id).ok_or("no such job")?;
        if job.state_kind() != JobStateKind::Paused {
            return Err("the job is not held");
        }
        if let Some(p) = priority {
            job.priority.store(p, Ordering::SeqCst);
        }
        self.enqueue(&job);
        Ok(())
    }

    /// A worker gave up a paused job: back in the queue, or held (or ended,
    /// when it was cancelled meanwhile).
    pub fn requeue(&self, job: &Arc<JobHandle>, pause: JobPause) {
        if job.is_cancel_requested() {
            let _ = std::fs::remove_file(job.checkpoint_path());
            return job.push(JobEvent::Cancelled);
        }
        if let Some(p) = pause.priority {
            job.priority.store(p, Ordering::SeqCst);
        }
        job.push(JobEvent::Paused { held: pause.hold });
        if !pause.hold {
            self.enqueue(job);
        }
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
        if queued || job.state_kind() == JobStateKind::Paused {
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

    /// Cancel a job: a running or queued one stops (see
    /// [`JobHandle::request_cancel`]); a held one ends now, its state deleted.
    pub fn cancel(&self, job: &JobHandle) {
        job.request_cancel();
        if job.state_kind() == JobStateKind::Paused {
            job.push(JobEvent::Cancelled);
            let _ = std::fs::remove_file(job.checkpoint_path());
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
        let h = store.create(
            std::path::Path::new("unused"),
            vec![1],
            priority,
            owner,
            None,
        );
        store.enqueue(&h);
        h
    }

    /// Take the next job and mark it running a model that can pause.
    fn run_pausable(store: &JobStore) -> Arc<JobHandle> {
        let h = store.take_next().unwrap();
        h.set_pausable(true);
        h.push(JobEvent::Started);
        h
    }

    fn hold(mode: PauseMode) -> JobPause {
        JobPause {
            mode,
            hold: true,
            priority: None,
        }
    }

    #[test]
    fn a_queued_job_is_held_and_resumed() {
        let store = JobStore::new(Duration::from_secs(3600), 1);
        let a = submit(&store, 0, None);
        let b = submit(&store, 0, None);
        store.pause(&a.id, hold(PauseMode::StepEnd)).unwrap();
        assert_eq!(a.state_kind(), JobStateKind::Paused);
        assert_eq!(store.position(&a.id), None);
        assert!(
            store.pause(&a.id, hold(PauseMode::Now)).is_err(),
            "already held"
        );
        // Held: the queue runs past it.
        assert_eq!(store.take_next().unwrap().id, b.id);
        assert!(store.take_next().is_none());
        assert!(
            store.resume(&b.id, None).is_err(),
            "only a held job resumes"
        );
        store.resume(&a.id, Some(4)).unwrap();
        assert_eq!(a.priority(), 4);
        assert_eq!(a.state_kind(), JobStateKind::Queued);
        assert_eq!(store.take_next().unwrap().id, a.id);
    }

    #[test]
    fn a_running_job_pauses_only_when_its_model_can() {
        let store = JobStore::new(Duration::from_secs(3600), 1);
        submit(&store, 0, None);
        let a = store.take_next().unwrap();
        // Taken, still waiting for its key: the worker gives it back.
        store.pause(&a.id, hold(PauseMode::Now)).unwrap();
        assert_eq!(a.pause_mode(), Some(PauseMode::Now));
        a.take_pause();
        a.push(JobEvent::Started);
        assert!(store.pause(&a.id, hold(PauseMode::StepEnd)).is_err());
        assert!(!a.status(None).pausable);
        a.set_pausable(true);
        assert!(a.status(None).pausable);
        store.pause(&a.id, hold(PauseMode::StepEnd)).unwrap();
        assert_eq!(a.pause_mode(), Some(PauseMode::StepEnd));
        assert!(!a.status(None).pausable, "already pausing");
        // The worker hands it back: held until resumed.
        store.requeue(&a, a.take_pause().unwrap());
        assert_eq!(a.state_kind(), JobStateKind::Paused);
        assert_eq!(store.position(&a.id), None);
        // Requeued without hold: back in line.
        let store = JobStore::new(Duration::from_secs(3600), 1);
        submit(&store, 0, None);
        let b = run_pausable(&store);
        store.requeue(
            &b,
            JobPause {
                mode: PauseMode::StepEnd,
                hold: false,
                priority: Some(-3),
            },
        );
        assert_eq!(b.state_kind(), JobStateKind::Queued);
        assert_eq!(b.priority(), -3);
        assert_eq!(store.position(&b.id), Some(0));
    }

    #[test]
    fn a_preempting_job_pauses_a_running_one_and_runs_next() {
        let store = JobStore::new(Duration::from_secs(3600), 1);
        submit(&store, 0, Some(JobKey::generate().proof()));
        let running = run_pausable(&store);
        let high = submit(&store, 9, None);
        assert_eq!(running.pause_mode(), None, "priority alone never pauses");
        // Another client's job (its own key) pauses it and runs next, ahead
        // of a higher-priority one.
        let pre = store.create(
            std::path::Path::new("unused"),
            vec![1],
            0,
            Some(JobKey::generate().proof()),
            Some(PauseMode::Now),
        );
        store.enqueue(&pre);
        assert_eq!(running.pause_mode(), Some(PauseMode::Now));
        // A second preempting job finds nothing more to pause.
        let pre2 = store.create(
            std::path::Path::new("unused"),
            vec![1],
            0,
            None,
            Some(PauseMode::StepEnd),
        );
        store.enqueue(&pre2);
        assert_eq!(running.pause_mode(), Some(PauseMode::Now));
        store.requeue(&running, running.take_pause().unwrap());
        store.finish();
        let order: Vec<String> = (0..4)
            .map(|_| store.take_next().unwrap().id.clone())
            .collect();
        assert_eq!(
            order,
            [
                pre.id.clone(),
                pre2.id.clone(),
                high.id.clone(),
                running.id.clone()
            ]
        );
    }

    #[test]
    fn a_free_worker_means_no_preemption() {
        let store = JobStore::new(Duration::from_secs(3600), 2);
        submit(&store, 0, None);
        let running = run_pausable(&store);
        let pre = store.create(
            std::path::Path::new("unused"),
            vec![1],
            0,
            None,
            Some(PauseMode::StepEnd),
        );
        store.enqueue(&pre);
        assert_eq!(running.pause_mode(), None);
    }

    #[test]
    fn a_held_job_cancels_and_deletes() {
        let tmp = std::env::temp_dir().join(format!("thinfer-held-{}", random_id()));
        let store = JobStore::new(Duration::from_secs(3600), 1);
        let a = store.create(&tmp, vec![1], 0, None, None);
        store.enqueue(&a);
        store.pause(&a.id, hold(PauseMode::StepEnd)).unwrap();
        std::fs::create_dir_all(&a.dir).unwrap();
        std::fs::write(a.checkpoint_path(), b"sealed").unwrap();
        store.cancel(&a);
        assert_eq!(a.state_kind(), JobStateKind::Cancelled);
        assert!(!a.checkpoint_path().exists());
        let b = store.create(&tmp, vec![1], 0, None, None);
        store.enqueue(&b);
        store.pause(&b.id, hold(PauseMode::StepEnd)).unwrap();
        store.delete(&b.id);
        assert!(store.get(&b.id).is_none());
        // A job cancelled while it paused ends instead of requeueing.
        let c = submit(&store, 0, None);
        let c2 = run_pausable(&store);
        assert_eq!(c.id, c2.id);
        c.request_cancel();
        store.requeue(&c, hold(PauseMode::StepEnd));
        assert_eq!(c.state_kind(), JobStateKind::Cancelled);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn higher_priority_first_ties_in_arrival_order() {
        let store = JobStore::new(Duration::from_secs(3600), 1);
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
        let store = JobStore::new(Duration::from_secs(3600), 1);
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
        let store = JobStore::new(Duration::from_secs(1), 1);
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
