//! Opt-in text sessions: the one piece of state that outlives a request.
//!
//! A session is a tree of nodes. Each node is the model state one text
//! request added (its KV rows for positions `start..end`, plus any fixed
//! recurrent state at `end`), sealed under the client's session key; the
//! server never holds that key between requests (it rides inside each job's
//! sealed spec). Continuing from a node replays its root-to-node chain into a
//! fresh model state instead of re-running the conversation; any node can be
//! continued from, so the tree branches.
//!
//! Everything lives in RAM: nothing is written to disk. A session is
//! forgotten after `ttl` unused; past `budget` sealed bytes the least recently
//! used leaf nodes (across sessions) are evicted, and a request naming an
//! evicted node fails ("node gone") so the client can fall back to sending
//! the full conversation. `export` / `import` move a whole session out and
//! back as one blob of ciphertext plus its structure.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use thinfer_app::model::TextModelId;
use thinfer_app::wire::{SessionInfo, SessionNodeInfo};

/// Why a session operation failed (the API maps it to a status).
#[derive(Debug)]
pub enum SessionError {
    NotFound(&'static str),
    Forbidden,
    Conflict(String),
    Invalid(String),
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(what) => write!(f, "unknown {what}"),
            Self::Forbidden => f.write_str("missing or wrong session key proof"),
            Self::Conflict(m) | Self::Invalid(m) => f.write_str(m),
        }
    }
}

struct Node {
    parent: Option<String>,
    start: u32,
    end: u32,
    sealed: Arc<Vec<u8>>,
    created: Instant,
    touched: Instant,
}

struct Session {
    model: TextModelId,
    /// SHA-256 of the session key.
    owner: [u8; 32],
    ttl: Duration,
    touched: Instant,
    /// Weights identity of the nodes (set by the first one).
    weights: Option<String>,
    nodes: HashMap<String, Node>,
}

impl Session {
    fn bytes(&self) -> u64 {
        self.nodes.values().map(|n| n.sealed.len() as u64).sum()
    }

    fn info(&self, id: &str) -> SessionInfo {
        let mut nodes: Vec<SessionNodeInfo> = self
            .nodes
            .iter()
            .map(|(nid, n)| SessionNodeInfo {
                id: nid.clone(),
                parent: n.parent.clone(),
                start: n.start,
                end: n.end,
                bytes: n.sealed.len() as u64,
                age_secs: n.created.elapsed().as_secs(),
            })
            .collect();
        nodes.sort_by_key(|n| (n.start, std::cmp::Reverse(n.age_secs)));
        SessionInfo {
            id: id.to_string(),
            model: self.model,
            ttl_secs: self.ttl.as_secs(),
            expires_in_secs: self.ttl.saturating_sub(self.touched.elapsed()).as_secs(),
            bytes: self.bytes(),
            nodes,
        }
    }

    /// Root-to-`node` ids (root first), checking each link is contiguous.
    fn path(&self, node: &str) -> Result<Vec<String>, SessionError> {
        let mut path = vec![node.to_string()];
        let mut cur = self
            .nodes
            .get(node)
            .ok_or(SessionError::NotFound("node (gone?)"))?;
        while let Some(p) = &cur.parent {
            let parent = self
                .nodes
                .get(p)
                .ok_or(SessionError::NotFound("node (an ancestor is gone)"))?;
            if parent.end != cur.start {
                return Err(SessionError::Invalid(
                    "session chain is not contiguous".into(),
                ));
            }
            path.push(p.clone());
            cur = parent;
        }
        if cur.start != 0 {
            return Err(SessionError::Invalid(
                "session root does not start at 0".into(),
            ));
        }
        path.reverse();
        Ok(path)
    }

    /// Remove `node` and every descendant; returns the bytes freed.
    fn prune(&mut self, node: &str) -> u64 {
        let mut doomed = vec![node.to_string()];
        let mut i = 0;
        while i < doomed.len() {
            let kids: Vec<String> = self
                .nodes
                .iter()
                .filter(|(_, n)| n.parent.as_deref() == Some(&doomed[i]))
                .map(|(k, _)| k.clone())
                .collect();
            doomed.extend(kids);
            i += 1;
        }
        doomed
            .iter()
            .filter_map(|k| self.nodes.remove(k))
            .map(|n| n.sealed.len() as u64)
            .sum()
    }
}

/// A job's view of a session: its model, the weights its nodes were built
/// with, and the sealed root-to-`from` chain.
pub struct Chain {
    pub model: TextModelId,
    pub weights: Option<String>,
    /// `(start, end, sealed state)` from the root.
    pub links: Vec<(u32, u32, Arc<Vec<u8>>)>,
}

struct Inner {
    sessions: HashMap<String, Session>,
    used: u64,
}

pub struct SessionStore {
    inner: Mutex<Inner>,
    /// Max sealed bytes held across all sessions.
    budget: u64,
    default_ttl: Duration,
}

fn random_id() -> String {
    let mut b = [0u8; 16];
    rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut b);
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Export blob: magic, u32 LE header length, header JSON, then each node's
/// sealed state in header order. Version 02: sliding-window KV rows are
/// trimmed to the window (01 blobs hold the full rows and cannot resume).
const MAGIC: &[u8; 8] = b"TFSESS02";

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExportHeader {
    model: TextModelId,
    weights: Option<String>,
    ttl_secs: u64,
    /// Base64 SHA-256 of the session key (reveals nothing of it).
    key_proof: String,
    nodes: Vec<ExportNode>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExportNode {
    id: String,
    parent: Option<String>,
    start: u32,
    end: u32,
    len: u64,
}

impl SessionStore {
    pub fn new(budget: u64, default_ttl: Duration) -> Self {
        Self {
            inner: Mutex::new(Inner {
                sessions: HashMap::new(),
                used: 0,
            }),
            budget,
            default_ttl,
        }
    }

    fn ttl(&self, secs: Option<u64>) -> Duration {
        secs.map_or(self.default_ttl, |s| Duration::from_secs(s.max(1)))
    }

    pub fn create(
        &self,
        model: TextModelId,
        ttl_secs: Option<u64>,
        owner: [u8; 32],
    ) -> SessionInfo {
        let id = random_id();
        let session = Session {
            model,
            owner,
            ttl: self.ttl(ttl_secs),
            touched: Instant::now(),
            weights: None,
            nodes: HashMap::new(),
        };
        let info = session.info(&id);
        self.inner.lock().unwrap().sessions.insert(id, session);
        info
    }

    /// Run `f` on session `id` if `proof` owns it.
    fn owned<R>(
        &self,
        id: &str,
        proof: &[u8],
        f: impl FnOnce(&mut Inner, &str) -> Result<R, SessionError>,
    ) -> Result<R, SessionError> {
        let mut inner = self.inner.lock().unwrap();
        let s = inner
            .sessions
            .get(id)
            .ok_or(SessionError::NotFound("session"))?;
        if !ct_eq(&s.owner, proof) {
            return Err(SessionError::Forbidden);
        }
        f(&mut inner, id)
    }

    pub fn info(&self, id: &str, proof: &[u8]) -> Result<SessionInfo, SessionError> {
        self.owned(id, proof, |inner, id| Ok(inner.sessions[id].info(id)))
    }

    pub fn set_ttl(
        &self,
        id: &str,
        proof: &[u8],
        ttl_secs: u64,
    ) -> Result<SessionInfo, SessionError> {
        let ttl = self.ttl(Some(ttl_secs));
        self.owned(id, proof, |inner, id| {
            let s = inner.sessions.get_mut(id).expect("checked");
            s.ttl = ttl;
            s.touched = Instant::now();
            Ok(s.info(id))
        })
    }

    pub fn delete(&self, id: &str, proof: &[u8]) -> Result<(), SessionError> {
        self.owned(id, proof, |inner, id| {
            let s = inner.sessions.remove(id).expect("checked");
            inner.used -= s.bytes();
            Ok(())
        })
    }

    /// Remove `node` and its whole subtree.
    pub fn delete_node(
        &self,
        id: &str,
        proof: &[u8],
        node: &str,
    ) -> Result<SessionInfo, SessionError> {
        self.owned(id, proof, |inner, id| {
            let s = inner.sessions.get_mut(id).expect("checked");
            if !s.nodes.contains_key(node) {
                return Err(SessionError::NotFound("node"));
            }
            let freed = s.prune(node);
            let info = s.info(id);
            inner.used -= freed;
            Ok(info)
        })
    }

    /// The sealed chain a job resumes from (`from = None`: a new root),
    /// touching the session and the chain so eviction spares them.
    pub fn chain(&self, id: &str, proof: &[u8], from: Option<&str>) -> Result<Chain, SessionError> {
        self.owned(id, proof, |inner, id| {
            let s = inner.sessions.get_mut(id).expect("checked");
            let now = Instant::now();
            s.touched = now;
            let path = match from {
                Some(n) => s.path(n)?,
                None => Vec::new(),
            };
            let mut links = Vec::with_capacity(path.len());
            for k in &path {
                let n = s.nodes.get_mut(k).expect("on path");
                n.touched = now;
                links.push((n.start, n.end, n.sealed.clone()));
            }
            Ok(Chain {
                model: s.model,
                weights: s.weights.clone(),
                links,
            })
        })
    }

    /// Add a job's new node under `parent` (`None` = a root). Returns its id.
    /// Evicts LRU leaves past the budget; a node too big to keep is refused.
    pub fn insert(
        &self,
        id: &str,
        parent: Option<&str>,
        start: u32,
        end: u32,
        sealed: Vec<u8>,
        weights: String,
    ) -> Result<String, SessionError> {
        let len = sealed.len() as u64;
        if len > self.budget {
            return Err(SessionError::Conflict(
                "the reply's state exceeds session_ram_budget; not kept".into(),
            ));
        }
        let mut inner = self.inner.lock().unwrap();
        let s = inner
            .sessions
            .get_mut(id)
            .ok_or(SessionError::NotFound("session (expired?)"))?;
        let want_start = match parent {
            Some(p) => {
                s.nodes
                    .get(p)
                    .ok_or(SessionError::NotFound("parent node (evicted meanwhile)"))?
                    .end
            }
            None => 0,
        };
        if start != want_start {
            return Err(SessionError::Invalid(
                "node does not continue its parent".into(),
            ));
        }
        match &s.weights {
            Some(w) if *w != weights => {
                return Err(SessionError::Conflict(
                    "session was built with different model weights".into(),
                ));
            }
            _ => s.weights = Some(weights),
        }
        let node = random_id();
        let now = Instant::now();
        s.touched = now;
        s.nodes.insert(
            node.clone(),
            Node {
                parent: parent.map(str::to_string),
                start,
                end,
                sealed: Arc::new(sealed),
                created: now,
                touched: now,
            },
        );
        inner.used += len;
        self.evict(&mut inner, Some((id, &node)));
        Ok(node)
    }

    /// Evict least recently used leaves until within budget, sparing `keep`.
    fn evict(&self, inner: &mut Inner, keep: Option<(&str, &str)>) {
        while inner.used > self.budget {
            let victim = inner
                .sessions
                .iter()
                .flat_map(|(sid, s)| {
                    s.nodes
                        .iter()
                        .filter(|(nid, _)| {
                            !s.nodes.values().any(|c| c.parent.as_ref() == Some(*nid))
                        })
                        .filter(move |(nid, _)| keep != Some((sid.as_str(), nid.as_str())))
                        .map(move |(nid, n)| (n.touched, sid.clone(), nid.clone()))
                })
                .min();
            let Some((_, sid, nid)) = victim else { break };
            let s = inner.sessions.get_mut(&sid).expect("victim session");
            let freed = s.prune(&nid);
            inner.used -= freed;
            tracing::info!(freed, "session budget: evicted a node");
        }
    }

    /// The whole session as one blob (its structure in the clear, every
    /// node's state still sealed under the session key).
    pub fn export(&self, id: &str, proof: &[u8]) -> Result<Vec<u8>, SessionError> {
        use base64::Engine;
        self.owned(id, proof, |inner, id| {
            let s = &inner.sessions[id];
            // Parents before children (sorted by start).
            let mut order: Vec<(&String, &Node)> = s.nodes.iter().collect();
            order.sort_by_key(|(_, n)| n.start);
            let header = ExportHeader {
                model: s.model,
                weights: s.weights.clone(),
                ttl_secs: s.ttl.as_secs(),
                key_proof: base64::engine::general_purpose::STANDARD.encode(s.owner),
                nodes: order
                    .iter()
                    .map(|(k, n)| ExportNode {
                        id: (*k).clone(),
                        parent: n.parent.clone(),
                        start: n.start,
                        end: n.end,
                        len: n.sealed.len() as u64,
                    })
                    .collect(),
            };
            let head =
                serde_json::to_vec(&header).map_err(|e| SessionError::Invalid(e.to_string()))?;
            let total: usize = order.iter().map(|(_, n)| n.sealed.len()).sum();
            let mut out = Vec::with_capacity(MAGIC.len() + 4 + head.len() + total);
            out.extend_from_slice(MAGIC);
            out.extend_from_slice(&(head.len() as u32).to_le_bytes());
            out.extend_from_slice(&head);
            for (_, n) in &order {
                out.extend_from_slice(&n.sealed);
            }
            Ok(out)
        })
    }

    /// Restore an exported session under a new id.
    pub fn import(&self, blob: &[u8]) -> Result<SessionInfo, SessionError> {
        use base64::Engine;
        let bad = |m: &str| SessionError::Invalid(format!("session import: {m}"));
        if blob.len() < MAGIC.len() + 4 || &blob[..MAGIC.len()] != MAGIC {
            return Err(bad(if blob.starts_with(b"TFSESS") {
                "an export from another thinfer version (its state layout differs)"
            } else {
                "not a thinfer session export"
            }));
        }
        let hl = u32::from_le_bytes(blob[8..12].try_into().unwrap()) as usize;
        let head = blob
            .get(12..12 + hl)
            .ok_or_else(|| bad("truncated header"))?;
        let header: ExportHeader =
            serde_json::from_slice(head).map_err(|e| bad(&format!("header: {e}")))?;
        let owner: [u8; 32] = base64::engine::general_purpose::STANDARD
            .decode(&header.key_proof)
            .ok()
            .and_then(|v| v.try_into().ok())
            .ok_or_else(|| bad("keyProof"))?;
        let mut at = 12 + hl;
        let mut nodes = HashMap::new();
        for n in &header.nodes {
            let len = n.len as usize;
            let sealed = blob
                .get(at..at + len)
                .ok_or_else(|| bad("truncated node"))?;
            at += len;
            let now = Instant::now();
            if nodes
                .insert(
                    n.id.clone(),
                    Node {
                        parent: n.parent.clone(),
                        start: n.start,
                        end: n.end,
                        sealed: Arc::new(sealed.to_vec()),
                        created: now,
                        touched: now,
                    },
                )
                .is_some()
            {
                return Err(bad("duplicate node id"));
            }
        }
        if at != blob.len() {
            return Err(bad("trailing bytes"));
        }
        let session = Session {
            model: header.model,
            owner,
            ttl: self.ttl(Some(header.ttl_secs)),
            touched: Instant::now(),
            weights: header.weights,
            nodes,
        };
        for k in session.nodes.keys() {
            session.path(k)?;
        }
        let bytes = session.bytes();
        if bytes > self.budget {
            return Err(SessionError::Conflict(
                "session exceeds session_ram_budget".into(),
            ));
        }
        let id = random_id();
        let info = session.info(&id);
        let mut inner = self.inner.lock().unwrap();
        inner.sessions.insert(id, session);
        inner.used += bytes;
        self.evict(&mut inner, None);
        Ok(info)
    }

    /// Forget sessions unused past their TTL. Returns the count.
    pub fn reap(&self) -> usize {
        let mut inner = self.inner.lock().unwrap();
        let expired: Vec<String> = inner
            .sessions
            .iter()
            .filter(|(_, s)| s.touched.elapsed() > s.ttl)
            .map(|(k, _)| k.clone())
            .collect();
        for k in &expired {
            let s = inner.sessions.remove(k).expect("listed");
            inner.used -= s.bytes();
        }
        expired.len()
    }
}

/// Spawn the session reaper (every 30 s).
pub fn spawn_reaper(store: Arc<SessionStore>) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(30));
        loop {
            tick.tick().await;
            let n = store.reap();
            if n > 0 {
                tracing::info!(expired = n, "session reaper removed idle sessions");
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    const OWNER: [u8; 32] = [7; 32];

    fn store(budget: u64) -> SessionStore {
        SessionStore::new(budget, Duration::from_secs(600))
    }

    #[test]
    fn tree_branches_and_chains() {
        let st = store(1 << 20);
        let id = st.create(TextModelId::DEFAULT, None, OWNER).id;
        let root = st
            .insert(&id, None, 0, 10, vec![1; 10], "w".into())
            .unwrap();
        let a = st
            .insert(&id, Some(&root), 10, 15, vec![2; 5], "w".into())
            .unwrap();
        let b = st
            .insert(&id, Some(&root), 10, 12, vec![3; 2], "w".into())
            .unwrap();
        let ch = st.chain(&id, &OWNER, Some(&b)).unwrap();
        assert_eq!(
            ch.links.iter().map(|l| (l.0, l.1)).collect::<Vec<_>>(),
            [(0, 10), (10, 12)]
        );
        assert_eq!(ch.weights.as_deref(), Some("w"));
        assert!(
            st.insert(&id, Some(&a), 14, 20, vec![0], "w".into())
                .is_err()
        );
        assert!(
            st.insert(&id, Some(&a), 15, 20, vec![0], "other".into())
                .is_err()
        );
        let info = st.delete_node(&id, &OWNER, &root).unwrap();
        assert!(info.nodes.is_empty() && info.bytes == 0);
    }

    #[test]
    fn proof_gates_everything() {
        let st = store(1 << 20);
        let id = st.create(TextModelId::DEFAULT, None, OWNER).id;
        assert!(matches!(
            st.info(&id, &[0; 32]),
            Err(SessionError::Forbidden)
        ));
        assert!(st.chain(&id, &[], None).is_err());
        assert!(st.export(&id, &[1; 32]).is_err());
        assert!(st.delete(&id, &OWNER).is_ok());
        assert!(st.info(&id, &OWNER).is_err());
    }

    #[test]
    fn budget_evicts_lru_leaves() {
        let st = store(25);
        let id = st.create(TextModelId::DEFAULT, None, OWNER).id;
        let root = st
            .insert(&id, None, 0, 10, vec![0; 10], "w".into())
            .unwrap();
        let old = st
            .insert(&id, Some(&root), 10, 11, vec![0; 10], "w".into())
            .unwrap();
        std::thread::sleep(Duration::from_millis(5));
        // Over budget: the least recently used leaf (`old`) goes.
        let new = st
            .insert(&id, Some(&root), 10, 12, vec![0; 10], "w".into())
            .unwrap();
        assert!(st.chain(&id, &OWNER, Some(&old)).is_err());
        assert!(st.chain(&id, &OWNER, Some(&new)).is_ok());
        assert!(st.insert(&id, None, 0, 1, vec![0; 26], "w".into()).is_err());
    }

    #[test]
    fn export_import_round_trips() {
        let st = store(1 << 20);
        let id = st.create(TextModelId::DEFAULT, Some(99), OWNER).id;
        let root = st.insert(&id, None, 0, 3, vec![5; 3], "w".into()).unwrap();
        st.insert(&id, Some(&root), 3, 4, vec![6; 1], "w".into())
            .unwrap();
        let blob = st.export(&id, &OWNER).unwrap();
        let back = st.import(&blob).unwrap();
        assert_ne!(back.id, id);
        assert_eq!(back.ttl_secs, 99);
        assert_eq!(back.nodes.len(), 2);
        let ch = st.chain(&back.id, &OWNER, Some(&root)).unwrap();
        assert_eq!(*ch.links[0].2, vec![5; 3]);
        let mut cut = blob.clone();
        cut.pop();
        assert!(st.import(&cut).is_err());
    }

    #[test]
    fn reap_forgets_idle_sessions() {
        let st = SessionStore::new(1 << 20, Duration::from_secs(600));
        let id = st.create(TextModelId::DEFAULT, Some(1), OWNER).id;
        st.insert(&id, None, 0, 1, vec![0; 4], "w".into()).unwrap();
        std::thread::sleep(Duration::from_millis(1100));
        assert_eq!(st.reap(), 1);
        assert_eq!(st.inner.lock().unwrap().used, 0);
    }
}
