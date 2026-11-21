//! The client's error type.

/// Everything a client call can fail with. Messages carry no job data (no
/// prompt, reply or key material).
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The base URL is not an `http(s)://` URL.
    InvalidUrl(String),
    /// A key is malformed (not base64, wrong length).
    InvalidKey(String),
    /// A sealed blob did not open: wrong key, or corrupt / truncated data.
    Decrypt,
    /// The connection or the HTTP exchange failed.
    Transport(String),
    /// The server answered with a non-success status and this message.
    Server { status: u16, message: String },
    /// The server's answer did not decode, or the event stream broke off.
    Protocol(String),
    /// The job ran and failed with this message.
    JobFailed(String),
    /// The job was cancelled.
    Cancelled,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidUrl(u) => write!(f, "not an http(s) URL: {u:?}"),
            Error::InvalidKey(m) => write!(f, "invalid key: {m}"),
            Error::Decrypt => f.write_str("decrypt failed (wrong key or corrupt data)"),
            Error::Transport(m) => write!(f, "transport: {m}"),
            Error::Server { status, message } => write!(f, "server returned {status}: {message}"),
            Error::Protocol(m) => write!(f, "protocol: {m}"),
            Error::JobFailed(m) => write!(f, "job failed: {m}"),
            Error::Cancelled => f.write_str("job was cancelled"),
        }
    }
}

impl std::error::Error for Error {}

/// For callers whose errors are plain strings.
impl From<Error> for String {
    fn from(e: Error) -> Self {
        e.to_string()
    }
}
