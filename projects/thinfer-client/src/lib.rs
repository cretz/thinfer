//! A client for the `thinfer-serve` job API: image, video, face-swap and text
//! (LLM) generation on a thinfer box, end-to-end encrypted.
//!
//! Every job is sealed under a fresh per-job key ([`jobkey::JobKey`]) before
//! it leaves the client. The server queues ciphertext, asks for the key only
//! when a worker is about to run the job, and returns the artifact sealed
//! under the same key; [`Client`] does all of that for you and hands back
//! plaintext. Text sessions ([`Session`]) keep the model's state on the
//! server, sealed under a key only the client holds.
//!
#![cfg_attr(feature = "http", doc = "```no_run")]
#![cfg_attr(not(feature = "http"), doc = "```ignore")]
//! use thinfer_client::wire::{ImageSpec, JobSpec};
//! use thinfer_client::{Client, Update};
//!
//! # async fn demo() -> Result<(), thinfer_client::Error> {
//! let client = Client::builder("https://gpu-box:8080")
//!     .token("secret")
//!     .build()?;
//! let spec = JobSpec::Image(ImageSpec::new("a lighthouse at dusk"));
//! let out = client
//!     .run(&spec, |u| {
//!         if let Update::Progress(stage) = u {
//!             eprintln!("{stage:?}");
//!         }
//!     })
//!     .await?;
//! std::fs::write("lighthouse.png", &*out.artifact).unwrap();
//! # Ok(())
//! # }
//! ```
//!
//! Features: `http` (default) is the async [`Client`] (reqwest + rustls,
//! needs a Tokio runtime). Without it the crate is just the wire types, model
//! ids and encryption. `clap` and `utoipa` add derives for front ends.

mod error;
pub mod jobkey;
pub mod model;
pub mod progress;
pub mod text_format;
pub mod tools;
pub mod wire;

#[cfg(feature = "http")]
mod client;
#[cfg(feature = "http")]
mod sse;

#[cfg(feature = "http")]
pub use client::{Client, ClientBuilder, Finished, Job, Output, Session, SubmitOptions, Update};
pub use error::Error;
pub use jobkey::JobKey;
