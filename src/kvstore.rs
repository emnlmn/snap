//! On-disk KV snapshots for `snap grep`: what the model computed reading a
//! chunk, kept so that a query decodes only its own question. Content-
//! addressed (the key hashes the exact prefix tokens, so a changed chunk,
//! path, template or prompt format is simply a different key), append-only
//! (a pack of blobs plus a log of fixed-size index records; a torn tail is
//! dropped on open) and bound to the engine that wrote it: a snapshot is
//! only valid for the same weights and KV layout, so a binding mismatch
//! refuses to open instead of feeding the model foreign memory.
#![allow(dead_code)] // wired by grep.rs

use std::collections::HashSet;
use std::path::Path;

use anyhow::Result;

pub struct Store {
    _todo: (),
}

impl Store {
    /// Open or create the store in `dir`, bound to `binding` (model identity
    /// and KV layout, opaque here). Fails if `dir` was created under a
    /// different binding or another process holds the store.
    pub fn open(dir: &Path, binding: &str) -> Result<Store> {
        let _ = (dir, binding);
        todo!()
    }

    /// Snapshots stored.
    pub fn len(&self) -> usize {
        todo!()
    }

    /// Bytes the pack holds, dead entries included until `compact`.
    pub fn bytes(&self) -> u64 {
        todo!()
    }

    /// Tokens the snapshot under `key` holds, if one is stored.
    pub fn tokens(&self, key: u128) -> Option<usize> {
        let _ = key;
        todo!()
    }

    pub fn read(&self, key: u128) -> Result<Option<Vec<u8>>> {
        let _ = key;
        todo!()
    }

    /// Append a snapshot; a key already stored is left as is.
    pub fn put(&mut self, key: u128, tokens: usize, blob: &[u8]) -> Result<()> {
        let _ = (key, tokens, blob);
        todo!()
    }

    /// Make every put so far durable: the pack reaches disk before the
    /// index records that point into it.
    pub fn sync(&mut self) -> Result<()> {
        todo!()
    }

    /// Read `keys` in order on a background thread, at most `depth` blobs
    /// ahead of the consumer.
    pub fn prefetch(&self, keys: Vec<u128>, depth: usize) -> Prefetch {
        let _ = (keys, depth);
        todo!()
    }

    /// Rewrite pack and index keeping only `live` keys.
    pub fn compact(&mut self, live: &HashSet<u128>) -> Result<()> {
        let _ = live;
        todo!()
    }
}

/// Blobs read ahead by `Store::prefetch`, in the order asked; a key the
/// store lacks yields Ok(None).
pub struct Prefetch {
    _todo: (),
}

impl Iterator for Prefetch {
    type Item = (u128, Result<Option<Vec<u8>>>);

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

/// FNV-1a over 128 bits: stable across builds and platforms, which is all
/// a content address needs (std's hasher promises neither).
pub fn key(bytes: &[u8]) -> u128 {
    let _ = bytes;
    todo!()
}
