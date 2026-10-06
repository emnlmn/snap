//! Repository → chunks for `snap grep`. ripgrep's own walker (`ignore`)
//! decides which files exist — .gitignore, .ignore, hidden files, globs —
//! and a language-agnostic splitter cuts each file into declaration-sized
//! chunks. A chunk is the unit everything downstream ranks, snapshots and
//! cites, so its bounds must be stable: the same file text always splits
//! the same way, and an unchanged chunk keeps its KV snapshot.
#![allow(dead_code)] // wired by grep.rs

use std::path::{Path, PathBuf};

use anyhow::Result;

/// Files above this are skipped: generated or data, not code to read.
pub const MAX_FILE_BYTES: u64 = 1 << 20;
/// Chunk budget: what one relevance probe reads. ~400 tokens of code keeps
/// a snapshot near 17 MB of f16 KV on snap1-2b (42 layers, 2 KV heads).
pub const MAX_LINES: usize = 60;
pub const MAX_CHARS: usize = 1600;

#[derive(Debug, Clone, Default)]
pub struct WalkOpts {
    /// search hidden files and directories
    pub hidden: bool,
    /// don't respect .gitignore / .ignore / .rgignore / git excludes
    pub no_ignore: bool,
    /// ripgrep `-g` globs: `*.rs` includes, `!vendor/**` excludes
    pub globs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// root-relative, `/`-separated on every platform
    pub path: String,
    /// 1-based first line
    pub start: usize,
    /// 1-based last line, inclusive
    pub end: usize,
    /// the declared name, best effort (`pub fn new(` → "new")
    pub symbol: Option<String>,
    /// lines start..=end verbatim, joined by '\n', no trailing newline
    pub text: String,
}

/// Every searchable file under `root` as ripgrep would list it:
/// (root-relative `/` path, filesystem path), sorted by relative path.
pub fn files(root: &Path, opts: &WalkOpts) -> Result<Vec<(String, PathBuf)>> {
    let _ = (root, opts);
    todo!()
}

/// File text, or None for what isn't code to read: over MAX_FILE_BYTES,
/// binary (a NUL in the first 8 KiB), not UTF-8, or minified.
pub fn read_text(path: &Path) -> Option<String> {
    let _ = path;
    todo!()
}

/// Split one file into chunks that cover every non-blank line exactly once,
/// in order, each within MAX_LINES and MAX_CHARS.
pub fn split(path: &str, text: &str) -> Vec<Chunk> {
    let _ = (path, text);
    todo!()
}

/// `files` + `read_text` + `split` over the whole tree, in path order.
pub fn corpus(root: &Path, opts: &WalkOpts) -> Result<Vec<Chunk>> {
    let _ = (root, opts);
    todo!()
}

/// Which chunks match `pattern`: a Rust regex over the chunk text, smart
/// case like ripgrep's `-S` (case-insensitive unless the pattern has an
/// uppercase letter).
pub fn grep(chunks: &[Chunk], pattern: &str) -> Result<Vec<bool>> {
    let _ = (chunks, pattern);
    todo!()
}
