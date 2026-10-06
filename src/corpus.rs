//! Repository → chunks for `snap grep`. ripgrep's own walker (`ignore`)
//! decides which files exist — .gitignore, .ignore, hidden files, globs —
//! and a language-agnostic splitter cuts each file into declaration-sized
//! chunks, section-sized for prose. A chunk is the unit everything
//! downstream ranks, snapshots and cites, so its bounds must be stable: the
//! same file text always splits the same way, and an unchanged chunk keeps
//! its KV snapshot.
#![allow(dead_code)] // wired by grep.rs

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{anyhow, ensure, Result};
use ignore::overrides::OverrideBuilder;
use ignore::WalkBuilder;
use regex::{Regex, RegexBuilder};

/// Files above this are skipped: generated or data, not code to read.
pub const MAX_FILE_BYTES: u64 = 1 << 20;
/// Chunk budget: what one relevance probe reads. ~400 tokens of code keeps
/// a snapshot near 17 MB of f16 KV on snap1-2b (42 layers, 2 KV heads).
pub const MAX_LINES: usize = 60;
pub const MAX_CHARS: usize = 1600;

/// A NUL in this many leading bytes marks a file binary, as in git and ripgrep.
const SNIFF_BYTES: usize = 8192;
/// Average non-blank line length past which a file is minified or generated.
/// Prose keeps whole paragraphs on one line (hundreds of bytes); bundles and
/// base64 blobs run to thousands.
const MINIFIED_AVG: usize = 1000;
/// Prose: a heading opens a new chunk unless the current one is shorter than
/// this, so a title and its line of intro join the first section.
const MIN_SECTION: usize = 5;

/// Lockfiles and minified/map files are noise for code search: they stay out
/// even under a wildcard include (`-g '*.json'`) unless a glob names them
/// (`-g Cargo.lock`, `-g '*.min.js'`).
const NOISE_NAMES: [&str; 5] = [
    "Cargo.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
];
const NOISE_SUFFIXES: [&str; 3] = [".min.js", ".min.css", ".map"];

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
    /// the declared name, best effort (`pub fn new(` → "new"); in prose the
    /// section heading; a piece of a big impl or class, the method it holds
    /// or continues
    pub symbol: Option<String>,
    /// lines start..=end verbatim, joined by '\n', no trailing newline
    pub text: String,
}

fn is_noise(name: &str) -> bool {
    NOISE_NAMES.contains(&name) || NOISE_SUFFIXES.iter().any(|s| name.ends_with(s))
}

/// Every searchable file under `root` (a directory) as ripgrep would list
/// it: (root-relative `/` path, filesystem path), sorted by relative path.
pub fn files(root: &Path, opts: &WalkOpts) -> Result<Vec<(String, PathBuf)>> {
    ensure!(root.is_dir(), "{}: not a directory", root.display());
    // The last matching glob wins: noise excludes follow the user's globs so
    // wildcards can't re-include them, and a glob naming noise follows them.
    let mut ov = OverrideBuilder::new(root);
    for g in &opts.globs {
        ov.add(g)?;
    }
    for n in NOISE_NAMES {
        ov.add(&format!("!{n}"))?;
    }
    for s in NOISE_SUFFIXES {
        ov.add(&format!("!*{s}"))?;
    }
    for g in &opts.globs {
        if !g.starts_with('!') && g.rsplit('/').next().is_some_and(is_noise) {
            ov.add(g)?;
        }
    }
    let ig = !opts.no_ignore;
    let mut walk = WalkBuilder::new(root);
    walk.hidden(!opts.hidden)
        .ignore(ig)
        .git_ignore(ig)
        .git_global(ig)
        .git_exclude(ig)
        .parents(ig)
        .require_git(false)
        .overrides(ov.build()?)
        .filter_entry(|e| e.file_name() != ".git");
    if ig {
        walk.add_custom_ignore_filename(".rgignore");
    }
    let mut out = Vec::new();
    for entry in walk.build() {
        let e = match entry {
            Ok(e) => e,
            Err(err) => {
                tracing::warn!("{err}");
                continue;
            }
        };
        let Ok(rel) = e.path().strip_prefix(root) else {
            continue;
        };
        if e.file_type().is_some_and(|t| t.is_file()) {
            let rel = rel.components().map(|c| c.as_os_str().to_string_lossy());
            out.push((rel.collect::<Vec<_>>().join("/"), e.into_path()));
        }
    }
    out.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// File text, or None for what isn't code to read: over MAX_FILE_BYTES,
/// binary (a NUL in the first 8 KiB), not UTF-8, or minified (non-blank
/// lines averaging over 1,000 bytes).
pub fn read_text(path: &Path) -> Option<String> {
    if std::fs::metadata(path).ok()?.len() > MAX_FILE_BYTES {
        return None;
    }
    let bytes = std::fs::read(path).ok()?;
    if bytes[..bytes.len().min(SNIFF_BYTES)].contains(&0) {
        return None;
    }
    let body = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&bytes);
    let text = std::str::from_utf8(body).ok()?;
    let (n, total) = text
        .lines()
        .filter(|l| !is_blank(l))
        .fold((0, 0), |(n, total), l| (n + 1, total + l.len()));
    (total <= MINIFIED_AVG * n).then(|| text.to_owned())
}

/// 0-based line indices s..=e, non-blank at both edges.
type Span = (usize, usize);

/// A file's lines with char prefix sums, so any run's size is O(1).
struct Lines<'a> {
    lines: Vec<&'a str>,
    chars: Vec<usize>,
}

impl<'a> Lines<'a> {
    fn new(text: &'a str) -> Self {
        let lines: Vec<&str> = text.lines().collect();
        let sums = lines.iter().scan(0, |n, l| {
            *n += l.chars().count();
            Some(*n)
        });
        let chars = std::iter::once(0).chain(sums).collect();
        Lines { lines, chars }
    }

    /// Do lines s..=e, joined, stay within both budgets?
    fn fits(&self, s: usize, e: usize) -> bool {
        e - s < MAX_LINES && self.chars[e + 1] - self.chars[s] + (e - s) <= MAX_CHARS
    }

    /// End of the longest chunk from `s` that fits, up to `limit`. Lines are
    /// atomic: one line always stands, however long.
    fn grow(&self, s: usize, limit: usize) -> usize {
        let mut e = s;
        while e < limit && self.fits(s, e + 1) {
            e += 1;
        }
        e
    }
}

/// Split one file into chunks that cover every non-blank line exactly once,
/// in order, each within MAX_LINES and MAX_CHARS. Code is cut along
/// declarations, prose along sections: a chunk holds whole units, or one
/// piece of a unit too big for any chunk. A piece of code is named for what it
/// holds or continues (see `piece_names`), a piece of prose for its section.
pub fn split(path: &str, text: &str) -> Vec<Chunk> {
    let t = Lines::new(text);
    let prose = prose_ext(path);
    let heads = match &prose {
        Some(ext) => headings(ext, &t.lines),
        None => vec![None; t.lines.len()],
    };
    let symbol = |(s, e): Span| match prose {
        Some(_) => heads[s..=e].iter().flatten().next().map(|h| h.to_string()),
        None => first_declared(&t.lines[s..=e]),
    };
    let chunk = |(s, e): Span, symbol: Option<String>| Chunk {
        path: path.to_string(),
        start: s + 1,
        end: e + 1,
        symbol,
        text: t.lines[s..=e].join("\n"),
    };
    let mut out = Vec::new();
    let mut cur: Option<Span> = None;
    for unit in units(&t.lines, &heads, prose.is_some()) {
        let (us, ue) = (unit[0].0, unit[unit.len() - 1].1);
        if let Some((s, e)) = cur.take() {
            // a heading closes a chunk that already has some body
            if t.fits(s, ue) && !(heads[us].is_some() && e - s + 1 >= MIN_SECTION) {
                cur = Some((s, ue));
                continue;
            }
            out.push(chunk((s, e), symbol((s, e))));
        }
        if us == ue || t.fits(us, ue) {
            cur = Some((us, ue));
        } else {
            let sym = symbol((us, ue));
            let ps = pieces(&t, &unit);
            let names = match prose {
                Some(_) => vec![sym; ps.len()],
                None => piece_names(&t.lines, &ps, sym),
            };
            out.extend(ps.into_iter().zip(names).map(|(p, name)| chunk(p, name)));
        }
    }
    out.extend(cur.map(|c| chunk(c, symbol(c))));
    out
}

/// Units, each a list of blocks (maximal runs of non-blank lines). In code a
/// block opens a unit when it starts at the file's least indent and isn't a
/// closing token; any other block continues the unit before it, as a blank
/// line inside a body does. In prose a heading opens a unit: a unit is a
/// section.
fn units(lines: &[&str], heads: &[Option<&str>], prose: bool) -> Vec<Vec<Span>> {
    let base = lines
        .iter()
        .filter(|l| !is_blank(l))
        .map(|l| indent(l))
        .min();
    let mut units: Vec<Vec<Span>> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if is_blank(lines[i]) {
            i += 1;
            continue;
        }
        let s = i;
        i += 1;
        while i < lines.len() && !is_blank(lines[i]) && heads[i].is_none() {
            i += 1;
        }
        let opens = heads[s].is_some()
            || (!prose && Some(indent(lines[s])) == base && !closes(lines[s].trim_start()));
        match units.last_mut() {
            Some(u) if !opens => u.push((s, i - 1)),
            _ => units.push(vec![(s, i - 1)]),
        }
    }
    units
}

/// An oversized unit cut into chunks: greedy over its blocks, so cuts fall at
/// blank lines, and a block too big on its own is cut at line boundaries.
fn pieces(t: &Lines, unit: &[Span]) -> Vec<Span> {
    let mut out = Vec::new();
    let mut cur: Option<Span> = None;
    for &(bs, be) in unit {
        if let Some((s, _)) = cur {
            if t.fits(s, be) {
                cur = Some((s, be));
                continue;
            }
            out.extend(cur.take());
        }
        let mut s = bs;
        loop {
            let e = t.grow(s, be);
            if e == be {
                cur = Some((s, be));
                break;
            }
            out.push((s, e));
            s = e + 1;
        }
    }
    out.extend(cur);
    out
}

/// What the pieces of an oversized unit of code are called: the first item a
/// piece declares (a method of a big impl), else the item it continues, the
/// nearest earlier one indented less than the piece's first line (the method
/// whose body it is in), else the unit's own name. A `const` or `let` inside a
/// body is no item.
fn piece_names<'a>(
    lines: &[&'a str],
    pieces: &[Span],
    unit: Option<String>,
) -> Vec<Option<String>> {
    // items declared so far, outermost first: (indent, name)
    let mut open: Vec<(usize, &'a str)> = Vec::new();
    let mut names = Vec::with_capacity(pieces.len());
    for &(s, e) in pieces {
        let first = indent(lines[s]);
        let continued = open.iter().rev().find(|&&(i, _)| i < first);
        let continued = continued.map(|&(_, name)| name);
        let mut declares = None;
        for line in lines[s..=e].iter().copied() {
            if let Some((name, true)) = declared(line) {
                let at = indent(line);
                open.retain(|&(i, _)| i < at);
                open.push((at, name));
                declares.get_or_insert(name);
            }
        }
        let name = declares.or(continued).map(str::to_owned);
        names.push(name.or_else(|| unit.clone()));
    }
    names
}

fn is_blank(line: &str) -> bool {
    line.trim().is_empty()
}

/// Leading whitespace width, a tab counting 4.
fn indent(line: &str) -> usize {
    line.chars()
        .take_while(|c| c.is_whitespace())
        .map(|c| if c == '\t' { 4 } else { 1 })
        .sum()
}

/// A closing token (`}`, `)`, `]`, `end`, `fi`, `done`, `esac`, `</`) starts
/// this trimmed line: the block it opens belongs to the unit before it.
fn closes(line: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^(?:[})\]]|</|(?:end|fi|done|esac)\b)").unwrap())
        .is_match(line)
}

/// The name a line declares, best effort: Rust, Python, JS/TS, Go, and the
/// keyword-led declarations of their cousins. Modifiers come first
/// (`pub(crate) async`, `export default`); the capture groups are, in order,
/// `impl X for Type`, `impl Type`, a Go method, `fn name`-style keywords,
/// `macro_rules! name` and `const`/`static`/`let`/`var name:`/`=`. Only the
/// last is no item: inside a body it is a local.
fn declared(line: &str) -> Option<(&str, bool)> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(
            r#"(?x) ^\s*
            (?: (?: pub(?:\([^)]*\))? | public | private | protected | internal | export
                  | default | declare | abstract | final | static | async | unsafe
                  | extern(?:\s+"[^"]*")? | const | override | open | sealed | data
                  | inline | suspend | partial | local ) \s+ )*
            (?: impl\b [^{;]*? \bfor\s+ &? \s* (?:'\w+\s+)? (?:mut\s+)? (?:\w+::)* (\w+)
              | impl\b \s* (?:<.*?>\s*)? (?:\w+::)* (\w+)
              | func \s* \( [^)]* \) \s* (\w+)
              | (?: fn | def | func | fun | function\*? | class | struct | enum | trait
                  | interface | protocol | extension | object | module | mod | namespace
                  | record | type ) \s+ (?:self\.)? (\w+)
              | macro_rules! \s* (\w+)
              | (?: const | static | let | var ) \s+ (?:mut\s+)? (\w+) \s* [:=] )"#,
        )
        .unwrap()
    });
    let caps = re.captures(line)?;
    let (i, m) = caps
        .iter()
        .enumerate()
        .skip(1)
        .find_map(|(i, m)| Some((i, m?)))?;
    Some((m.as_str(), i + 1 < caps.len()))
}

/// The first name `lines` declare.
fn first_declared(lines: &[&str]) -> Option<String> {
    let found = lines.iter().find_map(|l| declared(l));
    found.map(|(name, _)| name.to_owned())
}

/// Extension of a prose document: split by section, not by declaration.
fn prose_ext(path: &str) -> Option<String> {
    let ext = Path::new(path).extension()?.to_str()?.to_ascii_lowercase();
    ["md", "markdown", "mdx", "rst", "adoc", "txt", "org"]
        .contains(&ext.as_str())
        .then_some(ext)
}

/// The heading each line begins, if any, in a prose file: ATX (`## Title`;
/// `= Title` in AsciiDoc, `* Title` in org), a setext line over `====` or
/// `----`, and rst's title between two such rules. A setext heading's underline
/// stays with it (no heading begins there). Markdown's fenced code and YAML
/// front matter hold none: a `# comment` in a shell block is not a section.
fn headings<'a>(ext: &str, lines: &[&'a str]) -> Vec<Option<&'a str>> {
    let mark = match ext {
        "org" => '*',
        "adoc" => '=',
        _ => '#',
    };
    // org has no setext, and AsciiDoc's `----` is a listing fence, not an underline
    let setext = !matches!(ext, "org" | "adoc");
    // `~~~` is an rst underline, so fences are markdown's alone
    let md = matches!(ext, "md" | "markdown" | "mdx");
    let rule = |i: usize| lines.get(i).is_some_and(|l| is_rule(l));
    // a line of text that could be a heading: neither blank nor a rule
    let title = |i: usize| {
        let line = lines.get(i)?;
        (!is_blank(line) && !is_rule(line)).then(|| line.trim())
    };
    let mut heads = vec![None; lines.len()];
    let mut fenced = false;
    // the closing `---` of front matter would underline its last key
    let mut i = match lines.first() {
        Some(l) if md && l.trim_end() == "---" => lines[1..]
            .iter()
            .position(|l| matches!(l.trim_end(), "---" | "..."))
            .map_or(0, |k| k + 2),
        _ => 0,
    };
    while i < lines.len() {
        let line = lines[i];
        let t = line.trim();
        if md && (t.starts_with("```") || t.starts_with("~~~")) {
            fenced = !fenced;
        } else if !fenced && !t.is_empty() {
            if let Some(h) = atx(line, mark) {
                heads[i] = Some(h);
            } else if setext && rule(i + 1) && title(i).is_some() {
                heads[i] = title(i);
                i += 1;
            } else if setext && is_rule(line) && rule(i + 2) && title(i + 1).is_some() {
                heads[i] = title(i + 1);
                i += 2;
            }
        }
        i += 1;
    }
    heads
}

/// `## Title` → "Title": up to six `mark`s at column 0 (any number for org's
/// `*`) and a space.
fn atx(line: &str, mark: char) -> Option<&str> {
    let rest = line.trim_start_matches(mark);
    let n = line.len() - rest.len();
    let text = rest.strip_prefix([' ', '\t'])?.trim();
    (n > 0 && (mark == '*' || n <= 6) && !text.is_empty()).then_some(text)
}

/// A setext/rst underline: three or more `=` or `-`, nothing else.
fn is_rule(line: &str) -> bool {
    let t = line.trim_end();
    t.len() >= 3 && (t.bytes().all(|b| b == b'=') || t.bytes().all(|b| b == b'-'))
}

/// `files` + `read_text` + `split` over the whole tree, in path order.
pub fn corpus(root: &Path, opts: &WalkOpts) -> Result<Vec<Chunk>> {
    let mut out = Vec::new();
    for (rel, path) in files(root, opts)? {
        if let Some(text) = read_text(&path) {
            out.extend(split(&rel, &text));
        }
    }
    Ok(out)
}

/// Which chunks match `pattern`: a Rust regex over the chunk text, smart
/// case like ripgrep's `-S` (case-insensitive unless the pattern has an
/// uppercase letter).
pub fn grep(chunks: &[Chunk], pattern: &str) -> Result<Vec<bool>> {
    let re = RegexBuilder::new(pattern)
        .multi_line(true)
        .case_insensitive(!has_upper(pattern))
        .build()
        .map_err(|e| anyhow!("invalid pattern: {e}"))?;
    Ok(chunks.iter().map(|c| re.is_match(&c.text)).collect())
}

/// An uppercase literal, as ripgrep reads it: escapes like `\S` and `\W` don't count.
fn has_upper(pattern: &str) -> bool {
    let mut it = pattern.chars();
    while let Some(c) = it.next() {
        if c == '\\' {
            it.next();
        } else if c.is_uppercase() {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    const RUST: &str = r#"//! Idle-connection pool.

use std::collections::HashMap;
use std::sync::Mutex;

/// Idle connections, keyed by host.
#[derive(Debug, Default)]
pub struct Pool {
    idle: Mutex<HashMap<String, Vec<Conn>>>,
    limit: usize,
}

impl Pool {
    pub fn new(limit: usize) -> Self {
        Self { idle: Mutex::default(), limit }
    }

    /// Take an idle connection for `host`, if one is left alive.
    pub(crate) async fn acquire(&self, host: &str) -> Option<Conn> {
        let mut idle = self.idle.lock().unwrap();
        let conns = idle.get_mut(host)?;
        while let Some(conn) = conns.pop() {
            if conn.is_alive() {
                return Some(conn);
            }
        }
        None
    }

    /// Hand a connection back; it is dropped once the host holds `limit`.
    pub fn release(&self, host: &str, conn: Conn) {
        let mut idle = self.idle.lock().unwrap();
        let conns = idle.entry(host.to_string()).or_default();
        if conns.len() < self.limit {
            conns.push(conn);
        }
    }
}

impl<T: Into<String>> From<T> for Pool {
    fn from(host: T) -> Self {
        let pool = Pool::new(4);
        pool.release(&host.into(), Conn::default());
        pool
    }
}

/// Open `n` connections to `host` ahead of demand.
pub async fn warm(pool: &Pool, host: &str, n: usize) -> Result<usize, Error> {
    let mut opened = 0;
    for attempt in 0..n {
        match Conn::connect(host).await {
            Ok(conn) => {
                pool.release(host, conn);
                opened += 1;
            }
            Err(err) if err.is_transient() => {
                tracing::warn!(%host, attempt, "warm-up connect failed: {err}");
                tokio::time::sleep(Duration::from_millis(50 << attempt)).await;
            }
            Err(err) => return Err(err),
        }
    }
    Ok(opened)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_respects_limit() {
        let pool = Pool::new(1);
        pool.release("a", Conn::default());
        pool.release("a", Conn::default());
        assert_eq!(pool.idle.lock().unwrap()["a"].len(), 1);
    }
}
"#;

    const PYTHON: &str = r#""""Retry helpers."""

import time
from functools import wraps


def retry(times=3, delay=0.5):
    """Retry a function when it raises."""

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return fn(*args, **kwargs)
                except Exception:
                    if attempt == times - 1:
                        raise
                    time.sleep(delay * 2 ** attempt)

        return wrapper

    return decorator


class Backoff:
    """Exponential backoff with a ceiling."""

    def __init__(self, base=0.5, cap=30.0):
        self.base = base
        self.cap = cap
        self.attempt = 0

    def next_delay(self):
        delay = min(self.cap, self.base * 2 ** self.attempt)
        self.attempt += 1
        return delay

    def reset(self):
        self.attempt = 0


async def sleep_for(backoff):
    await asyncio.sleep(backoff.next_delay())


class Budget:
    """Caps the total time spent retrying."""

    def __init__(self, seconds):
        self.deadline = time.monotonic() + seconds

    def left(self):
        return max(0.0, self.deadline - time.monotonic())

    def spent(self, total):
        return 1.0 - self.left() / total if total else 1.0

    def spend(self, delay):
        if delay > self.left():
            raise TimeoutError("retry budget exhausted")
        time.sleep(delay)
"#;

    const GO: &str = r#"package store

import (
	"context"
	"database/sql"
	"fmt"
)

// Store wraps the users table.
type Store struct {
	db *sql.DB
}

// New opens a store on db.
func New(db *sql.DB) *Store {
	return &Store{db: db}
}

// Get returns the user with the given id.
func (s *Store) Get(ctx context.Context, id int64) (*User, error) {
	row := s.db.QueryRowContext(ctx, `SELECT id, name, email FROM users WHERE id = $1`, id)
	var u User
	if err := row.Scan(&u.ID, &u.Name, &u.Email); err != nil {
		if err == sql.ErrNoRows {
			return nil, ErrNotFound
		}
		return nil, fmt.Errorf("get user %d: %w", id, err)
	}
	return &u, nil
}

// Rename changes the user's display name.
func (s *Store) Rename(ctx context.Context, id int64, name string) error {
	res, err := s.db.ExecContext(ctx, `UPDATE users SET name = $1 WHERE id = $2`, name, id)
	if err != nil {
		return fmt.Errorf("rename user %d: %w", id, err)
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return ErrNotFound
	}
	return nil
}

// Close releases the connection pool.
func (s *Store) Close() error {
	return s.db.Close()
}

// List returns users ordered by name, at most limit of them.
func (s *Store) List(ctx context.Context, limit int) ([]User, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT id, name, email FROM users ORDER BY name LIMIT $1`, limit)
	if err != nil {
		return nil, fmt.Errorf("list users: %w", err)
	}
	defer rows.Close()
	var out []User
	for rows.Next() {
		var u User
		if err := rows.Scan(&u.ID, &u.Name, &u.Email); err != nil {
			return nil, err
		}
		out = append(out, u)
	}
	return out, rows.Err()
}
"#;

    const README: &str = r#"---
title: Snap
tags: [llm, search]
---

# Snap

Local semantic code search.

## Install

```bash
# build from source
make build
```

Then put `./target/release/snap` on your PATH.

## Usage

Run it from the repo root:

    snap grep "where do we retry"

Options
-------

* `-g` restricts files
* `--hidden` includes dotfiles
"#;

    /// (start, end, symbol) of each chunk
    fn spans(chunks: &[Chunk]) -> Vec<(usize, usize, Option<&str>)> {
        chunks
            .iter()
            .map(|c| (c.start, c.end, c.symbol.as_deref()))
            .collect()
    }

    /// `sig {`, n body lines, `}`: a declaration of n + 2 lines
    fn func(sig: &str, n: usize) -> String {
        let body: String = (1..=n).map(|i| format!("    step_{i}(ctx);\n")).collect();
        format!("{sig} {{\n{body}}}\n")
    }

    /// `split`, and everything it promises, checked against the source lines
    fn check(path: &str, text: &str) -> Vec<Chunk> {
        let chunks = split(path, text);
        assert_eq!(chunks, split(path, text), "same text, same chunks");
        let lines: Vec<&str> = text.lines().collect();
        let mut hits = vec![0; lines.len()];
        let mut prev_end = 0;
        for c in &chunks {
            assert_eq!(c.path, path);
            assert!(prev_end < c.start && c.start <= c.end, "in order: {c:?}");
            assert!(c.end <= lines.len(), "in bounds: {c:?}");
            prev_end = c.end;
            assert!(!is_blank(lines[c.start - 1]), "starts on text: {c:?}");
            assert!(!is_blank(lines[c.end - 1]), "ends on text: {c:?}");
            assert_eq!(c.text, lines[c.start - 1..c.end].join("\n"), "{c:?}");
            assert!(c.end - c.start < MAX_LINES, "lines: {c:?}");
            assert!(
                c.start == c.end || c.text.chars().count() <= MAX_CHARS,
                "chars: {c:?}"
            );
            hits[c.start - 1..c.end].iter_mut().for_each(|h| *h += 1);
        }
        for (l, h) in lines.iter().zip(hits) {
            assert!(is_blank(l) || h == 1, "{l:?} is in {h} chunks");
        }
        chunks
    }

    /// 1-based first line of each code unit
    fn unit_starts(text: &str) -> Vec<usize> {
        let lines: Vec<&str> = text.lines().collect();
        let heads = vec![None; lines.len()];
        let units = units(&lines, &heads, false);
        units.iter().map(|u| u[0].0 + 1).collect()
    }

    #[test]
    fn rust_chunks() {
        let c = check("src/pool.rs", RUST);
        assert_eq!(spans(&c), [(1, 46, Some("Pool")), (48, 78, Some("warm"))]);
        // whole declarations: the impls stay in one chunk, the tests ride with `warm`
        assert!(c[0].text.ends_with("        pool\n    }\n}"));
        assert!(c[1].text.starts_with("/// Open `n` connections"));
        assert!(c[1].text.contains("mod tests {"));
    }

    #[test]
    fn python_chunks() {
        let c = check("retry.py", PYTHON);
        // blank lines inside `retry` (indented blocks) don't end it
        assert_eq!(
            spans(&c),
            [(1, 44, Some("retry")), (47, 62, Some("Budget"))]
        );
        assert!(c[0]
            .text
            .ends_with("await asyncio.sleep(backoff.next_delay())"));
        assert!(c[1].text.starts_with("class Budget:"));
    }

    #[test]
    fn go_chunks() {
        let c = check("store.go", GO);
        assert_eq!(spans(&c), [(1, 47, Some("Store")), (49, 65, Some("List"))]);
        assert!(c[1].text.starts_with("// List returns users"));
    }

    #[test]
    fn declared_names() {
        let named = [
            ("pub fn new(limit: usize) -> Self {", "new"),
            ("    pub(crate) async fn acquire(&self) {", "acquire"),
            ("pub unsafe extern \"C\" fn callback(x: i32) {", "callback"),
            ("pub const fn zero() -> u32 {", "zero"),
            ("pub const MAX: usize = 3;", "MAX"),
            ("static mut COUNTER: u32 = 0;", "COUNTER"),
            ("pub struct Pool {", "Pool"),
            ("pub enum Kind {", "Kind"),
            ("pub trait Backend: Send {", "Backend"),
            ("mod tests {", "tests"),
            ("type Result<T> = std::result::Result<T, Error>;", "Result"),
            ("macro_rules! hashmap {", "hashmap"),
            ("impl Pool {", "Pool"),
            ("impl<T: Into<String>> Pool<T> {", "Pool"),
            ("impl<T> fmt::Display for Wrapper<T> {", "Wrapper"),
            ("impl<'a> Iterator for &'a mut Cursor<'a> {", "Cursor"),
            ("unsafe impl Send for Pool {}", "Pool"),
            ("def parse(self, text):", "parse"),
            ("async def fetch(url):", "fetch"),
            ("class Backoff(Base):", "Backoff"),
            ("function render(props) {", "render"),
            ("function* ids() {", "ids"),
            (
                "export default async function handler(req, res) {",
                "handler",
            ),
            ("export const useThing = () => {", "useThing"),
            ("const limit = 10;", "limit"),
            ("export abstract class Shape {", "Shape"),
            ("export interface Props {", "Props"),
            ("export type Id = string;", "Id"),
            ("export enum Color {", "Color"),
            ("func main() {", "main"),
            (
                "func (s *Store) Get(ctx context.Context) (*User, error) {",
                "Get",
            ),
            ("func (Store) Close() error {", "Close"),
            ("type Store struct {", "Store"),
            ("public final class Parser {", "Parser"),
            ("    fun parse(text: String) {", "parse"),
            ("def self.build(opts)", "build"),
        ];
        for (line, want) in named {
            assert_eq!(declared(line).map(|d| d.0), Some(want), "{line}");
        }
        let unnamed = [
            "export default function () {",
            "    return fn(x);",
            "// fn commented() {}",
            "use std::fmt;",
            "#[derive(Debug)]",
            "classify(x)",
            "typeof x",
            "export PATH=/usr/bin",
            "default: break;",
        ];
        for line in unnamed {
            assert_eq!(declared(line), None, "{line}");
        }
    }

    #[test]
    fn items_are_not_bindings() {
        let bindings = [
            "pub const MAX: usize = 3;",
            "static mut COUNTER: u32 = 0;",
            "    let total = 0;",
            "var y = 1",
            "const limit = 10;",
        ];
        for line in bindings {
            assert_eq!(declared(line).map(|d| d.1), Some(false), "{line}");
        }
        let items = [
            "pub fn new() {",
            "pub const fn zero() -> u32 {",
            "impl Pool {",
            "impl<T> Tr for Pool<T> {",
            "macro_rules! hashmap {",
            "func (s *Store) Get() {",
            "class Backoff:",
            "type Id = u32;",
            "mod tests {",
        ];
        for line in items {
            assert_eq!(declared(line).map(|d| d.1), Some(true), "{line}");
        }
    }

    #[test]
    fn units_follow_indentation() {
        // a blank line inside a body, and the `}` after it, stay with the function
        assert_eq!(
            unit_starts("fn a() {\n    x();\n\n    y();\n\n}\n\nfn b() {}\n"),
            [1, 8]
        );
        // closing tokens continue the unit before them; words that only start like one don't
        for close in [
            "}", ") {", "]", "end", "end.", "fi", "done", "esac", "</div>", "} else {",
        ] {
            assert_eq!(
                unit_starts(&format!("open\n  body\n\n{close}\n")),
                [1],
                "{close}"
            );
        }
        for word in [
            "endpoint()",
            "fib(3)",
            "done_cb()",
            "esac_x",
            "<b>x</b>",
            "(a, b)",
        ] {
            assert_eq!(
                unit_starts(&format!("open\n  body\n\n{word}\n")),
                [1, 4],
                "{word}"
            );
        }
        // base indent is the file's least, a tab counts 4, the first block always opens
        assert_eq!(unit_starts("    a()\n\n    b()\n\n        c()\n"), [1, 3]);
        assert_eq!(unit_starts("    a\n\n\tb\n\n\t\tc\n"), [1, 3]);
        assert_eq!(unit_starts("    deep()\n\nshallow()\n"), [1, 3]);
        assert_eq!(unit_starts("}\n\nfn a() {}\n"), [1, 3]);
    }

    #[test]
    fn small_units_pack_big_ones_split() {
        // two 32-line functions can't share a chunk (65 lines with the gap), small ones ride along
        let text = [
            func("fn a()", 30),
            func("fn b()", 30),
            func("fn c()", 3),
            func("fn d()", 3),
        ]
        .join("\n");
        let c = check("a.rs", &text);
        assert_eq!(spans(&c), [(1, 32, Some("a")), (34, 77, Some("b"))]);
        // a unit over the budget is cut at line boundaries when it has no blank line to cut at
        let c = check("big.rs", &func("fn big()", 200));
        let pieces = [(1, 60), (61, 120), (121, 180), (181, 202)];
        assert_eq!(spans(&c), pieces.map(|(s, e)| (s, e, Some("big"))));
        // and at its blank lines when it has them: a cut never lands inside a block
        let body: String = (1..=200)
            .map(|i| {
                format!(
                    "    step_{i}(ctx);\n{}",
                    if i % 10 == 0 { "\n" } else { "" }
                )
            })
            .collect();
        let text = format!("fn big() {{\n{body}}}\n");
        let c = check("big.rs", &text);
        assert!(c.len() >= 4 && c.iter().all(|c| c.symbol.as_deref() == Some("big")));
        let lines: Vec<&str> = text.lines().collect();
        assert!(
            c[..c.len() - 1].iter().all(|c| is_blank(lines[c.end])),
            "cuts at blank lines"
        );
        // the tail of a cut block packs with the blocks after it
        let text = format!(
            "fn m() {{\n{}\n    tail();\n}}\n",
            "    step();\n".repeat(100)
        );
        assert_eq!(
            spans(&check("m.rs", &text)),
            [(1, 60, Some("m")), (61, 104, Some("m"))]
        );
    }

    #[test]
    fn pieces_of_an_impl_take_their_methods_names() {
        // six 24-line methods: the impl is cut between them, two to a piece, and a
        // piece is named for the first item in it (the first holds the `impl` line)
        let method = |name: &str| {
            let body = "        step(self);\n".repeat(22);
            format!("    pub fn {name}(&self) {{\n{body}    }}\n")
        };
        let methods = ["new", "open", "plan", "wave", "evict", "drop"].map(method);
        let c = check("kv.rs", &format!("impl Kv {{\n{}}}\n", methods.join("\n")));
        let want = [(1, 50, "Kv"), (52, 100, "plan"), (102, 151, "evict")];
        assert_eq!(spans(&c), want.map(|(s, e, n)| (s, e, Some(n))));
        // the same in Python
        let method = |name: &str| {
            format!(
                "    def {name}(self):\n{}",
                "        self.step()\n".repeat(22)
            )
        };
        let methods = ["load", "save", "plan", "wave"].map(method);
        let c = check(
            "client.py",
            &format!("class Client:\n{}", methods.join("\n")),
        );
        assert_eq!(spans(&c), [(1, 48, Some("Client")), (50, 96, Some("plan"))]);
    }

    #[test]
    fn pieces_of_a_function_keep_its_name_despite_locals() {
        // `let`, `const` and `static` inside a body are locals, not items: later pieces
        // full of them are still named for the function
        let body: String = (1..=200)
            .map(|i| match i % 3 {
                0 => format!("    const C{i}: u32 = {i};\n"),
                1 => format!("    let x{i} = {i};\n"),
                _ => format!("    static S{i}: u32 = {i};\n"),
            })
            .collect();
        let c = check("big.rs", &format!("fn big() {{\n{body}}}\n"));
        let pieces = [(1, 60), (61, 120), (121, 180), (181, 202)];
        assert_eq!(spans(&c), pieces.map(|(s, e)| (s, e, Some("big"))));
    }

    #[test]
    fn a_piece_inside_a_long_method_takes_the_methods_name() {
        // `wave` runs over three pieces: the first declares it, the other two only
        // continue its body and say `wave`, not `Kv`; `tail` opens a piece of its own
        let steps = |n: usize| "        step(self);\n".repeat(n);
        let text = format!(
            "impl Kv {{\n    pub fn short(&self) {{\n{}    }}\n\n    pub fn wave(&self) {{\n{}    }}\n\n    pub fn tail(&self) {{\n{}    }}\n}}\n",
            steps(1),
            steps(130),
            steps(50)
        );
        let c = check("kv.rs", &text);
        let want = [
            (1, 4, "Kv"),
            (6, 65, "wave"),
            (66, 125, "wave"),
            (126, 137, "wave"),
            (139, 191, "tail"),
        ];
        assert_eq!(spans(&c), want.map(|(s, e, n)| (s, e, Some(n))));
    }

    #[test]
    fn a_nested_item_names_only_the_pieces_inside_it() {
        // a helper declared in the second piece of a big function names that piece and
        // the one still in its body; back in the function's own body, pieces say `big`
        let text = format!(
            "fn big() {{\n{}    fn helper() {{\n{}    }}\n{}}}\n",
            "    step();\n".repeat(70),
            "        inner();\n".repeat(60),
            "    step();\n".repeat(70)
        );
        let c = check("big.rs", &text);
        let want = [
            (1, 60, "big"),
            (61, 120, "helper"),
            (121, 180, "helper"),
            (181, 204, "big"),
        ];
        assert_eq!(spans(&c), want.map(|(s, e, n)| (s, e, Some(n))));
    }

    #[test]
    fn a_long_line_stands_alone() {
        let long = "x".repeat(2 * MAX_CHARS);
        let c = check("a.rs", &format!("fn a() {{\n    let s = \"{long}\";\n}}\n"));
        assert_eq!(
            spans(&c),
            [(1, 1, Some("a")), (2, 2, Some("a")), (3, 3, Some("a"))]
        );
        let c = check("a.rs", &format!("fn a() {{}}\n\n{long}\n\nfn b() {{}}\n"));
        assert_eq!(
            spans(&c),
            [(1, 1, Some("a")), (3, 3, None), (5, 5, Some("b"))]
        );
    }

    #[test]
    fn budget_counts_chars() {
        let rows = |n: usize| vec!["世界".repeat(15); n].join("\n");
        // 50 rows of 30 chars: 1549 chars (4500 bytes) is one chunk, 54 rows are not
        assert_eq!(spans(&check("a.txt", &rows(50))), [(1, 50, None)]);
        assert_eq!(
            spans(&check("a.txt", &rows(54))),
            [(1, 51, None), (52, 54, None)]
        );
    }

    #[test]
    fn blank_files_and_line_endings() {
        for text in ["", "\n", "\n\n", "  \n\t\n   ", "\r\n\r\n"] {
            assert!(split("a.rs", text).is_empty(), "{text:?}");
        }
        let c = check("a.rs", "fn a() {\r\n    x();\r\n}\r\n\r\nfn b() {}");
        assert_eq!(spans(&c), [(1, 5, Some("a"))]);
        assert_eq!(c[0].text, "fn a() {\n    x();\n}\n\nfn b() {}");
    }

    #[test]
    fn large_file_invariants() {
        let mut text = String::new();
        for i in 0..120 {
            text += &func(&format!("fn f{i}()"), i % 17);
            text += if i % 3 == 0 { "\n" } else { "\n\n" };
        }
        text += &func("fn big()", 200);
        text += "\n";
        text += &func("fn after()", 3);
        let c = check("big.rs", &text);
        let big = c
            .iter()
            .filter(|c| c.symbol.as_deref() == Some("big"))
            .count();
        assert_eq!(
            big, 4,
            "the 202-line function is four pieces, each saying `big`"
        );
        assert_eq!(c.last().and_then(|c| c.symbol.as_deref()), Some("after"));
    }

    #[test]
    fn fuzz_invariants() {
        let long = "y".repeat(2 * MAX_CHARS);
        #[rustfmt::skip]
        let bits = [
            "", "", "   ", "fn f() {", "    x += 1;", "\t\tdeep();", "}", ")", "end", "</div>",
            "# Heading", "## Sub", "Title", "=====", "-----", "```", "~~~", "* star", "- bullet",
            "---", "= Adoc", "// héllo wörld 世界", "def g(self):", "class K:", &long,
        ];
        let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = |n: usize| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as usize % n
        };
        for _ in 0..300 {
            let n = next(150);
            let eol = if next(4) == 0 { "\r\n" } else { "\n" };
            let lines: Vec<&str> = (0..n).map(|_| bits[next(bits.len())]).collect();
            let text = lines.join(eol);
            for path in [
                "a.rs", "a.py", "a.md", "a.rst", "a.org", "a.adoc", "a.txt", "Makefile",
            ] {
                check(path, &text);
            }
        }
    }

    #[test]
    fn prose_sections() {
        // front matter and fenced `# comments` are not headings; the setext heading
        // keeps its underline; the 4-line front matter + title join the first section,
        // which then closes (8 lines), as does every section after it
        let c = check("README.md", README);
        let want = [
            (1, 8, "Snap"),
            (10, 17, "Install"),
            (19, 23, "Usage"),
            (25, 29, "Options"),
        ];
        assert_eq!(spans(&c), want.map(|(s, e, h)| (s, e, Some(h))));
        assert!(c[3].text.starts_with("Options\n-------\n"));
    }

    #[test]
    fn prose_short_chunks_take_the_next_heading() {
        // 4 lines held: the next heading joins them; 5 lines: it opens a chunk
        let four = "# T\na\nb\nc\n## S\nx\n";
        let five = "# T\na\nb\nc\nd\n## S\nx\n";
        assert_eq!(spans(&check("a.md", four)), [(1, 6, Some("T"))]);
        assert_eq!(
            spans(&check("a.md", five)),
            [(1, 5, Some("T")), (6, 7, Some("S"))]
        );
    }

    #[test]
    fn prose_setext_and_rst() {
        let text = "=====\nGuide\n=====\n\nIntro.\n\nSetup\n-----\n\none\ntwo\nthree\nfour\n\nRun\n---\n\ngo\n";
        for path in ["guide.rst", "guide.txt", "guide.md"] {
            let c = check(path, text);
            let want = [(1, 5, "Guide"), (7, 13, "Setup"), (15, 18, "Run")];
            assert_eq!(spans(&c), want.map(|(s, e, h)| (s, e, Some(h))), "{path}");
            assert!(
                c[0].text.starts_with("=====\nGuide\n=====\n"),
                "the overline opens it"
            );
            assert!(
                c[1].text.starts_with("Setup\n-----\n"),
                "the underline stays"
            );
        }
    }

    #[test]
    fn prose_org() {
        let text = "#+TITLE: Notes\n# a comment\n\n* Build\n  - make\n  - make test\n  * a list item\n** Targets\n   text\n   more\n   even more\n* Release\n  tag it\n";
        let c = check("notes.org", text);
        assert_eq!(spans(&c), [(1, 7, Some("Build")), (8, 13, Some("Targets"))]);
    }

    #[test]
    fn prose_big_sections_keep_their_heading() {
        // 30 two-line paragraphs: the section is cut between paragraphs, every piece
        // names it (a line that reads like `fn name` is text here), and the next
        // section still opens a chunk of its own
        let paras: String = (0..30)
            .map(|i| format!("fn para_{i}() one\npara {i} two\n\n"))
            .collect();
        let text = format!("## Big\n\n{paras}## Next\n\n1\n2\n3\n4\n");
        let c = check("a.md", &text);
        let (big, next) = c.split_at(c.len() - 1);
        assert!(big.len() >= 2 && big.iter().all(|c| c.symbol.as_deref() == Some("Big")));
        assert_eq!(next[0].symbol.as_deref(), Some("Next"));
        // prose with no heading at all is just paragraphs
        let c = check("a.md", &paras);
        assert!(c.len() >= 2 && c.iter().all(|c| c.symbol.is_none()));
    }

    #[test]
    fn prose_rules_stay_in_prose() {
        let py = |name: &str| {
            format!("# Part {name}\ndef {name}():\n    a()\n    b()\n    c()\n    d()\n")
        };
        let text = format!("{}\n{}", py("one"), py("two"));
        let c = check("a.py", &text);
        assert_eq!(spans(&c), [(1, 13, Some("one"))]);
        let c = check("a.md", &text);
        assert_eq!(
            spans(&c),
            [(1, 6, Some("Part one")), (8, 13, Some("Part two"))]
        );
    }

    #[test]
    fn heading_syntaxes() {
        fn heads<'a>(ext: &str, text: &'a str) -> Vec<&'a str> {
            let lines: Vec<&str> = text.lines().collect();
            headings(ext, &lines).into_iter().flatten().collect()
        }
        let md = "# One\n##Two\n## Two\n####### Seven\n  # Indented\n#\n#hash\n##### Five  \n";
        assert_eq!(heads("md", md), ["One", "Two", "Five"]);
        // setext wants three or more rule characters, directly under a text line
        assert_eq!(heads("md", "A\n==\nB\n===\n\n---\nC\n\n---\nD\n"), ["B"]);
        assert_eq!(heads("md", "---\ntitle: x\n---\n# Doc\n"), ["Doc"]);
        assert_eq!(heads("md", "```\n# no\nTitle\n---\n```\n# yes\n"), ["yes"]);
        // `~~~` underlines an rst title, it doesn't open a fence
        assert_eq!(heads("rst", "Title\n~~~\ntext\n\nNext\n----\n"), ["Next"]);
        assert_eq!(
            heads(
                "adoc",
                "= Title\n== Section\n# no\nText\n----\ncode\n----\n"
            ),
            ["Title", "Section"]
        );
        assert_eq!(
            heads("org", "* One\n** Two\n*bold*\n# c\nText\n-----\n"),
            ["One", "Two"]
        );
        assert_eq!(heads("txt", "# One\nTwo\n---\n"), ["One", "Two"]);
        for ext in ["md", "markdown", "mdx", "rst", "adoc", "txt", "org"] {
            assert_eq!(prose_ext(&format!("a.{ext}")).as_deref(), Some(ext));
        }
        assert_eq!(prose_ext("docs/A.MD").as_deref(), Some("md"));
        assert_eq!(prose_ext("main.rs"), None);
        assert_eq!(prose_ext("Makefile"), None);
    }

    /// A scratch tree under the system temp dir, removed on drop.
    struct Tmp(PathBuf);

    impl Tmp {
        fn new(name: &str) -> Tmp {
            let dir = std::env::temp_dir()
                .join(format!("snap-test-corpus-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Tmp(dir)
        }

        fn put(&self, rel: &str, body: impl AsRef<[u8]>) {
            let path = self.0.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, body).unwrap();
        }

        fn rels(&self, opts: &WalkOpts) -> Vec<String> {
            let files = files(&self.0, opts).unwrap();
            files.into_iter().map(|(rel, _)| rel).collect()
        }
    }

    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn opts(hidden: bool, no_ignore: bool, globs: &[&str]) -> WalkOpts {
        let globs = globs.iter().map(|g| g.to_string()).collect();
        WalkOpts {
            hidden,
            no_ignore,
            globs,
        }
    }

    fn tree(name: &str) -> Tmp {
        let t = Tmp::new(name);
        t.put(".gitignore", "build/\n");
        t.put(".ignore", "ignored.txt\n");
        t.put(".rgignore", "scratch.txt\n");
        t.put(".git/config", "[core]\n");
        t.put(".hidden/inner.txt", "in a hidden dir\n");
        t.put(".dotfile", "hidden\n");
        t.put("src/main.rs", "fn main() {}\n");
        t.put("src/lib.rs", "pub fn lib() {}\n");
        t.put("src/util/mod.rs", "pub fn util() {}\n");
        t.put("vendor/dep.rs", "pub fn dep() {}\n");
        t.put("docs/guide.md", "# Guide\n\nHow to.\n");
        t.put("web/app.js", "export const app = 1;\n");
        t.put("web/app.min.js", "var a=1;\n");
        t.put("web/app.js.map", "{}\n");
        t.put("web/yarn.lock", "# lock\n");
        t.put("data.json", "{}\n");
        t.put("package-lock.json", "{}\n");
        t.put("Cargo.lock", "# lock\n");
        t.put("notes.txt", "plain\n");
        t.put("ignored.txt", "named in .ignore\n");
        t.put("scratch.txt", "named in .rgignore\n");
        t.put("build/out.rs", "a directory named in .gitignore\n");
        t.put("blob.dat", [0u8, 1, 2, 0xff]);
        t
    }

    const LISTED: [&str; 9] = [
        "blob.dat",
        "data.json",
        "docs/guide.md",
        "notes.txt",
        "src/lib.rs",
        "src/main.rs",
        "src/util/mod.rs",
        "vendor/dep.rs",
        "web/app.js",
    ];

    #[test]
    fn walk_ignore_files_and_hidden() {
        // .gitignore, .ignore and .rgignore, dotfiles, `.git`, lockfiles and minified
        // files are out; paths are `/`-separated and sorted
        let t = tree("walk");
        assert_eq!(t.rels(&WalkOpts::default()), LISTED);
        for (rel, path) in files(&t.0, &WalkOpts::default()).unwrap() {
            assert_eq!(path, t.0.join(rel));
        }
        // --hidden adds dotfiles, never `.git`
        let hidden = t.rels(&opts(true, false, &[]));
        let dots = [
            ".dotfile",
            ".gitignore",
            ".hidden/inner.txt",
            ".ignore",
            ".rgignore",
        ];
        assert_eq!(hidden[..5], dots);
        assert_eq!(hidden[5..], LISTED);
        // --no-ignore lists what the ignore files hid, but not dotfiles or noise
        let all = t.rels(&opts(false, true, &[]));
        let extra = ["build/out.rs", "ignored.txt", "scratch.txt"];
        let mut want: Vec<&str> = LISTED.iter().chain(&extra).copied().collect();
        want.sort_unstable();
        assert_eq!(all, want);
    }

    #[test]
    fn walk_globs() {
        let t = tree("globs");
        let rels = |globs: &[&str]| t.rels(&opts(false, false, globs));
        let rs = [
            "src/lib.rs",
            "src/main.rs",
            "src/util/mod.rs",
            "vendor/dep.rs",
        ];
        assert_eq!(rels(&["*.rs"]), rs);
        assert_eq!(rels(&["*.rs", "!vendor/**"]), rs[..3]);
        assert_eq!(rels(&["src/**"]), rs[..3]);
        assert_eq!(
            rels(&["!src/**", "!vendor/**", "!docs/**"]),
            ["blob.dat", "data.json", "notes.txt", "web/app.js"]
        );
    }

    #[test]
    fn walk_noise_needs_naming() {
        let t = tree("noise");
        let rels = |globs: &[&str]| t.rels(&opts(false, false, globs));
        // wildcards never bring lockfiles or minified files back by accident
        assert_eq!(rels(&["*.json"]), ["data.json"]);
        assert_eq!(rels(&["*.js"]), ["web/app.js"]);
        assert!(rels(&["*.lock"]).is_empty());
        // a glob that names one does
        assert_eq!(rels(&["package-lock.json"]), ["package-lock.json"]);
        assert_eq!(
            rels(&["*.json", "package-lock.json"]),
            ["data.json", "package-lock.json"]
        );
        assert_eq!(rels(&["**/Cargo.lock"]), ["Cargo.lock"]);
        assert_eq!(rels(&["yarn.lock"]), ["web/yarn.lock"]);
        assert_eq!(rels(&["*.min.js"]), ["web/app.min.js"]);
        assert_eq!(rels(&["web/app.min.js"]), ["web/app.min.js"]);
        assert_eq!(rels(&["*.map"]), ["web/app.js.map"]);
    }

    #[cfg(unix)]
    #[test]
    fn walk_relative_root() {
        // the same tree through a root spelled relative to the cwd, as `snap grep` gets it
        let t = tree("relative");
        let up = "../".repeat(std::env::current_dir().unwrap().components().count() - 1);
        let root = PathBuf::from(format!("{up}{}", t.0.strip_prefix("/").unwrap().display()));
        let listed = files(&root, &WalkOpts::default()).unwrap();
        assert_eq!(
            listed.iter().map(|(r, _)| r.as_str()).collect::<Vec<_>>(),
            LISTED
        );
        assert!(listed.iter().all(|(rel, path)| *path == root.join(rel)));
        let globbed = files(&root, &opts(false, false, &["*.rs", "!vendor/**"])).unwrap();
        let rels: Vec<&str> = globbed.iter().map(|(r, _)| r.as_str()).collect();
        assert_eq!(rels, ["src/lib.rs", "src/main.rs", "src/util/mod.rs"]);
    }

    #[cfg(unix)]
    #[test]
    fn walk_skips_symlinks() {
        let t = Tmp::new("links");
        t.put("real.rs", "fn a() {}\n");
        std::os::unix::fs::symlink(t.0.join("real.rs"), t.0.join("link.rs")).unwrap();
        std::os::unix::fs::symlink(&t.0, t.0.join("loop")).unwrap();
        assert_eq!(t.rels(&WalkOpts::default()), ["real.rs"]);
    }

    #[test]
    fn walk_errors_are_clear() {
        let t = Tmp::new("errors");
        t.put("f.rs", "fn f() {}\n");
        let err = files(&t.0.join("f.rs"), &WalkOpts::default()).unwrap_err();
        assert!(err.to_string().contains("not a directory"), "{err}");
        assert!(files(&t.0.join("nope"), &WalkOpts::default()).is_err());
        assert!(files(&t.0, &opts(false, false, &["[z-a]"])).is_err());
    }

    #[test]
    fn corpus_reads_what_is_readable() {
        let t = tree("corpus");
        let c = corpus(&t.0, &WalkOpts::default()).unwrap();
        // blob.dat is listed by `files` and dropped by `read_text`; one chunk per tiny file
        let paths: Vec<&str> = c.iter().map(|c| c.path.as_str()).collect();
        let readable: Vec<&str> = LISTED
            .iter()
            .copied()
            .filter(|p| *p != "blob.dat")
            .collect();
        assert_eq!(paths, readable);
        let symbol = |p: &str| {
            c.iter()
                .find(|c| c.path == p)
                .and_then(|c| c.symbol.as_deref())
        };
        assert_eq!(symbol("src/lib.rs"), Some("lib"));
        assert_eq!(symbol("web/app.js"), Some("app"));
        assert_eq!(symbol("docs/guide.md"), Some("Guide"));
        assert_eq!(symbol("notes.txt"), None);
    }

    #[test]
    fn read_text_skips_what_is_not_code() {
        let t = Tmp::new("read");
        t.put("ok.rs", "fn main() {}\n");
        t.put("bom.rs", "\u{feff}fn main() {}\n");
        t.put("empty.rs", "");
        t.put("nul.dat", "fn main() {}\0\n");
        t.put("latin1.txt", b"caf\xe9\n");
        t.put("at_limit.txt", "x\n".repeat(MAX_FILE_BYTES as usize / 2));
        t.put("over.txt", "x\n".repeat(MAX_FILE_BYTES as usize / 2 + 1));
        t.put("late_nul.txt", format!("{}\0", "line\n".repeat(2000)));
        let read = |name: &str| read_text(&t.0.join(name));
        assert_eq!(read("ok.rs").as_deref(), Some("fn main() {}\n"));
        assert_eq!(
            read("bom.rs").as_deref(),
            Some("fn main() {}\n"),
            "BOM stripped"
        );
        assert_eq!(read("empty.rs").as_deref(), Some(""));
        assert_eq!(read("nul.dat"), None, "binary");
        assert_eq!(read("latin1.txt"), None, "not UTF-8");
        assert!(read("at_limit.txt").is_some());
        assert_eq!(read("over.txt"), None, "over MAX_FILE_BYTES");
        assert!(
            read("late_nul.txt").is_some(),
            "only the first 8 KiB are sniffed"
        );
        assert_eq!(read("missing.rs"), None);
    }

    #[test]
    fn read_text_minified() {
        let t = Tmp::new("minified");
        let rows = |n: usize, width: usize| format!("{}\n", "x".repeat(width)).repeat(n);
        t.put("bundle.js", rows(3, 3000));
        t.put("just_over.txt", rows(1, MINIFIED_AVG + 1));
        t.put("at_limit.txt", rows(2, MINIFIED_AVG));
        // blank lines don't dilute the average
        t.put("sparse.js", format!("{}\n\n\n\n", "x".repeat(2000)));
        // prose keeps a paragraph on a line: 600 chars each is not minified
        let para = format!("{}\n\n", "word ".repeat(120));
        t.put("README.md", format!("# Guide\n\n{}", para.repeat(10)));
        let read = |name: &str| read_text(&t.0.join(name));
        assert_eq!(read("bundle.js"), None);
        assert_eq!(read("just_over.txt"), None);
        assert!(read("at_limit.txt").is_some());
        assert_eq!(read("sparse.js"), None);
        // and such a file is chunked a couple of paragraphs at a time, each saying "Guide"
        let text = read("README.md").expect("600-char paragraphs are kept");
        let c = check("README.md", &text);
        let want = [(1, 5), (7, 9), (11, 13), (15, 17), (19, 21)];
        assert_eq!(spans(&c), want.map(|(s, e)| (s, e, Some("Guide"))));
    }

    fn chunk(text: &str) -> Chunk {
        Chunk {
            path: "f".into(),
            start: 1,
            end: 1,
            symbol: None,
            text: text.into(),
        }
    }

    #[test]
    fn grep_smart_case_and_anchors() {
        let cs = [
            chunk("fn Parse() {}"),
            chunk("fn parse() {}"),
            chunk("PARSE"),
        ];
        assert_eq!(grep(&cs, "parse").unwrap(), [true, true, true]);
        assert_eq!(grep(&cs, "Parse").unwrap(), [true, false, false]);
        // an escape isn't an uppercase literal: `\S` leaves the match case-insensitive
        assert_eq!(grep(&[chunk("FOOx")], r"foo\S").unwrap(), [true]);
        assert_eq!(grep(&[chunk("FOOx")], r"Foo\S").unwrap(), [false]);
        // `^` and `$` are per line, as in ripgrep
        let c = [chunk("use a;\nfn main() {}\n// done")];
        for (pattern, want) in [
            ("^fn main", true),
            ("^main", false),
            (";$", true),
            ("^// done$", true),
            ("a;\nfn", true),
        ] {
            assert_eq!(grep(&c, pattern).unwrap(), [want], "{pattern}");
        }
        assert!(grep(&[], "x").unwrap().is_empty());
    }

    #[test]
    fn grep_invalid_pattern() {
        let err = grep(&[chunk("x")], "fn(").unwrap_err();
        assert!(err.to_string().starts_with("invalid pattern"), "{err}");
    }
}
