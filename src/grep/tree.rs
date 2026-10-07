//! Tree descent: the half of `snap grep`'s recall that does not depend on the
//! words of the question, run beside BM25 and united with its candidates
//! (`mod.rs`). Folders, files and groups of a file's chunks become
//! previews (names, declared symbols, first lines), the model answers the
//! same yes/no probe on them level by level, and only the best `beam`
//! branches of each kind are opened. A big file is not terminal: it is split
//! into contiguous groups and the descent keeps going inside it, so one file
//! cannot flood the candidates with every chunk it has. A paraphrase that
//! shares no word with the code can still be reached, because the model, not
//! a word match, decides which branch is worth opening.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use anyhow::{ensure, Result};

use super::corpus::{prose_ext, Chunk};
use super::lexical::humps;

/// A preview stays small: the model reads one per node of the frontier.
const PREVIEW_CHARS: usize = 1200;
/// Symbols a folder preview borrows from the files directly inside it.
const FOLDER_SYMBOLS: usize = 8;
/// Groups a file or group splits into; at most this many chunks is terminal.
const FANOUT: usize = 4;
/// Lines of a file's leading comment block that its preview keeps.
const DOC_LINES: usize = 6;

#[derive(Default)]
struct Folder {
    dirs: BTreeSet<String>,
    /// indices into `files`, in walk order
    files: Vec<usize>,
}

#[derive(Clone)]
enum Node {
    Dir(String),
    File(usize),
    /// contiguous chunks of one file
    Group(Range<usize>),
}

/// Chunk indices in tree order: the chunks of the terminal nodes the descent
/// reached (a file or group of at most `FANOUT` chunks), best node first,
/// each node's chunks in line order. `files` holds the chunks of each file,
/// adjacent as the walk emits them (`Haystack::files`). `score` reads a batch
/// of previews (a `Chunk` whose `path` is the folder or file and whose `text`
/// is the preview, `symbol` None; `start`/`end` 1 for a folder or file, the
/// lines covered for a group) and returns P(yes) for each, in order.
pub fn descend(
    chunks: &[Chunk],
    files: &[Range<usize>],
    beam: usize,
    mut score: impl FnMut(&[Chunk]) -> Result<Vec<f64>>,
) -> Result<Vec<usize>> {
    let path = |f: usize| chunks[files[f].start].path.as_str();
    let mut folders: BTreeMap<String, Folder> = BTreeMap::new();
    for f in (0..files.len()).filter(|&f| !files[f].is_empty()) {
        let parts: Vec<&str> = path(f).split('/').collect();
        let mut parent = String::new();
        for i in 1..parts.len() {
            let dir = parts[..i].join("/");
            folders.entry(parent).or_default().dirs.insert(dir.clone());
            parent = dir;
        }
        folders.entry(parent).or_default().files.push(f);
    }
    let children = |dir: &str| -> Vec<Node> {
        let Some(d) = folders.get(dir) else {
            return Vec::new();
        };
        let dirs = d.dirs.iter().cloned().map(Node::Dir);
        dirs.chain(d.files.iter().copied().map(Node::File))
            .collect()
    };

    let mut frontier = if beam == 0 { Vec::new() } else { children("") };
    let mut reached: Vec<(Range<usize>, f64)> = Vec::new();
    while !frontier.is_empty() {
        let previews: Vec<Chunk> = frontier
            .iter()
            .map(|n| match n {
                Node::Dir(d) => folder_preview(d, &folders[d], chunks, files),
                Node::File(f) => file_preview(path(*f), &chunks[files[*f].clone()]),
                Node::Group(r) => group_preview(&chunks[r.clone()]),
            })
            .collect();
        let p = score(&previews)?;
        ensure!(
            p.len() == previews.len(),
            "score returned {} values for {} previews",
            p.len(),
            previews.len()
        );
        for (c, p) in previews.iter().zip(&p) {
            tracing::debug!("{p:.3} {}:{}-{}", c.path, c.start, c.end);
        }
        let (mut dirs, mut open_files, mut groups) = (Vec::new(), Vec::new(), Vec::new());
        for (n, p) in frontier.iter().zip(p) {
            match n {
                Node::Dir(d) => dirs.push((d, p)),
                Node::File(f) if files[*f].len() <= FANOUT => reached.push((files[*f].clone(), p)),
                Node::File(f) => open_files.push((*f, p)),
                Node::Group(r) if r.len() <= FANOUT => reached.push((r.clone(), p)),
                Node::Group(r) => groups.push(((r.start, r.end), p)),
            }
        }
        let mut next: Vec<Node> = best(dirs, beam).flat_map(|d| children(d)).collect();
        for f in best(open_files, beam) {
            next.extend(split(files[f].clone()).map(Node::Group));
        }
        for (start, end) in best(groups, beam) {
            next.extend(split(start..end).map(Node::Group));
        }
        frontier = next;
    }
    reached.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.start.cmp(&b.0.start)));
    Ok(reached.into_iter().flat_map(|(r, _)| r).collect())
}

/// The `beam` best keys by P, ties by key order.
fn best<K: Ord>(mut scored: Vec<(K, f64)>, beam: usize) -> impl Iterator<Item = K> {
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scored.into_iter().take(beam).map(|(k, _)| k)
}

/// `FANOUT` contiguous parts of near-equal size; `r` has more than `FANOUT`
/// chunks, so none is empty.
fn split(r: Range<usize>) -> impl Iterator<Item = Range<usize>> {
    let n = r.len();
    (0..FANOUT).map(move |i| r.start + i * n / FANOUT..r.start + (i + 1) * n / FANOUT)
}

fn preview(path: String, lines: impl IntoIterator<Item = String>) -> Chunk {
    let mut text = String::new();
    for line in lines {
        let line: String = line.chars().take(PREVIEW_CHARS).collect();
        let sep = usize::from(!text.is_empty());
        if text.chars().count() + sep + line.chars().count() > PREVIEW_CHARS {
            break;
        }
        text.push_str(&"\n".repeat(sep));
        text.push_str(&line);
    }
    Chunk {
        path,
        start: 1,
        end: 1,
        symbol: None,
        text,
    }
}

/// Distinct symbols in first-seen order.
fn symbols(chunks: &[Chunk]) -> impl Iterator<Item = &str> {
    let mut seen = BTreeSet::new();
    chunks
        .iter()
        .filter_map(|c| c.symbol.as_deref())
        .filter(move |s| seen.insert(*s))
}

/// What the model reads of `chunks`: the `lead` lines, an outline line per
/// chunk ("symbol words: its first comment line"), then the scrubbed bodies.
/// No line repeats, so the cap goes to meaning before it goes to echoes.
fn describe(chunks: &[Chunk], mut lines: Vec<String>) -> Vec<String> {
    let mut seen: BTreeSet<String> = lines.iter().cloned().collect();
    let bodies: Vec<Vec<Line>> = chunks.iter().map(|c| scrub(&c.path, &c.text)).collect();
    for (c, body) in chunks.iter().zip(&bodies) {
        let note = body.iter().find(|l| !l.code).map(|l| l.text.as_str());
        let note = note.filter(|n| seen.insert((*n).to_owned()));
        let name = c.symbol.as_deref().map(say);
        let line = [name.as_deref(), note].into_iter().flatten();
        let line = line
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(": ");
        if !line.is_empty() && seen.insert(line.clone()) {
            lines.push(line);
        }
    }
    let rest = bodies.into_iter().flatten().map(|l| l.text);
    lines.extend(rest.filter(|l| seen.insert(l.clone())));
    lines
}

/// The file's leading comment block, scrubbed, from its first chunk.
fn docs(first: &Chunk) -> Vec<String> {
    let lines = scrub(&first.path, &first.text).into_iter();
    lines
        .take_while(|l| !l.code)
        .take(DOC_LINES)
        .map(|l| l.text)
        .collect()
}

fn file_preview(path: &str, chunks: &[Chunk]) -> Chunk {
    preview(path.to_owned(), describe(chunks, docs(&chunks[0])))
}

/// The file path, the lines the group covers and its chunks, outlined and
/// then flattened, as far as the cap allows.
fn group_preview(chunks: &[Chunk]) -> Chunk {
    Chunk {
        start: chunks[0].start,
        end: chunks[chunks.len() - 1].end,
        ..preview(chunks[0].path.clone(), describe(chunks, Vec::new()))
    }
}

/// Subfolders by name, files by name and the first line of their module
/// docs, then a few of their symbols.
fn folder_preview(dir: &str, folder: &Folder, chunks: &[Chunk], files: &[Range<usize>]) -> Chunk {
    let name = |p: &str| p.rsplit('/').next().unwrap_or(p).to_owned();
    let mut lines: Vec<String> = folder.dirs.iter().map(|d| say(&name(d)) + "/").collect();
    let mut syms = Vec::new();
    for &f in &folder.files {
        let cs = &chunks[files[f].clone()];
        let file = name(&cs[0].path);
        let stem = file.rsplit_once('.').map_or(file.as_str(), |(s, _)| s);
        let head = say(if stem.is_empty() { &file } else { stem });
        lines.push(match docs(&cs[0]).first() {
            Some(d) => format!("{head}: {d}"),
            None => head,
        });
        syms.extend(symbols(cs).map(say));
    }
    let mut seen = BTreeSet::new();
    syms.retain(|s| seen.insert(s.clone()));
    lines.extend(syms.into_iter().take(FOLDER_SYMBOLS));
    preview(format!("{dir}/"), lines)
}

/// One scrubbed source line; `code` is whether it holds anything but comment.
struct Line {
    text: String,
    code: bool,
}

/// Words of code that say nothing about what it is for.
const NOISE: &[&str] = &[
    "fn",
    "pub",
    "let",
    "mut",
    "self",
    "impl",
    "use",
    "mod",
    "crate",
    "super",
    "return",
    "if",
    "else",
    "for",
    "while",
    "in",
    "match",
    "struct",
    "enum",
    "trait",
    "type",
    "const",
    "static",
    "where",
    "as",
    "ref",
    "move",
    "async",
    "await",
    "def",
    "class",
    "import",
    "from",
    "function",
    "var",
    "new",
    "loop",
    "break",
    "continue",
    "true",
    "false",
    "none",
    "null",
    "nil",
    "some",
    "ok",
    "err",
    "dyn",
    "unsafe",
    "extern",
    "elif",
    "try",
    "except",
    "with",
    "pass",
    "lambda",
    "raise",
    "yield",
    "is",
    "not",
    "and",
    "or",
    "del",
    "global",
    "export",
    "default",
    "public",
    "private",
    "protected",
    "void",
    "final",
    "package",
    "interface",
    "namespace",
    "throw",
    "catch",
    "switch",
    "case",
    "do",
    "include",
    "define",
    "val",
    "fun",
    "vec",
    "option",
    "box",
    "str",
    "bool",
    "char",
    "usize",
    "isize",
    "u8",
    "u16",
    "u32",
    "u64",
    "u128",
    "i8",
    "i16",
    "i32",
    "i64",
    "i128",
    "f32",
    "f64",
    "int",
    "float",
];

/// Lowercase words of `s`: letters and digits (an inner apostrophe stays),
/// split at `_` and camelCase humps, never stemmed; lone numbers dropped.
fn words(s: &str) -> Vec<String> {
    let run = |c: char| !(c.is_alphanumeric() || matches!(c, '_' | '\''));
    let parts = s.split(run).flat_map(|r| r.trim_matches('\'').split('_'));
    let parts = parts.flat_map(humps).map(str::to_lowercase);
    parts
        .filter(|w| !w.bytes().all(|b| b.is_ascii_digit()))
        .collect()
}

fn say(s: &str) -> String {
    words(s).join(" ")
}

/// What a stretch of a line is: code (keywords dropped), a string literal
/// (kept, still code) or a comment (kept, and a line of only comment is
/// documentation).
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Code,
    Str,
    Comment,
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Code,
    /// inside `/* */`
    Block,
    /// inside a Python triple-quoted string
    Doc(&'static str),
    /// inside a Rust `#[...]`, at this bracket depth
    Attr(usize),
}

/// How a file writes comments and strings, by extension.
struct Syntax {
    slash: bool,
    hash: bool,
    rust: bool,
    py: bool,
}

impl Syntax {
    fn of(path: &str) -> Syntax {
        let file = path.rsplit('/').next().unwrap_or(path);
        let ext = file.rsplit_once('.').map_or("", |(_, e)| e);
        let ext = ext.to_ascii_lowercase();
        let py = ext == "py";
        let hash = py
            || matches!(file, "Makefile" | "Dockerfile")
            || [
                "sh", "bash", "zsh", "yml", "yaml", "toml", "rb", "mk", "ini", "cfg", "conf",
            ]
            .contains(&ext.as_str());
        Syntax {
            slash: !hash,
            hash,
            rust: ext == "rs",
            py,
        }
    }

    /// Does a comment, attribute or string start at the head of `t`?
    fn opens(&self, t: &str) -> bool {
        match t.as_bytes()[0] {
            b'/' => self.slash && (t.starts_with("//") || t.starts_with("/*")),
            b'#' => self.hash || (self.rust && (t.starts_with("#[") || t.starts_with("#!["))),
            b'"' => true,
            b'\'' | b'`' => !self.rust,
            _ => false,
        }
    }
}

struct Scan {
    mode: Mode,
    words: Vec<String>,
    code: bool,
}

impl Scan {
    fn push(&mut self, seg: &str, kind: Kind) {
        for w in words(seg) {
            let noise = w.chars().count() < 2 || NOISE.contains(&w.as_str());
            if kind != Kind::Code || !noise {
                self.code |= kind != Kind::Comment;
                self.words.push(w);
            }
        }
    }

    /// Text up to `end` is comment or string; the mode returns to code there.
    fn until<'a>(&mut self, rest: &'a str, end: &str) -> &'a str {
        let Some(n) = rest.find(end) else {
            self.push(rest, Kind::Comment);
            return "";
        };
        self.push(&rest[..n], Kind::Comment);
        self.mode = Mode::Code;
        &rest[n + end.len()..]
    }

    fn line(&mut self, mut rest: &str, x: &Syntax) {
        while !rest.is_empty() {
            match self.mode {
                Mode::Block => rest = self.until(rest, "*/"),
                Mode::Doc(end) => rest = self.until(rest, end),
                Mode::Attr(depth) => {
                    let (mut d, mut cut) = (depth, rest.len());
                    for (i, c) in rest.char_indices() {
                        d = (d + usize::from(c == '[')).saturating_sub(usize::from(c == ']'));
                        if d == 0 {
                            cut = i + 1;
                            break;
                        }
                    }
                    self.mode = if d == 0 { Mode::Code } else { Mode::Attr(d) };
                    rest = &rest[cut..];
                }
                Mode::Code => {
                    let at = rest.char_indices().find(|&(i, _)| x.opens(&rest[i..]));
                    let Some((i, _)) = at else {
                        return self.push(rest, Kind::Code);
                    };
                    self.push(&rest[..i], Kind::Code);
                    rest = &rest[i..];
                    if (x.slash && rest.starts_with("//")) || (x.hash && rest.starts_with('#')) {
                        return self.push(rest, Kind::Comment);
                    }
                    let triple = ["\"\"\"", "'''"].into_iter();
                    if x.rust && rest.starts_with('#') {
                        self.mode = Mode::Attr(0);
                        rest = &rest[1 + usize::from(rest.starts_with("#!"))..];
                    } else if rest.starts_with("/*") {
                        self.mode = Mode::Block;
                        rest = &rest[2..];
                    } else if let Some(q) = triple.into_iter().find(|q| x.py && rest.starts_with(q))
                    {
                        self.mode = Mode::Doc(q);
                        rest = &rest[3..];
                    } else {
                        let q = rest.chars().next().unwrap_or('"');
                        rest = &rest[1..];
                        let mut esc = false;
                        let end = rest.char_indices().find(|&(_, c)| {
                            let hit = !esc && c == q;
                            esc = !esc && c == '\\';
                            hit
                        });
                        let n = end.map_or(rest.len(), |(n, _)| n);
                        self.push(&rest[..n], Kind::Str);
                        rest = rest.get(n + 1..).unwrap_or("");
                    }
                }
            }
        }
    }
}

/// Flattens a file to plain words, one `Line` per source line that has any:
/// comment text and string contents (markers stripped), identifiers split
/// into words, language keywords and punctuation dropped; Rust attributes
/// vanish. Pure extraction: no word the file does not contain. Prose files
/// pass through but for their `#`s and backticks.
fn scrub(path: &str, text: &str) -> Vec<Line> {
    if prose_ext(path).is_some() {
        let lines = text.lines().map(|l| l.replace('`', ""));
        let lines = lines.map(|l| l.trim_start_matches('#').trim().to_owned());
        let lines = lines.filter(|l| !l.is_empty());
        return lines.map(|text| Line { text, code: true }).collect();
    }
    let x = Syntax::of(path);
    let mut s = Scan {
        mode: Mode::Code,
        words: Vec::new(),
        code: false,
    };
    let mut out = Vec::new();
    for l in text.lines() {
        s.words.clear();
        s.code = false;
        s.line(l, &x);
        if !s.words.is_empty() {
            out.push(Line {
                text: s.words.join(" "),
                code: s.code,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(path: &str, symbol: Option<&str>, line: usize, text: &str) -> Chunk {
        Chunk {
            path: path.into(),
            start: line,
            end: line,
            symbol: symbol.map(Into::into),
            text: text.into(),
        }
    }

    /// One chunk per `(path, symbols)` file, one per symbol, adjacent.
    fn corpus(spec: &[(&str, &[&str])]) -> (Vec<Chunk>, Vec<Range<usize>>) {
        let (mut chunks, mut files) = (Vec::new(), Vec::new());
        for (path, syms) in spec {
            let start = chunks.len();
            for (i, s) in syms.iter().enumerate() {
                chunks.push(chunk(path, Some(s), i + 1, &format!("fn {s}() {{}}")));
            }
            files.push(start..chunks.len());
        }
        (chunks, files)
    }

    /// P = 0.9 for previews that mention `key`, 0.1 otherwise; logs each batch.
    fn keyed<'a>(
        key: &'a str,
        log: &'a mut Vec<Vec<String>>,
    ) -> impl FnMut(&[Chunk]) -> Result<Vec<f64>> + 'a {
        move |batch| {
            log.push(batch.iter().map(|c| c.path.clone()).collect());
            Ok(batch
                .iter()
                .map(|c| {
                    if c.text.contains(key) || c.path.contains(key) {
                        0.9
                    } else {
                        0.1
                    }
                })
                .collect())
        }
    }

    #[test]
    fn an_empty_corpus_reaches_nothing() {
        let mut called = false;
        let out = descend(&[], &[], 3, |_| {
            called = true;
            Ok(vec![])
        })
        .unwrap();
        assert!(out.is_empty() && !called);
    }

    #[test]
    fn a_zero_beam_never_calls_the_scorer() {
        let (chunks, files) = corpus(&[("a.rs", &["x"])]);
        let out = descend(&chunks, &files, 0, |_| panic!("scored")).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn a_single_file_root_is_scored_once() {
        let (chunks, files) = corpus(&[("main.rs", &["alpha", "beta"])]);
        let mut log = Vec::new();
        let out = descend(&chunks, &files, 2, keyed("alpha", &mut log)).unwrap();
        assert_eq!(out, [0, 1]);
        assert_eq!(log, [["main.rs"]]);
    }

    #[test]
    fn score_runs_once_per_level_and_the_root_is_never_scored() {
        let (chunks, files) = corpus(&[
            ("top.rs", &["t"]),
            ("net/http/client.rs", &["send"]),
            ("net/http/server.rs", &["serve"]),
            ("net/tcp.rs", &["dial"]),
            ("ui/view.rs", &["draw"]),
        ]);
        let mut log = Vec::new();
        descend(&chunks, &files, 8, keyed("zzz", &mut log)).unwrap();
        assert_eq!(
            log,
            [
                vec!["net/", "ui/", "top.rs"],
                vec!["net/http/", "net/tcp.rs", "ui/view.rs"],
                vec!["net/http/client.rs", "net/http/server.rs"]
            ]
        );
    }

    #[test]
    fn a_pruned_folder_is_never_opened() {
        let (chunks, files) = corpus(&[
            ("auth/login.rs", &["check_password"]),
            ("auth/token.rs", &["issue"]),
            ("ui/view.rs", &["draw"]),
        ]);
        let mut log = Vec::new();
        let out = descend(&chunks, &files, 1, keyed("auth", &mut log)).unwrap();
        assert_eq!(log.len(), 2);
        assert!(log
            .iter()
            .flatten()
            .all(|p| !p.starts_with("ui/") || p == "ui/"));
        assert_eq!(out, [0, 1]);
    }

    #[test]
    fn the_best_file_comes_first_with_its_chunks_in_line_order() {
        let (chunks, files) = corpus(&[
            ("a/dull.rs", &["one"]),
            ("a/hit.rs", &["first", "needle", "last"]),
            ("b.rs", &["two"]),
        ]);
        let mut log = Vec::new();
        let out = descend(&chunks, &files, 1, keyed("needle", &mut log)).unwrap();
        assert_eq!(out, [1, 2, 3, 0, 4]);
    }

    #[test]
    fn a_file_without_symbols_previews_its_first_lines() {
        let c = [chunk("notes.txt", None, 1, "alpha\nbeta")];
        let p = file_preview("notes.txt", &c);
        assert_eq!(
            (p.path.as_str(), p.text.as_str()),
            ("notes.txt", "alpha\nbeta")
        );
    }

    #[test]
    fn previews_are_capped_on_a_line_boundary() {
        let lines = (0..400).map(|i| format!("line{i}"));
        let p = preview("f".into(), lines);
        assert!(p.text.chars().count() <= PREVIEW_CHARS);
        assert!(
            p.text.ends_with(|c: char| c.is_ascii_digit())
                && p.text.lines().all(|l| l.starts_with("line"))
        );
    }

    #[test]
    fn a_folder_preview_lists_children_then_symbols() {
        let (chunks, files) = corpus(&[
            ("net/http/c.rs", &["send"]),
            ("net/tcp.rs", &["dial", "dial"]),
        ]);
        let f = Folder {
            dirs: ["net/http".to_owned()].into(),
            files: vec![1],
        };
        let p = folder_preview("net", &f, &chunks, &files);
        assert_eq!(
            (p.path.as_str(), p.text.as_str()),
            ("net/", "http/\ntcp\ndial")
        );
    }

    #[test]
    fn a_big_file_is_descended_group_by_group_down_to_the_keyword() {
        let names: Vec<String> = (0..40)
            .map(|i| {
                if i == 23 {
                    "needle".into()
                } else {
                    format!("s{i}")
                }
            })
            .collect();
        let syms: Vec<&str> = names.iter().map(String::as_str).collect();
        let (chunks, files) = corpus(&[("big.rs", &syms)]);
        let mut log = Vec::new();
        let out = descend(&chunks, &files, 1, keyed("needle", &mut log)).unwrap();
        assert_eq!(log.iter().map(Vec::len).collect::<Vec<_>>(), [1, 4, 4]);
        assert_eq!(out.len(), 10);
        assert_eq!(out[..3], [22, 23, 24]);
        assert!(out.iter().all(|i| (20..30).contains(i)));
    }

    #[test]
    fn a_file_of_up_to_fanout_chunks_is_reached_whole_without_groups() {
        let (chunks, files) = corpus(&[("a.rs", &["w", "x", "y", "z"])]);
        let mut log = Vec::new();
        let out = descend(&chunks, &files, 1, keyed("zzz", &mut log)).unwrap();
        assert_eq!(out, [0, 1, 2, 3]);
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn a_group_preview_carries_the_file_the_lines_and_the_symbols() {
        let (chunks, files) = corpus(&[("f.rs", &["a", "b", "a", "c", "d"])]);
        let p = group_preview(&chunks[files[0].start + 1..files[0].end]);
        assert_eq!((p.path.as_str(), p.start, p.end), ("f.rs", 2, 5));
        assert_eq!(p.symbol, None);
        assert_eq!(p.text, "b\na\nc\nd");
    }

    #[test]
    fn a_group_preview_skips_blank_lines_and_stops_at_the_cap() {
        let c = [
            chunk("g.rs", None, 1, "\n  \nbody"),
            chunk("g.rs", None, 2, &"x".repeat(PREVIEW_CHARS)),
        ];
        assert_eq!(group_preview(&c).text, "body");
    }

    fn flat(path: &str, text: &str) -> String {
        let lines: Vec<String> = scrub(path, text).into_iter().map(|l| l.text).collect();
        lines.join("\n")
    }

    #[test]
    fn scrub_keeps_rust_comments_and_strings_and_drops_the_rest() {
        let src = "//! Axum surface: the POST /v1/systemone API\n\
                   #![allow(dead_code)]\n\
                   #[derive(Debug,\n    Clone)]\n\
                   /// FNV-1a: stable fold assignment\n\
                   pub fn fold_of(&self, name: &str) -> GrepState<'a> {\n    \
                   let msg = \"model loaded\"; // not reloaded\n    \
                   /* block\n       note */ x + 1\n}\n";
        assert_eq!(
            flat("a.rs", src),
            "axum surface the post v1 systemone api\n\
             fnv 1a stable fold assignment\n\
             fold of name grep state\n\
             msg model loaded not reloaded\n\
             block\n\
             note"
        );
    }

    #[test]
    fn scrub_marks_the_lines_that_hold_code() {
        let l = scrub(
            "a.rs",
            "// about\nfn go() {}\nlet state = \"only\";\n\"lone\",",
        );
        let marks: Vec<(&str, bool)> = l.iter().map(|l| (l.text.as_str(), l.code)).collect();
        assert_eq!(
            marks,
            [
                ("about", false),
                ("go", true),
                ("state only", true),
                ("lone", true)
            ]
        );
    }

    #[test]
    fn scrub_reads_python_docstrings_and_hash_comments() {
        let src = "\"\"\"Handle the nightly sync.\n\nSecond line.\"\"\"\nimport os\n\
                   # cache directory\ndef fold_of(x):  # stable\n    return os.path.join(\"a b\")\n";
        assert_eq!(
            flat("a.py", src),
            "handle the nightly sync\nsecond line\nos\ncache directory\nfold of stable\nos path join a b"
        );
    }

    #[test]
    fn scrub_does_not_take_a_url_in_a_string_for_a_comment() {
        assert_eq!(
            flat("a.js", "get(\"http://x.io/a\") // done"),
            "get http x io a done"
        );
    }

    #[test]
    fn scrub_passes_prose_through_without_markup() {
        let md = "# Title `x`\n\n## Loaded models\nthe model, loaded once\n";
        assert_eq!(
            flat("README.md", md),
            "Title x\nLoaded models\nthe model, loaded once"
        );
    }

    #[test]
    fn a_file_preview_leads_with_the_module_docs_then_outlines_the_symbols() {
        let head = "//! Axum surface: the single POST /v1/systemone API\nuse std::io;\n";
        let warm =
            "/// Compile the hot pipelines before the first real request\npub fn warmup() {}";
        let bare = "pub fn active() {}";
        let c = [
            chunk("src/server.rs", None, 1, head),
            chunk("src/server.rs", Some("warmup"), 3, warm),
            chunk("src/server.rs", Some("active"), 5, bare),
        ];
        let p = file_preview("src/server.rs", &c);
        assert_eq!(
            p.text,
            "axum surface the single post v1 systemone api\n\
             warmup: compile the hot pipelines before the first real request\n\
             active\n\
             std io\n\
             warmup"
        );
    }

    #[test]
    fn a_folder_preview_names_each_file_with_its_module_docs() {
        let c = [
            chunk("net/server.rs", None, 1, "//! Axum surface\nfn serve() {}"),
            chunk("net/tcp.rs", None, 1, "fn dial() {}"),
        ];
        let f = Folder {
            dirs: ["net/http_api".to_owned()].into(),
            files: vec![0, 1],
        };
        let p = folder_preview("net", &f, &c, &[0..1, 1..2]);
        assert_eq!(p.text, "http api/\nserver: axum surface\ntcp");
    }
}
