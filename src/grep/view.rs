//! What a search prints: hits on a rail with the query's words in bold, a
//! JSON document, or just the paths. Pure functions of the hits (no model, no
//! store) so the layout is tested as plain strings; a pipe gets the same
//! layout as a terminal, without colors or the width cut.

use std::collections::HashSet;
use std::ops::Range;
use std::path::Path;

use serde_json::{json, Value};

use super::corpus::Chunk;
use super::lexical;

/// 1146 -> "1,146"
pub(super) fn thousands(n: usize) -> String {
    let d = n.to_string();
    let mut out = String::new();
    for (i, ch) in d.chars().enumerate() {
        if i > 0 && (d.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// A chunk's path as the user would type it. Chunks name their file below the
/// root searched; read from where the user stands that is the root as given,
/// then the chunk's path — as ripgrep prints it, and nothing for `.`.
pub(super) struct Shown {
    root: String,
    file: bool,
}

impl Shown {
    pub(super) fn new(root: &Path) -> Shown {
        let r = root.to_string_lossy();
        Shown {
            root: if r == "." {
                String::new()
            } else {
                r.into_owned()
            },
            file: root.is_file(),
        }
    }

    pub(super) fn path(&self, rel: &str) -> String {
        match (self.file, self.root.is_empty()) {
            (true, _) => self.root.clone(),
            (false, true) => rel.to_string(),
            (false, false) => format!("{}/{rel}", self.root.trim_end_matches(['/', '\\'])),
        }
    }
}

pub(super) struct Hit<'a> {
    pub(super) chunk: &'a Chunk,
    pub(super) shown: String,
    /// None when only recall ran
    pub(super) p: Option<f64>,
    /// below the threshold, shown only to make up MIN_SHOWN
    pub(super) weak: bool,
    /// 1-based position in the list
    pub(super) rank: usize,
}

impl Hit<'_> {
    /// The first `lines` source lines of the chunk.
    fn head(&self, lines: usize) -> impl Iterator<Item = &str> {
        self.chunk.text.lines().take(lines)
    }
}

/// Source lines a hit shows: the window where the query's words fall.
pub(super) const SNIPPET: usize = 6;
/// What `--json` carries per hit unless `--lines` says otherwise.
pub(super) const JSON_LINES: usize = 20;
/// The widest a line is drawn.
const MAX_WIDTH: usize = 120;

// SGR codes. The palette is quiet on purpose: text in the terminal's own
// color, gray for what is secondary, a tinted bullet for the confidence.
const BOLD: &str = "1";
pub(super) const DIM: &str = "2";
const GREEN: &str = "32";
const YELLOW: &str = "33";

/// How hits are drawn: ANSI colors, and a width that long lines are cut to.
/// A pipe gets the same layout without either.
#[derive(Clone, Copy, Default)]
pub(super) struct Look {
    color: bool,
    width: Option<usize>,
}

impl Look {
    /// For a stream: colors as `--color` forces them, else on a terminal
    /// unless NO_COLOR is set; a terminal's width, capped.
    pub(super) fn of(tty: bool, color: Option<bool>) -> Look {
        let no_color = std::env::var_os("NO_COLOR").is_some_and(|v| !v.is_empty());
        Look {
            color: color.unwrap_or(tty && !no_color),
            width: tty.then(term_width).flatten().map(|w| w.min(MAX_WIDTH)),
        }
    }

    pub(super) fn paint(&self, sgr: &str, s: &str) -> String {
        if self.color && !s.is_empty() {
            format!("\x1b[{sgr}m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    /// `● path:start-end · symbol · 85%`, the bullet green from 80%, yellow
    /// from 60%, gray below; a hit under the threshold takes a gray `○`, a
    /// recall rank `#3` a `◇`. A long symbol gives way to the width, the path
    /// never does.
    fn header(&self, h: &Hit) -> String {
        let c = h.chunk;
        let loc = format!("{}:{}-{}", h.shown, c.start, c.end);
        let (bullet, score) = match h.p {
            Some(p) => {
                let tone = match p {
                    p if p >= 0.8 => GREEN,
                    p if p >= 0.6 => YELLOW,
                    _ => DIM,
                };
                let bullet = if h.weak {
                    self.paint(DIM, "○")
                } else {
                    self.paint(tone, "●")
                };
                (bullet, format!("{:.0}%", p * 100.0))
            }
            None => (self.paint(DIM, "◇"), format!("#{}", h.rank)),
        };
        let dot = self.paint(DIM, " · ");
        let mut sym = c.symbol.clone().unwrap_or_default();
        if let Some(w) = self.width {
            // bullet, the location, the score and both separators stay whole
            sym = cut(
                &sym,
                w.saturating_sub(loc.chars().count() + score.len() + 8),
            );
        }
        let mut out = format!("{bullet} {}", self.paint(BOLD, &loc));
        if !sym.is_empty() {
            out += &format!("{dot}{sym}");
        }
        out + &dot + &score
    }

    /// `n` lines of the chunk on a rail, numbered: the window holding most of
    /// the query's words (in bold), dedented, cut to the width, with `⋮`
    /// where lines are left out. A chunk only two lines longer shows whole.
    fn snippet(&self, c: &Chunk, n: usize, words: &HashSet<String>) -> String {
        let lines: Vec<String> = c.text.lines().map(|l| l.replace('\t', "    ")).collect();
        let marks: Vec<Vec<Range<usize>>> = lines.iter().map(|l| matches(l, words)).collect();
        let (from, to) = if lines.len() <= n + 2 {
            (0, lines.len())
        } else {
            let w = window(&marks, n);
            (w, w + n)
        };
        let indent = lines[from..to]
            .iter()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.len() - l.trim_start_matches(' ').len())
            .min()
            .unwrap_or(0);
        let gw = (c.start + to.max(1) - 1).to_string().len();
        let room = self.width.map(|w| w.saturating_sub(gw + 4));
        let gap = self.paint(DIM, &format!("{:>gw$}", "⋮"));
        let mut rows = Vec::new();
        if from > 0 {
            rows.push(gap.clone());
        }
        for i in from..to {
            let text = lines[i].get(indent..).unwrap_or("");
            let shown = room.map_or(text.to_string(), |r| cut(text, r));
            // a cut line keeps its marks up to where its text stops
            let keep = shown.strip_suffix('…').unwrap_or(&shown).len();
            let mut line = String::new();
            let mut at = 0;
            for m in &marks[i] {
                let (a, b) = (
                    m.start.saturating_sub(indent),
                    m.end.saturating_sub(indent).min(keep),
                );
                if a >= b || a < at {
                    continue;
                }
                line += &shown[at..a];
                line += &self.paint(BOLD, &shown[a..b]);
                at = b;
            }
            line += &shown[at..];
            let num = self.paint(DIM, &format!("{:>gw$}", c.start + i));
            rows.push(format!("{num}  {line}").trim_end().to_string());
        }
        if to < lines.len() {
            rows.push(gap);
        }
        let last = rows.len().saturating_sub(1);
        let mut out = String::new();
        for (i, r) in rows.iter().enumerate() {
            let rail = self.paint(DIM, if i == last { "└" } else { "│" });
            out += &format!("{rail} {r}\n");
        }
        out
    }
}

/// The terminal's columns: stdout's window size, else $COLUMNS.
fn term_width() -> Option<usize> {
    #[cfg(unix)]
    {
        // SAFETY: TIOCGWINSZ fills the winsize it is handed, nothing else
        let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
        if unsafe { libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &mut ws) } == 0
            && ws.ws_col > 0
        {
            return Some(ws.ws_col as usize);
        }
    }
    std::env::var("COLUMNS").ok()?.parse().ok()
}

/// `s` in at most `n` columns: past them, the head and an ellipsis.
fn cut(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        return s.to_string();
    }
    let head: String = s.chars().take(n.saturating_sub(1)).collect();
    if n == 0 {
        head
    } else {
        head + "…"
    }
}

/// Byte ranges of the words of `line` that share a term with the query,
/// under recall's own analyzer: `tunes` and `fine_tuning` meet `tuning`.
fn matches(line: &str, words: &HashSet<String>) -> Vec<Range<usize>> {
    let word = |c: char| c.is_alphanumeric() || c == '_';
    let mut out = Vec::new();
    let mut start = None;
    for (i, ch) in line.char_indices().chain([(line.len(), ' ')]) {
        match (word(ch), start) {
            (true, None) => start = Some(i),
            (false, Some(a)) => {
                if lexical::terms(&line[a..i])
                    .iter()
                    .any(|t| words.contains(t))
                {
                    out.push(a..i);
                }
                start = None;
            }
            _ => {}
        }
    }
    out
}

/// Where the `n`-line window with the most marked words starts, moved to
/// center the marked lines it holds; the top when nothing is marked.
fn window(marks: &[Vec<Range<usize>>], n: usize) -> usize {
    let counts: Vec<usize> = marks.iter().map(Vec::len).collect();
    let last = counts.len().saturating_sub(n);
    let best = (0..=last)
        .rev()
        .max_by_key(|&s| counts[s..(s + n).min(counts.len())].iter().sum::<usize>())
        .unwrap_or(0);
    let mut held = (best..(best + n).min(counts.len())).filter(|&i| counts[i] > 0);
    let Some(lo) = held.next() else {
        return best;
    };
    let hi = held.next_back().unwrap_or(lo);
    ((lo + hi) / 2).saturating_sub((n - 1) / 2).min(last)
}

/// Each hit: its header, then its snippet unless `lines` is 0, a blank line
/// between hits that show source.
pub(super) fn human(hits: &[Hit], lines: usize, query: &str, look: Look) -> String {
    let words: HashSet<String> = lexical::terms(query).into_iter().collect();
    let mut out = String::new();
    for (i, h) in hits.iter().enumerate() {
        if i > 0 && lines > 0 {
            out.push('\n');
        }
        out += &look.header(h);
        out.push('\n');
        if lines > 0 {
            out += &look.snippet(h.chunk, lines, &words);
        }
    }
    out
}

pub(super) fn json_doc(query: &str, hits: &[Hit], lines: usize, x_snap: Value) -> Value {
    let hits: Vec<Value> = hits
        .iter()
        .map(|h| {
            json!({
                "path": h.shown,
                "start": h.chunk.start,
                "end": h.chunk.end,
                "symbol": h.chunk.symbol,
                "p": h.p,
                "text": h.head(lines).collect::<Vec<_>>().join("\n"),
            })
        })
        .collect();
    json!({"query": query, "hits": hits, "x_snap": x_snap})
}

/// `-l`: each path once, in the order of its first hit.
pub(super) fn files(hits: &[Hit]) -> String {
    let mut seen = HashSet::new();
    hits.iter()
        .filter(|h| seen.insert(&h.shown))
        .map(|h| format!("{}\n", h.shown))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::tests::{chunk, Tmp};
    use super::*;

    fn hit<'a>(c: &'a Chunk, shown: &str, p: Option<f64>, rank: usize) -> Hit<'a> {
        Hit {
            chunk: c,
            shown: shown.into(),
            p,
            weak: false,
            rank,
        }
    }

    #[test]
    fn human_output_hangs_each_hits_source_on_a_rail() {
        let a = chunk("src/a.rs", 10, Some("a"), "fn a() {\n    1\n}");
        let b = chunk("src/b.rs", 3, None, "x\n\ty");
        let hits = [
            hit(&a, "src/a.rs", Some(0.944), 1),
            hit(&b, "src/b.rs", Some(0.5), 2),
        ];
        let pipe = Look::default();
        assert_eq!(
            human(&hits, 6, "q", pipe),
            "● src/a.rs:10-12 · a · 94%\n│ 10  fn a() {\n│ 11      1\n└ 12  }\n\n\
             ● src/b.rs:3-4 · 50%\n│ 3  x\n└ 4      y\n"
        );
        // locations only: a line per hit, nothing between
        assert_eq!(
            human(&hits, 0, "q", pipe),
            "● src/a.rs:10-12 · a · 94%\n● src/b.rs:3-4 · 50%\n"
        );
        // recall only: a hollow bullet and the rank where the probability would be
        let r = [hit(&a, "src/a.rs", None, 1)];
        assert_eq!(human(&r, 0, "q", pipe), "◇ src/a.rs:10-12 · a · #1\n");
        // under the threshold, shown to make up the number: a hollow bullet
        let w = [Hit {
            weak: true,
            ..hit(&a, "src/a.rs", Some(0.47), 1)
        }];
        assert_eq!(human(&w, 0, "q", pipe), "○ src/a.rs:10-12 · a · 47%\n");
        assert_eq!(human(&[], 6, "q", pipe), "");
        // on a terminal: a tinted bullet, the path bold, the rest gray
        let tty = Look {
            color: true,
            width: None,
        };
        assert_eq!(
            tty.header(&hits[0]),
            "\x1b[32m●\x1b[0m \x1b[1msrc/a.rs:10-12\x1b[0m\x1b[2m · \x1b[0ma\x1b[2m · \x1b[0m94%"
        );
    }

    #[test]
    fn a_snippet_shows_the_window_where_the_query_falls() {
        let body: Vec<String> = (1..=20).map(|i| format!("    line {i}")).collect();
        let mut text = body.join("\n");
        text = text.replace("line 14", "the pool is drained here");
        let c = chunk("src/p.rs", 100, Some("p"), &text);
        let words: HashSet<String> = lexical::terms("where is the pool drained")
            .into_iter()
            .collect();
        let pipe = Look::default();
        // 6 lines around the words, dedented, `⋮` for what is left out on both sides
        assert_eq!(
            pipe.snippet(&c, 6, &words),
            "│   ⋮\n│ 111  line 12\n│ 112  line 13\n│ 113  the pool is drained here\n\
             │ 114  line 15\n│ 115  line 16\n│ 116  line 17\n└   ⋮\n"
        );
        // no word of the query anywhere: the top of the chunk
        let none: HashSet<String> = HashSet::new();
        assert!(pipe
            .snippet(&c, 2, &none)
            .starts_with("│ 100  line 1\n│ 101  line 2\n└   ⋮\n"));
        // the words in bold, a long line cut at the width (rail and number
        // take 5 of its 20 columns), its marks with it
        let tty = Look {
            color: true,
            width: Some(20),
        };
        let one = chunk(
            "a.rs",
            1,
            None,
            "fn drain_pool() { pool.drain(); pool.close() }",
        );
        assert_eq!(
            tty.snippet(&one, 6, &words),
            "\x1b[2m└\x1b[0m \x1b[2m1\x1b[0m  fn \x1b[1mdrain_pool\x1b[0m(…\n"
        );
    }

    #[test]
    fn query_words_meet_their_inflections_and_cuts_count_columns() {
        let words: HashSet<String> = lexical::terms("fine tuning").into_iter().collect();
        let line = "it fine-tunes a model; fine_tuned too, not finest";
        let got: Vec<&str> = matches(line, &words)
            .into_iter()
            .map(|r| &line[r])
            .collect();
        assert_eq!(got, ["fine", "tunes", "fine_tuned"]);
        assert_eq!(cut("abcdef", 4), "abc…");
        assert_eq!(cut("abc", 4), "abc");
        assert_eq!(cut("àèìòù", 3), "àè…");
        assert_eq!(cut("abc", 0), "");
        assert_eq!(thousands(1146), "1,146");
        assert_eq!(thousands(108489), "108,489");
        assert_eq!(thousands(999), "999");
    }

    #[test]
    fn json_output_is_one_document_of_hits_and_stats() {
        let a = chunk("src/a.rs", 10, Some("a"), "fn a() {\n    1\n}");
        let b = chunk("src/b.rs", 3, None, "x");
        let hits = [
            hit(&a, "./src/a.rs", Some(0.9), 1),
            hit(&b, "./src/b.rs", None, 2),
        ];
        let doc = json_doc("where?", &hits, 2, json!({"hits": 2}));
        let want = json!({
            "query": "where?",
            "hits": [
                {"path": "./src/a.rs", "start": 10, "end": 12, "symbol": "a", "p": 0.9, "text": "fn a() {\n    1"},
                {"path": "./src/b.rs", "start": 3, "end": 3, "symbol": null, "p": null, "text": "x"},
            ],
            "x_snap": {"hits": 2},
        });
        assert_eq!(doc, want);
        // locations only leaves the text empty, the field stays
        assert_eq!(json_doc("q", &hits, 0, json!({}))["hits"][0]["text"], "");
    }

    #[test]
    fn files_only_lists_each_path_once_in_hit_order() {
        let c = chunk("x", 1, None, "x");
        let hits: Vec<Hit> = ["b.rs", "a.rs", "b.rs", "c.rs", "a.rs"]
            .iter()
            .enumerate()
            .map(|(i, p)| hit(&c, p, Some(0.9), i + 1))
            .collect();
        assert_eq!(files(&hits), "b.rs\na.rs\nc.rs\n");
    }

    #[test]
    fn paths_are_shown_from_where_the_user_stands() {
        let t = Tmp::new("shown");
        let file = t.write("one.rs", "fn one() {}");
        let show = |root: &str, rel: &str| Shown::new(Path::new(root)).path(rel);
        assert_eq!(show(".", "src/a.rs"), "src/a.rs");
        assert_eq!(show("src", "a.rs"), "src/a.rs");
        assert_eq!(show("./src/", "a.rs"), "./src/a.rs");
        assert_eq!(show("../other", "src/a.rs"), "../other/src/a.rs");
        // a file searched on its own is shown as it was named
        assert_eq!(Shown::new(&file).path("one.rs"), file.to_string_lossy());
    }
}
