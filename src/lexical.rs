//! First-stage recall for `snap grep`: BM25F over code-aware terms. The
//! model reranks; this stage only has to put the right chunks in the first
//! wave, cheaply and without a model. Identifiers split the way people
//! paraphrase them (`ConnectionPool` → connection, pool, and the whole
//! identifier), paths and declared names weigh more than bodies, and
//! several rankings fuse by reciprocal rank.
//!
//! Queries and documents share one analyzer, so casing, plurals and tense
//! cancel out between them. Words are Unicode and keep their accents; Han,
//! kana and Hangul, written without spaces, index as overlapping character
//! pairs. Stemming and stopwords are English-only and leave non-ASCII words
//! alone. BM25F's weights and length normalization depend on the document
//! alone, so `build` folds them into one factor per field and `search` is a
//! walk over the postings of the query terms.
#![allow(dead_code)] // wired by grep.rs

use std::collections::HashMap;

/// BM25F per field (path, symbol, body): weight, and `B`, how strongly the
/// field's length normalizes its term frequency. `K1` saturates the sum.
const WEIGHT: [f32; 3] = [2.0, 3.0, 1.0];
const B: [f32; 3] = [0.5, 0.5, 0.75];
const K1: f32 = 1.2;

/// The glue of an English query. Code words (`new`, `get`, `set`, `file`,
/// `error`) are not here. Sorted, for `binary_search`.
const STOP: &[&str] = &[
    "about", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "how", "if", "in", "is", "it", "its",
    "of", "on", "or", "shall", "should", "than", "that", "the", "their", "them", "these", "they",
    "this", "those", "to", "was", "we", "were", "what", "when", "where", "which", "who", "why",
    "will", "with", "would", "you", "your",
];

/// Terms of a text, queries and documents alike: identifiers whole and
/// split at `_`, `-`, `::`, `.`, `/` and camelCase humps, lowercased,
/// lightly stemmed, English function words dropped.
///
/// In document order, duplicates kept: a document's term frequency is how
/// often a term occurs. A word is a run of letters and digits, accents
/// included. Of an identifier the parts come first; with two or more, the
/// whole name follows, lowercased, underscores gone and unstemmed
/// (`connectionpool` for `ConnectionPool`, `connection_pool` and
/// `connectionPool`). Han, kana and Hangul runs become overlapping
/// character bigrams, a lone character standing for itself. Stemming and
/// stopwords apply to ASCII words only.
pub fn terms(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    scan(text, &mut |t| out.push(t.to_owned()));
    out
}

/// `terms` without an allocation per term: what `Index` runs over every
/// chunk.
fn scan(text: &str, emit: &mut impl FnMut(&str)) {
    let (mut part, mut whole) = (String::new(), String::new());
    for (run, han) in runs(text) {
        if han {
            bigrams(run, emit);
            continue;
        }
        whole.clear();
        let mut n = 0;
        for p in run.split('_').flat_map(humps) {
            n += 1;
            lower(p, &mut part);
            whole.push_str(&part);
            let ascii = p.is_ascii();
            // a lone ASCII letter is noise; a lone `è` or `λ` is a word
            if p.len() < 2 || (ascii && stop(&part)) {
                continue;
            }
            if ascii {
                stem(&mut part);
            }
            emit(&part);
        }
        if n > 1 && whole.chars().count() >= 3 {
            emit(&whole);
        }
    }
}

fn word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Han, Hiragana, Katakana and Hangul syllables: no spaces between words,
/// so the unit that matches is the character pair. Punctuation inside the
/// blocks (the katakana middle dot) still separates.
fn cjk(c: char) -> bool {
    matches!(c, '\u{3040}'..='\u{30ff}' | '\u{3400}'..='\u{4dbf}' | '\u{4e00}'..='\u{9fff}' | '\u{ac00}'..='\u{d7a3}')
        && c.is_alphanumeric()
}

/// Maximal runs of word characters, split again where CJK meets anything
/// else (`数据库connection`): (run, is it CJK).
fn runs(text: &str) -> impl Iterator<Item = (&str, bool)> {
    let mut rest = text;
    std::iter::from_fn(move || {
        rest = rest.trim_start_matches(|c| !word(c));
        let han = cjk(rest.chars().next()?);
        let end = rest
            .find(|c| !word(c) || cjk(c) != han)
            .unwrap_or(rest.len());
        let (run, tail) = rest.split_at(end);
        rest = tail;
        Some((run, han))
    })
}

fn bigrams(run: &str, emit: &mut impl FnMut(&str)) {
    let mut prev = None;
    for (i, c) in run.char_indices() {
        if let Some(p) = prev {
            emit(&run[p..i + c.len_utf8()]);
        }
        prev = Some(i);
    }
    // one character only: the loop never paired it
    if prev == Some(0) {
        emit(run);
    }
}

/// The parts of an underscore-free segment: split at camelCase humps, with
/// acronyms whole (`HTTPServer` → HTTP, Server) and digits stuck to the
/// letters before them (`parseJSON2` → parse, JSON2).
fn humps(seg: &str) -> impl Iterator<Item = &str> {
    let mut from = 0;
    std::iter::from_fn(move || {
        if from == seg.len() {
            return None;
        }
        let start = from;
        from = seg[start..]
            .char_indices()
            .skip(1)
            .map(|(i, _)| start + i)
            .find(|&i| hump(seg, i))
            .unwrap_or(seg.len());
        Some(&seg[start..from])
    })
}

/// A part starts at byte `i`: an uppercase letter after anything but one
/// (`parseJSON`), or the last of an acronym's capitals before lowercase
/// (`HTTPServer`).
fn hump(s: &str, i: usize) -> bool {
    let mut at = s[i..].chars();
    at.next().is_some_and(char::is_uppercase)
        && (!s[..i].chars().next_back().is_some_and(char::is_uppercase)
            || at.next().is_some_and(char::is_lowercase))
}

/// `s` lowercased into `buf`; ASCII, nearly all code, in place.
fn lower(s: &str, buf: &mut String) {
    buf.clear();
    if s.is_ascii() {
        buf.push_str(s);
        buf.make_ascii_lowercase();
    } else {
        buf.push_str(&s.to_lowercase());
    }
}

fn stop(w: &str) -> bool {
    STOP.binary_search(&w).is_ok()
}

/// Plural, -ing and -ed endings off, so `connections`, `policies` and
/// `stopped` meet `connection`, `policy` and `stop`. Plurals go first, so
/// `strings` and `string` land together. Length floors keep short words
/// whole (`bus`, `thing`, `need`) and `ss`/`us`/`is` endings are no plurals
/// (`class`, `status`, `analysis`). ASCII only.
fn stem(w: &mut String) {
    let n = w.len();
    if n > 4 && w.ends_with("ies") {
        w.truncate(n - 3);
        w.push('y');
    } else if w.ends_with("sses") {
        w.truncate(n - 2);
    } else if n > 3
        && w.ends_with('s')
        && !w.ends_with("ss")
        && !w.ends_with("us")
        && !w.ends_with("is")
    {
        w.pop();
    }
    let n = w.len();
    let cut = if n > 5 && w.ends_with("ing") {
        3
    } else if n > 4 && w.ends_with("ed") {
        2
    } else {
        return;
    };
    w.truncate(n - cut);
    // `running` → `run`; but `call`, `pass`, `diff` and `add` keep the double
    // letter of the word itself, or `called` and `call` would part ways
    if matches!(w.as_bytes(), [_, _, .., x, y] if x == y && x.is_ascii_lowercase() && !b"aeioulsfz".contains(x))
    {
        w.pop();
    }
}

/// One searchable unit, in the fields BM25F weighs separately.
#[derive(Debug, Clone, Copy)]
pub struct Doc<'a> {
    pub path: &'a str,
    pub symbol: Option<&'a str>,
    pub body: &'a str,
}

/// A term's count in one doc, per field (path, symbol, body).
struct Posting {
    doc: u32,
    tf: [u16; 3],
}

/// In-memory inverted index over a corpus, rebuilt per run.
pub struct Index {
    dict: HashMap<String, u32>,
    /// per term id, in doc order
    postings: Vec<Vec<Posting>>,
    idf: Vec<f32>,
    /// per doc and field: weight / (1 - b + b * len / avglen)
    norm: Vec<[f32; 3]>,
}

impl Index {
    pub fn build<'a>(docs: impl IntoIterator<Item = Doc<'a>>) -> Index {
        let mut dict: HashMap<String, u32> = HashMap::new();
        let mut postings: Vec<Vec<Posting>> = Vec::new();
        let mut lens: Vec<[u32; 3]> = Vec::new();
        for (d, doc) in docs.into_iter().enumerate() {
            let d = d as u32;
            let mut len = [0; 3];
            let fields = [doc.path, doc.symbol.unwrap_or(""), doc.body];
            for (f, text) in fields.into_iter().enumerate() {
                scan(text, &mut |t| {
                    len[f] += 1;
                    let id = match dict.get(t) {
                        Some(&id) => id,
                        None => {
                            let id = postings.len() as u32;
                            dict.insert(t.to_owned(), id);
                            postings.push(Vec::new());
                            id
                        }
                    };
                    let list = &mut postings[id as usize];
                    // a doc's fields arrive back to back: its posting is the last
                    match list.last_mut() {
                        Some(p) if p.doc == d => p.tf[f] = p.tf[f].saturating_add(1),
                        _ => list.push(Posting {
                            doc: d,
                            tf: std::array::from_fn(|g| u16::from(g == f)),
                        }),
                    }
                });
            }
            lens.push(len);
        }
        let n = lens.len();
        let avg: [f32; 3] = std::array::from_fn(|f| {
            lens.iter().map(|l| u64::from(l[f])).sum::<u64>() as f32 / n.max(1) as f32
        });
        let norm = lens
            .iter()
            .map(|l| {
                std::array::from_fn(|f| {
                    // a field no doc has (avglen 0) has no terms to weigh
                    let rel = if avg[f] > 0.0 {
                        l[f] as f32 / avg[f]
                    } else {
                        0.0
                    };
                    WEIGHT[f] / (1.0 - B[f] + B[f] * rel)
                })
            })
            .collect();
        let idf = postings
            .iter()
            .map(|p| {
                let df = p.len() as f64;
                ((n as f64 - df + 0.5) / (df + 0.5)).ln_1p() as f32
            })
            .collect();
        Index {
            dict,
            postings,
            idf,
            norm,
        }
    }

    pub fn len(&self) -> usize {
        self.norm.len()
    }

    /// Top `k` docs for `query`, best first: (doc index, score > 0), ties
    /// broken by doc index. Docs sharing no term with the query never show.
    pub fn search(&self, query: &str, k: usize) -> Vec<(usize, f32)> {
        let mut ids: Vec<u32> = Vec::new();
        scan(query, &mut |t| ids.extend(self.dict.get(t)));
        ids.sort_unstable();
        ids.dedup();
        // dense: 4 bytes a doc, and a common term touches most docs anyway
        let mut score = vec![0.0; self.norm.len()];
        for id in ids {
            let idf = self.idf[id as usize];
            for p in &self.postings[id as usize] {
                let w = &self.norm[p.doc as usize];
                let tf: f32 = p.tf.iter().zip(w).map(|(&t, &w)| f32::from(t) * w).sum();
                score[p.doc as usize] += idf * tf / (K1 + tf);
            }
        }
        let mut hits: Vec<(usize, f32)> = score
            .into_iter()
            .enumerate()
            .filter(|&(_, s)| s > 0.0)
            .collect();
        let best = |a: &(usize, f32), b: &(usize, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
        if hits.len() > k {
            hits.select_nth_unstable_by(k, best);
            hits.truncate(k);
        }
        hits.sort_unstable_by(best);
        hits
    }
}

/// Reciprocal rank fusion: score(d) = Σ 1 / (k + rank), rank 1-based, over
/// the rankings that hold d. Best first, ties broken by doc index.
pub fn rrf(rankings: &[Vec<usize>], k: f64) -> Vec<(usize, f64)> {
    let mut held: Vec<(usize, usize)> = rankings
        .iter()
        .flat_map(|r| r.iter().enumerate().map(|(i, &d)| (d, i + 1)))
        .collect();
    // a doc's ranks add up in rank order whatever ranking they came from, so
    // docs holding the same ranks tie exactly instead of by rounding
    held.sort_unstable();
    let mut fused: Vec<(usize, f64)> = Vec::new();
    for (d, rank) in held {
        let s = 1.0 / (k + rank as f64);
        match fused.last_mut() {
            Some(last) if last.0 == d => last.1 += s,
            _ => fused.push((d, s)),
        }
    }
    fused.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(body: &str) -> Doc<'_> {
        Doc {
            path: "",
            symbol: None,
            body,
        }
    }

    fn index(bodies: &[&str]) -> Index {
        Index::build(bodies.iter().map(|b| doc(b)))
    }

    fn ids<T>(hits: &[(usize, T)]) -> Vec<usize> {
        hits.iter().map(|h| h.0).collect()
    }

    #[test]
    fn identifier_forms_share_parts_and_whole() {
        for id in [
            "ConnectionPool",
            "connection_pool",
            "connectionPool",
            "CONNECTION_POOL",
        ] {
            assert_eq!(terms(id), ["connection", "pool", "connectionpool"], "{id}");
        }
        // the parts are stemmed, the whole name is not
        assert_eq!(
            terms("connection_pools"),
            ["connection", "pool", "connectionpools"]
        );
        assert_eq!(
            terms("close ConnectionPool open"),
            ["close", "connection", "pool", "connectionpool", "open"]
        );
    }

    #[test]
    fn acronyms_and_digits() {
        assert_eq!(
            terms("HTTPServerConfig"),
            ["http", "server", "config", "httpserverconfig"]
        );
        assert_eq!(terms("parseJSON2"), ["parse", "json2", "parsejson2"]);
        assert_eq!(
            terms("getHTTPResponse"),
            ["get", "http", "response", "gethttpresponse"]
        );
        assert_eq!(
            terms("XMLHttpRequest"),
            ["xml", "http", "request", "xmlhttprequest"]
        );
        assert_eq!(terms("sha_256"), ["sha", "256", "sha256"]);
        assert_eq!(terms("utf8"), ["utf8"]);
    }

    #[test]
    fn separators_and_short_parts() {
        assert_eq!(
            terms("std::collections::HashMap"),
            ["std", "collection", "hash", "map", "hashmap"]
        );
        assert_eq!(
            terms("src/kv_store.rs"),
            ["src", "kv", "store", "kvstore", "rs"]
        );
        assert!(terms("").is_empty() && terms(" \n\t").is_empty());
        assert!(terms("a.b/c-d$e").is_empty());
        // a single letter drops but still counts as a part; the whole name needs 3 chars
        assert!(terms("a_b").is_empty());
        assert_eq!(terms("x_yz"), ["yz", "xyz"]);
        // one part, no whole name
        assert_eq!(terms("Pool"), ["pool"]);
        assert_eq!(terms("__init__"), ["init"]);
        // term frequency needs the repeats
        assert_eq!(terms("foo foo"), ["foo", "foo"]);
    }

    #[test]
    fn stopwords_go_code_words_stay() {
        assert_eq!(
            terms("How does the cache of the new file get set?"),
            ["cache", "new", "file", "get", "set"]
        );
        assert_eq!(terms("what is an error"), ["error"]);
        // function-word parts drop, the whole name keeps them
        assert_eq!(terms("is_empty"), ["empty", "isempty"]);
        assert!(STOP.is_sorted());
        assert!(STOP
            .iter()
            .all(|w| w.len() > 1 && w.bytes().all(|c| c.is_ascii_lowercase())));
    }

    #[test]
    fn stemming() {
        for (word, stem) in [
            ("connections", "connection"),
            ("policies", "policy"),
            ("queries", "query"),
            ("running", "run"),
            ("stopped", "stop"),
            ("tagged", "tag"),
            ("occurred", "occur"),
            ("class", "class"),
            ("classes", "class"),
            ("status", "status"),
            ("analysis", "analysis"),
            ("passed", "pass"),
            ("passes", "pass"),
            ("uses", "use"),
            ("ties", "tie"),
            // too short to lose a suffix
            ("bus", "bus"),
            ("thing", "thing"),
            ("need", "need"),
        ] {
            assert_eq!(terms(word), [stem], "{word}");
        }
    }

    #[test]
    fn forms_of_a_word_land_together() {
        for forms in [
            &["string", "strings"][..],
            &["set", "sets", "setting", "settings"],
            &["add", "adds", "added", "adding"],
            &["call", "calls", "called", "calling"],
            &["diff", "diffs", "diffed", "diffing"],
            &["map", "maps", "mapped", "mapping"],
        ] {
            let stems: Vec<_> = forms.iter().map(|w| terms(w)).collect();
            assert!(
                stems.windows(2).all(|p| p[0] == p[1]),
                "{forms:?} -> {stems:?}"
            );
        }
    }

    #[test]
    fn unicode_words_stay_whole() {
        assert_eq!(
            terms("perché università Größe"),
            ["perché", "università", "größe"]
        );
        assert_eq!(terms("PERCHÉ UNIVERSITÀ"), ["perché", "università"]);
        // humps are Unicode too: capitals and lowercase beyond ASCII
        assert_eq!(terms("GrößeMaß"), ["größe", "maß", "größemaß"]);
        assert_eq!(terms("ÜberSetzung"), ["über", "setzung", "übersetzung"]);
        assert_eq!(terms("größeÜbung"), ["größe", "übung", "größeübung"]);
        assert_eq!(terms("ÉTATServer"), ["état", "server", "étatserver"]);
        assert_eq!(terms("ABCÉé"), ["abc", "éé", "abcéé"]);
        // the apostrophe separates; a lone ASCII letter is noise, a lone `è` is a word
        assert_eq!(terms("l'università è"), ["università", "è"]);
    }

    #[test]
    fn stemming_and_stopwords_are_ascii_only() {
        assert_eq!(terms("données"), ["données"]);
        assert_eq!(terms("donnees"), ["donnee"]);
        assert_eq!(terms("añadidos"), ["añadidos"]);
        assert_eq!(terms("Straßen größeren"), ["straßen", "größeren"]);
    }

    #[test]
    fn cjk_runs_become_bigrams() {
        assert_eq!(
            terms("数据库连接池"),
            ["数据", "据库", "库连", "连接", "接池"]
        );
        // a lone character stands for itself, and a pair is one bigram
        assert_eq!(terms("猫 狗 数据"), ["猫", "狗", "数据"]);
        // punctuation ends a run, so no bigram spans it
        assert_eq!(terms("数据库、连接池"), ["数据", "据库", "连接", "接池"]);
        assert_eq!(terms("ジョン・スミス"), ["ジョ", "ョン", "スミ", "ミス"]);
        // kanji, kana and Hangul pair up
        assert_eq!(
            terms("データベース接続"),
            ["デー", "ータ", "タベ", "ベー", "ース", "ス接", "接続"]
        );
        assert_eq!(
            terms("데이터베이스"),
            ["데이", "이터", "터베", "베이", "이스"]
        );
    }

    #[test]
    fn mixed_scripts_split_at_the_boundary() {
        assert_eq!(terms("数据库connection"), ["数据", "据库", "connection"]);
        assert_eq!(terms("connection数据库"), ["connection", "数据", "据库"]);
        // each side keeps its own rules: the whole name belongs to the ASCII piece
        assert_eq!(
            terms("数据库ConnectionPool"),
            ["数据", "据库", "connection", "pool", "connectionpool"]
        );
        assert_eq!(terms("foo_数据_bar"), ["foo", "数据", "bar"]);
        assert_eq!(terms("v2数据"), ["v2", "数据"]);
    }

    #[test]
    fn odd_unicode_never_panics() {
        let weird = "a\u{301}x ☃ 😀fooBar🙂 \u{200b}ЯдроКод ΑΒΓ_Δ ｆｕｌｌ１２ ﾊﾝｶｸ \u{0}nul \
                     한국어Test 日本語ひらがなカタカナ ǅungla İstanbul ΣΑΣ _ __ ǆ";
        assert!(!terms(weird).is_empty());
        for w in weird.split(' ') {
            terms(w);
        }
    }

    #[test]
    fn rare_term_beats_common() {
        let mut bodies = vec!["common alpha"; 9];
        bodies.push("zebra beta");
        let hits = index(&bodies).search("common zebra", 20);
        assert_eq!(ids(&hits), [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(hits[0].1 > hits[1].1);
    }

    #[test]
    fn symbol_beats_body_only_match() {
        let docs = [
            Doc {
                path: "",
                symbol: Some("remove"),
                body: "alpha beta gamma delta",
            },
            doc("remove beta gamma delta"),
            doc("epsilon zeta eta theta"),
        ];
        let hits = Index::build(docs).search("remove", 10);
        assert_eq!(ids(&hits), [0, 1]);
        assert!(hits[0].1 > hits[1].1);
    }

    #[test]
    fn path_matches_count() {
        let at = |path, body| Doc {
            path,
            symbol: None,
            body,
        };
        let docs = [
            at("src/pool.rs", "alpha beta gamma"),
            at("src/misc.rs", "pool beta gamma"),
            at("src/misc.rs", "delta beta gamma"),
        ];
        let idx = Index::build(docs);
        // a path hit weighs more than a body hit; the third doc has neither
        let hits = idx.search("pool", 10);
        assert_eq!(ids(&hits), [0, 1]);
        assert!(hits[0].1 > hits[1].1);
        // and a term only paths have is still found
        assert_eq!(ids(&idx.search("src", 10)), [0, 1, 2]);
    }

    #[test]
    fn shorter_doc_wins_an_equal_count() {
        let filler: Vec<_> = (0..30).map(|i| format!("filler{i}")).collect();
        let long = format!("target {}", filler.join(" "));
        let hits = index(&["target alpha", &long]).search("target", 10);
        assert_eq!(ids(&hits), [0, 1]);
        assert!(hits[0].1 > hits[1].1);
    }

    #[test]
    fn idf_stays_positive_when_every_doc_matches() {
        let hits = index(&["alpha beta", "alpha gamma"]).search("alpha", 10);
        assert_eq!(ids(&hits), [0, 1]);
        assert!(hits.iter().all(|h| h.1 > 0.0));
    }

    #[test]
    fn queries_without_overlap_find_nothing() {
        let idx = index(&["alpha beta", "gamma delta"]);
        for q in ["", "   ", "the of and", "zzz", "x y z", "!!! ???"] {
            assert!(idx.search(q, 10).is_empty(), "{q:?}");
        }
        let empty = Index::build(std::iter::empty());
        assert_eq!(empty.len(), 0);
        assert!(empty.search("alpha", 10).is_empty());
    }

    #[test]
    fn k_bounds() {
        let idx = index(&["alpha", "alpha beta", "gamma"]);
        assert_eq!(ids(&idx.search("alpha", 100)), [0, 1]);
        assert_eq!(ids(&idx.search("alpha", 1)), [0]);
        assert!(idx.search("alpha", 0).is_empty());
    }

    #[test]
    fn ties_break_by_doc_index() {
        let idx = index(&["alpha beta"; 30]);
        assert_eq!(ids(&idx.search("alpha", 30)), (0..30).collect::<Vec<_>>());
        // the cut falls inside a tie: the lowest indices stay
        assert_eq!(ids(&idx.search("alpha", 7)), (0..7).collect::<Vec<_>>());
        assert_eq!(idx.search("alpha beta", 7), idx.search("alpha beta", 7));
    }

    #[test]
    fn query_terms_count_once() {
        let idx = index(&["foo bar", "foo foo foo"]);
        assert_eq!(idx.search("foo foo FOO foos", 10), idx.search("foo", 10));
    }

    #[test]
    fn missing_fields_do_not_poison_scores() {
        // no paths, no symbols: their average length is 0
        let hits = index(&["alpha", "alpha beta"]).search("alpha", 10);
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().all(|h| h.1.is_finite() && h.1 > 0.0));
    }

    #[test]
    fn query_and_doc_forms_meet() {
        let idx = index(&[
            "fn close(&mut self, pool: &ConnectionPool)",
            "the connection_pool is idle",
            "unrelated words only",
        ]);
        for q in [
            "ConnectionPool",
            "connectionPool",
            "connection_pool",
            "connectionpool",
            "connection pools",
        ] {
            let mut got = ids(&idx.search(q, 10));
            got.sort();
            assert_eq!(got, [0, 1], "{q}");
        }
    }

    #[test]
    fn accented_words_match_themselves() {
        let idx = index(&[
            "Perché l'università è chiusa oggi",
            "Die Größe der Datei",
            "plain ascii words",
        ]);
        assert_eq!(ids(&idx.search("università", 10)), [0]);
        assert_eq!(ids(&idx.search("UNIVERSITÀ?", 10)), [0]);
        assert_eq!(ids(&idx.search("Perché", 10)), [0]);
        assert_eq!(ids(&idx.search("größe", 10)), [1]);
        // the unaccented spelling is another word
        assert!(idx.search("universita", 10).is_empty());
    }

    #[test]
    fn chinese_query_matches_through_bigrams() {
        let idx = index(&[
            "数据库连接池关闭失败",
            "用户登录验证",
            "文件系统读取错误",
            "数据库ConnectionPool",
        ]);
        assert_eq!(ids(&idx.search("连接池", 10)), [0]);
        assert_eq!(ids(&idx.search("验证用户", 10)), [1]);
        assert_eq!(ids(&idx.search("读取", 10)), [2]);
        // characters the docs hold, but never side by side
        assert!(idx.search("池数", 10).is_empty());
        // both scripts of a mixed run are searchable on their own
        assert_eq!(ids(&idx.search("connection", 10)), [3]);
        assert_eq!(ids(&idx.search("数据库", 10)).len(), 2);
    }

    struct Rng(u64);

    impl Rng {
        /// xorshift: deterministic randomness without a dependency
        fn below(&mut self, n: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % n as u64) as usize
        }
    }

    const VOCAB: [&str; 20] = [
        "connection",
        "pool",
        "close",
        "remove",
        "sequence",
        "decode",
        "cache",
        "token",
        "parse",
        "config",
        "server",
        "buffer",
        "stream",
        "error",
        "handle",
        "request",
        "università",
        "größe",
        "数据库",
        "连接池",
    ];

    fn caps(w: &str) -> String {
        let mut c = w.chars();
        c.next()
            .map(|f| f.to_uppercase().chain(c).collect())
            .unwrap_or_default()
    }

    /// Two vocabulary words as one identifier, in a random style, or apart.
    fn phrase(rng: &mut Rng) -> String {
        let (a, b) = (VOCAB[rng.below(20)], VOCAB[rng.below(20)]);
        match rng.below(5) {
            0 => a.to_string(),
            1 => format!("{a}_{b}"),
            2 => format!("{a}{}", caps(b)),
            3 => format!("{}{}", caps(a), caps(b)),
            _ => format!("{a} {b}"),
        }
    }

    fn corpus(n: usize) -> Vec<(String, Option<String>, String)> {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let mut docs: Vec<_> = (0..n)
            .map(|_| {
                let path = format!("src/{}/{}.rs", VOCAB[rng.below(20)], VOCAB[rng.below(20)]);
                let symbol = (rng.below(3) == 0).then(|| phrase(&mut rng));
                let words = 10 + rng.below(40);
                let body: Vec<_> = (0..words).map(|_| phrase(&mut rng)).collect();
                (path, symbol, body.join(" "))
            })
            .collect();
        docs[137].2.push_str(" QuantumFluxCapacitor");
        docs
    }

    /// BM25F straight from its definition, over `terms`, in f64: field
    /// weights 2/3/1, b 0.5/0.5/0.75, k1 1.2.
    fn naive(docs: &[Doc], query: &str) -> HashMap<usize, f64> {
        let (w, b, k1) = ([2.0, 3.0, 1.0], [0.5, 0.5, 0.75], 1.2);
        let fields: Vec<[Vec<String>; 3]> = docs
            .iter()
            .map(|d| [terms(d.path), terms(d.symbol.unwrap_or("")), terms(d.body)])
            .collect();
        let n = docs.len() as f64;
        let avg: Vec<f64> = (0..3)
            .map(|f| fields.iter().map(|d| d[f].len() as f64).sum::<f64>() / n)
            .collect();
        let mut query = terms(query);
        query.sort();
        query.dedup();
        let mut out = HashMap::new();
        for (i, d) in fields.iter().enumerate() {
            let mut score = 0.0;
            for t in &query {
                let df = fields
                    .iter()
                    .filter(|d| d.iter().any(|f| f.contains(t)))
                    .count() as f64;
                let tf: f64 = (0..3)
                    .filter(|&f| avg[f] > 0.0)
                    .map(|f| {
                        let tf = d[f].iter().filter(|x| *x == t).count() as f64;
                        w[f] * tf / (1.0 - b[f] + b[f] * d[f].len() as f64 / avg[f])
                    })
                    .sum();
                if tf > 0.0 {
                    score += (1.0 + (n - df + 0.5) / (df + 0.5)).ln() * tf / (k1 + tf);
                }
            }
            if score > 0.0 {
                out.insert(i, score);
            }
        }
        out
    }

    #[test]
    fn a_few_hundred_docs_score_as_defined() {
        let owned = corpus(300);
        let docs: Vec<Doc> = owned
            .iter()
            .map(|(p, s, b)| Doc {
                path: p,
                symbol: s.as_deref(),
                body: b,
            })
            .collect();
        let idx = Index::build(docs.iter().copied());
        assert_eq!(idx.len(), 300);
        for q in [
            "connection pool",
            "remove sequence after failed decode",
            "ConnectionPool close",
            "università größe",
            "数据库连接池",
            "cache_token parse",
            "config",
            "the of and",
        ] {
            let all = idx.search(q, docs.len());
            let want = naive(&docs, q);
            assert_eq!(all.len(), want.len(), "{q}");
            for &(i, s) in &all {
                let w = want[&i];
                assert!(
                    (f64::from(s) - w).abs() <= 1e-4 * w,
                    "{q}: doc {i}: {s} vs {w}"
                );
            }
            // best first, ties by doc index
            assert!(all
                .windows(2)
                .all(|p| p[0].1 > p[1].1 || (p[0].1 == p[1].1 && p[0].0 < p[1].0)));
            // top-k is the head of the full ranking, however the cut falls
            for k in [1, 5, 17, 100] {
                assert_eq!(idx.search(q, k), all[..k.min(all.len())], "{q} k={k}");
            }
        }
        assert_eq!(ids(&idx.search("quantum flux capacitor", 10)), [137]);
    }

    #[test]
    fn rrf_adds_reciprocal_ranks() {
        let fused = rrf(&[vec![1, 2, 3], vec![3, 1]], 60.0);
        // doc 1 holds ranks 1 and 2, doc 3 ranks 3 and 1, doc 2 rank 2 alone
        assert_eq!(ids(&fused), [1, 3, 2]);
        let want = [1.0 / 61.0 + 1.0 / 62.0, 1.0 / 61.0 + 1.0 / 63.0, 1.0 / 62.0];
        for (got, want) in fused.iter().zip(want) {
            assert!((got.1 - want).abs() < 1e-12, "{got:?} vs {want}");
        }
        // k flattens the head
        let flat = rrf(&[vec![1, 2]], 1e6);
        assert!((flat[0].1 - flat[1].1).abs() < 1e-6);
    }

    #[test]
    fn rrf_ties_go_to_the_lower_doc_index() {
        assert_eq!(ids(&rrf(&[vec![5], vec![3]], 60.0)), [3, 5]);
        // docs 1 and 2 hold ranks 1, 2, 3 in different rankings: an exact tie,
        // not a rounding race. At k = 2 adding them as (1, 2, 3) gives
        // 0.7833333333333332 and as (3, 1, 2) 0.7833333333333333.
        let fused = rrf(&[vec![1, 7, 2], vec![2, 1, 8], vec![9, 2, 1]], 2.0);
        assert_eq!(ids(&fused), [1, 2, 9, 7, 8]);
        assert_eq!(fused[0].1, fused[1].1);
    }

    #[test]
    fn rrf_of_nothing_and_of_one() {
        assert!(rrf(&[], 60.0).is_empty());
        assert!(rrf(&[vec![], vec![]], 60.0).is_empty());
        assert_eq!(ids(&rrf(&[vec![4, 2, 9]], 60.0)), [4, 2, 9]);
    }
}
