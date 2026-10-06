//! First-stage recall for `snap grep`: BM25F over code-aware terms. The
//! model reranks; this stage only has to put the right chunks in the first
//! wave, cheaply and without a model. Identifiers split the way people
//! paraphrase them (`ConnectionPool` → connection, pool, and the whole
//! identifier), paths and declared names weigh more than bodies, and
//! several rankings fuse by reciprocal rank.
#![allow(dead_code)] // wired by grep.rs

/// Terms of a text, queries and documents alike: identifiers whole and
/// split at `_`, `-`, `::`, `.`, `/` and camelCase humps, lowercased,
/// lightly stemmed, English function words dropped.
pub fn terms(text: &str) -> Vec<String> {
    let _ = text;
    todo!()
}

/// One searchable unit, in the fields BM25F weighs separately.
#[derive(Debug, Clone, Copy)]
pub struct Doc<'a> {
    pub path: &'a str,
    pub symbol: Option<&'a str>,
    pub body: &'a str,
}

/// In-memory inverted index over a corpus, rebuilt per run.
pub struct Index {
    _todo: (),
}

impl Index {
    pub fn build<'a>(docs: impl IntoIterator<Item = Doc<'a>>) -> Index {
        let _ = docs.into_iter();
        todo!()
    }

    pub fn len(&self) -> usize {
        todo!()
    }

    /// Top `k` docs for `query`, best first: (doc index, score > 0), ties
    /// broken by doc index. Docs sharing no term with the query never show.
    pub fn search(&self, query: &str, k: usize) -> Vec<(usize, f32)> {
        let _ = (query, k);
        todo!()
    }
}

/// Reciprocal rank fusion: score(d) = Σ 1 / (k + rank), rank 1-based, over
/// the rankings that hold d. Best first, ties broken by doc index.
pub fn rrf(rankings: &[Vec<usize>], k: f64) -> Vec<(usize, f64)> {
    let _ = (rankings, k);
    todo!()
}
