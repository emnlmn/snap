//! On-disk KV snapshots for `snap grep`: what the model computed reading a
//! chunk, kept so that a query decodes only its own question. Content-
//! addressed (the key hashes the exact prefix tokens, so a changed chunk,
//! path, template or prompt format is simply a different key), append-only
//! (a pack of blobs plus a log of fixed-size index records; a torn tail is
//! dropped on open) and bound to the engine that wrote it: a snapshot is
//! only valid for the same weights and KV layout, so a binding mismatch
//! refuses to open instead of feeding the model foreign memory.
//!
//! A cache may lose entries but must never serve wrong bytes, so every
//! ordering here is about that: a record reaches the index only after its
//! blob is durable, records are checksummed, and a compaction that dies
//! between its two renames is finished by the next open instead of leaving
//! an old index pointing into a new pack.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions, TryLockError};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, Receiver};
use std::thread::JoinHandle;

use anyhow::{anyhow, bail, Context, Result};

const BINDING: &str = "binding.txt";
const PACK: &str = "pack.bin";
const INDEX: &str = "index.bin";
const LOCK: &str = "lock";
const PACK_TMP: &str = "pack.tmp";
const INDEX_TMP: &str = "index.tmp";

/// index record: key u128 | off u64 | len u64 | tokens u32 | check u64, all
/// little-endian
const REC: usize = 44;
/// `check` (FNV-1a 64) covers every byte of the record but itself
const CHECKED: usize = REC - 8;

#[derive(Clone, Copy)]
struct Loc {
    off: u64,
    len: usize,
    tokens: u32,
}

/// The files and what they hold: the key map is rebuilt from the index on
/// every load, `end` and `idx_end` are how far each file is valid.
struct Disk {
    /// written at `end`
    pack: File,
    /// records written at `idx_end`
    index: File,
    /// positional reads only: no cursor to share between threads
    rd: File,
    map: HashMap<u128, Loc>,
    end: u64,
    idx_end: u64,
}

impl Disk {
    /// Open both files and cut what a crash left behind: a partial or bad
    /// record ends the index, and pack bytes past the last blob a record
    /// owns go too, so appends start clean. A key stored twice keeps its
    /// first record.
    fn load(dir: &Path) -> Result<Disk> {
        let (pack, mut index) = (open_rw(&dir.join(PACK))?, open_rw(&dir.join(INDEX))?);
        let mut raw = Vec::new();
        index
            .read_to_end(&mut raw)
            .context("cannot read the index")?;
        let pack_len = pack.metadata()?.len();
        let (mut map, mut idx_end, mut end) = (HashMap::new(), 0, 0);
        for r in raw.chunks_exact(REC) {
            let Some((key, l)) = decode(r, pack_len) else {
                break;
            };
            map.entry(key).or_insert(l);
            idx_end += REC as u64;
            end = end.max(l.off + l.len as u64);
        }
        index.set_len(idx_end)?;
        pack.set_len(end)?;
        Ok(Disk {
            rd: File::open(dir.join(PACK))?,
            pack,
            index,
            map,
            end,
            idx_end,
        })
    }
}

/// One store directory, held exclusively by this process while it lives.
pub struct Store {
    dir: PathBuf,
    disk: Disk,
    /// records of the puts not yet in the index file
    pending: Vec<u8>,
    /// last, so it is released only after the files above are closed
    _lock: File,
}

impl Store {
    /// Open or create the store in `dir`, bound to `binding` (model identity
    /// and KV layout, opaque here). Fails if `dir` was created under a
    /// different binding, holds snapshots with no binding on record, or is
    /// held by another process. Opening cuts what a crash left behind: a
    /// torn index record, pack bytes no record owns.
    pub fn open(dir: &Path, binding: &str) -> Result<Store> {
        fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
        let lock = open_rw(&dir.join(LOCK))?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                bail!("{} is in use by another process", dir.display())
            }
            Err(TryLockError::Error(e)) => {
                return Err(e).with_context(|| format!("cannot lock {}", dir.display()))
            }
        }
        let bound = dir.join(BINDING);
        match fs::read(&bound) {
            Ok(have) if have == binding.as_bytes() => {}
            Ok(have) => bail!(
                "{} holds snapshots for {:?}, not {binding:?}",
                dir.display(),
                String::from_utf8_lossy(&have)
            ),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                // snapshots nobody vouches for: refuse rather than adopt them
                if fs::metadata(dir.join(INDEX)).is_ok_and(|m| m.len() > 0) {
                    bail!(
                        "{} holds snapshots but no {BINDING}: cannot tell which engine wrote them",
                        dir.display()
                    );
                }
                // whole or absent, never torn: a half-written binding would
                // refuse its own store forever
                let tmp = dir.join("binding.tmp");
                let mut f = File::create(&tmp)?;
                f.write_all(binding.as_bytes())?;
                f.sync_all()?;
                fs::rename(&tmp, &bound)?;
            }
            Err(e) => return Err(e).with_context(|| format!("cannot read {}", bound.display())),
        }
        let (pt, it) = (dir.join(PACK_TMP), dir.join(INDEX_TMP));
        // a compaction that died between its renames left the new pack in
        // place and its index waiting here: finish it, the old index would
        // point into the wrong pack
        if it.exists() && !pt.exists() {
            fs::rename(&it, dir.join(INDEX))?;
        }
        let _ = fs::remove_file(&pt);
        let _ = fs::remove_file(&it);
        Ok(Store {
            dir: dir.into(),
            disk: Disk::load(dir)?,
            pending: Vec::new(),
            _lock: lock,
        })
    }

    /// Snapshots stored.
    pub fn len(&self) -> usize {
        self.disk.map.len()
    }

    /// Bytes the pack holds, dead entries included until `compact`.
    pub fn bytes(&self) -> u64 {
        self.disk.end
    }

    /// Tokens the snapshot under `key` holds, if one is stored.
    pub fn tokens(&self, key: u128) -> Option<usize> {
        self.disk.map.get(&key).map(|l| l.tokens as usize)
    }

    /// The snapshot under `key`, if one is stored; visible from its `put`
    /// on, synced or not. Reads are positional, so any number of threads
    /// may read at once.
    pub fn read(&self, key: u128) -> Result<Option<Vec<u8>>> {
        self.disk
            .map
            .get(&key)
            .map(|&l| read_blob(&self.disk.rd, l))
            .transpose()
    }

    /// Append a snapshot; a key already stored is left as is. Readable at
    /// once, durable after `sync`.
    pub fn put(&mut self, key: u128, tokens: usize, blob: &[u8]) -> Result<()> {
        if self.disk.map.contains_key(&key) {
            return Ok(());
        }
        let tokens = u32::try_from(tokens).context("token count past u32")?;
        let d = &mut self.disk;
        // `end` is the truth: a failed write can leave bytes past it, so
        // seek there every time instead of trusting the cursor
        d.pack.seek(SeekFrom::Start(d.end))?;
        d.pack
            .write_all(blob)
            .context("cannot append to the pack")?;
        let loc = Loc {
            off: d.end,
            len: blob.len(),
            tokens,
        };
        d.end += blob.len() as u64;
        d.map.insert(key, loc);
        record(&mut self.pending, key, loc);
        Ok(())
    }

    /// Make every put so far durable: the pack reaches disk before the
    /// index records that point into it. Dropping the store does the same,
    /// best effort.
    pub fn sync(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let d = &mut self.disk;
        d.pack.sync_data().context("cannot sync the pack")?;
        d.index.seek(SeekFrom::Start(d.idx_end))?;
        d.index
            .write_all(&self.pending)
            .context("cannot append to the index")?;
        d.index.sync_data().context("cannot sync the index")?;
        d.idx_end += self.pending.len() as u64;
        self.pending.clear();
        Ok(())
    }

    /// Read `keys` in order on a background thread, keeping up to `depth`
    /// blobs queued ahead of the consumer (the reader holds one more while
    /// it waits). The pack as it is now is what gets read: a key stored
    /// later yields Ok(None), a `compact` meanwhile changes nothing, and the
    /// `Prefetch` does not borrow the store.
    pub fn prefetch(&self, keys: Vec<u128>, depth: usize) -> Prefetch {
        let asked: Vec<_> = keys
            .into_iter()
            .map(|k| (k, self.disk.map.get(&k).copied()))
            .collect();
        // opened here, not in the thread: the locations above belong to this
        // file, and a compaction landing before the thread starts would
        // otherwise hand it the new pack with the old offsets
        let path = self.dir.join(PACK);
        let pack = File::open(&path).with_context(|| format!("cannot open {}", path.display()));
        let (tx, rx) = sync_channel(depth.max(1));
        let reader = std::thread::spawn(move || {
            for (key, loc) in asked {
                let got = match (loc, &pack) {
                    (None, _) => Ok(None),
                    (Some(l), Ok(f)) => read_blob(f, l).map(Some),
                    (Some(_), Err(e)) => Err(anyhow!("{e:#}")),
                };
                if tx.send((key, got)).is_err() {
                    return;
                }
            }
        });
        Prefetch {
            rx: Some(rx),
            reader: Some(reader),
        }
    }

    /// Rewrite pack and index keeping only `live` keys. Does nothing when
    /// every stored key is live and the pack holds no dead bytes.
    pub fn compact(&mut self, live: &HashSet<u128>) -> Result<()> {
        self.sync()?;
        let mut keep: Vec<(u128, Loc)> = self
            .disk
            .map
            .iter()
            .filter(|(k, _)| live.contains(*k))
            .map(|(&k, &l)| (k, l))
            .collect();
        // nothing dead, nothing to give back: don't copy gigabytes for it
        let live_bytes: u64 = keep.iter().map(|(_, l)| l.len as u64).sum();
        if keep.len() == self.disk.map.len() && live_bytes == self.disk.end {
            return Ok(());
        }
        keep.sort_by_key(|(_, l)| l.off);
        let (pt, it) = (self.dir.join(PACK_TMP), self.dir.join(INDEX_TMP));
        let mut pack =
            File::create(&pt).with_context(|| format!("cannot create {}", pt.display()))?;
        let (mut idx, mut off) = (Vec::new(), 0);
        for (key, l) in keep {
            pack.write_all(&read_blob(&self.disk.rd, l)?)
                .context("cannot write the compacted pack")?;
            record(&mut idx, key, Loc { off, ..l });
            off += l.len as u64;
        }
        pack.sync_all()?;
        let mut index =
            File::create(&it).with_context(|| format!("cannot create {}", it.display()))?;
        index
            .write_all(&idx)
            .context("cannot write the compacted index")?;
        index.sync_all()?;
        // pack first: `open` finishes a swap that died between the renames
        // from index.tmp, which only works in this order
        fs::rename(&pt, self.dir.join(PACK)).context("cannot swap in the compacted pack")?;
        fs::rename(&it, self.dir.join(INDEX)).context("cannot swap in the compacted index")?;
        self.disk = Disk::load(&self.dir)?;
        Ok(())
    }
}

impl Drop for Store {
    fn drop(&mut self) {
        let _ = self.sync();
    }
}

type Fetched = (u128, Result<Option<Vec<u8>>>);

/// Blobs read ahead by `Store::prefetch`, in the order asked; a key the
/// store lacked when asked yields Ok(None). Dropping it stops the reader
/// and waits for it: no thread outlives its handle, and an early exit
/// costs at most the blob in flight.
pub struct Prefetch {
    rx: Option<Receiver<Fetched>>,
    reader: Option<JoinHandle<()>>,
}

impl Iterator for Prefetch {
    type Item = Fetched;

    fn next(&mut self) -> Option<Self::Item> {
        self.rx.as_ref()?.recv().ok()
    }
}

impl Drop for Prefetch {
    fn drop(&mut self) {
        // hang up before joining: a reader blocked on a full channel wakes
        // with an error and exits, otherwise the join would wait forever
        drop(self.rx.take());
        if let Some(t) = self.reader.take() {
            let _ = t.join();
        }
    }
}

fn open_rw(path: &Path) -> Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .with_context(|| format!("cannot open {}", path.display()))
}

fn read_blob(f: &File, l: Loc) -> Result<Vec<u8>> {
    let mut b = vec![0; l.len];
    read_at(f, &mut b, l.off)
        .with_context(|| format!("cannot read {} bytes at {} of {PACK}", l.len, l.off))?;
    Ok(b)
}

#[cfg(unix)]
fn read_at(f: &File, buf: &mut [u8], off: u64) -> io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(f, buf, off)
}

#[cfg(windows)]
fn read_at(f: &File, mut buf: &mut [u8], mut off: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match f.seek_read(buf, off) {
            Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => {
                buf = &mut buf[n..];
                off += n as u64;
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Append the index record for `key` at `l`.
fn record(out: &mut Vec<u8>, key: u128, l: Loc) {
    let at = out.len();
    out.extend_from_slice(&key.to_le_bytes());
    out.extend_from_slice(&l.off.to_le_bytes());
    out.extend_from_slice(&(l.len as u64).to_le_bytes());
    out.extend_from_slice(&l.tokens.to_le_bytes());
    let check = fnv64(&out[at..]);
    out.extend_from_slice(&check.to_le_bytes());
}

/// One record of REC bytes, if its checksum holds and its blob lies inside
/// a pack of `pack_len` bytes.
fn decode(r: &[u8], pack_len: u64) -> Option<(u128, Loc)> {
    let le = |a: usize, b: usize| {
        let mut x = [0; 16];
        x[..b - a].copy_from_slice(&r[a..b]);
        u128::from_le_bytes(x)
    };
    if fnv64(&r[..CHECKED]) != le(CHECKED, REC) as u64 {
        return None;
    }
    let (off, len) = (le(16, 24) as u64, le(24, 32) as u64);
    let loc = Loc {
        off,
        len: usize::try_from(len).ok()?,
        tokens: le(32, 36) as u32,
    };
    (off.checked_add(len)? <= pack_len).then_some((le(0, 16), loc))
}

fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

/// FNV-1a over 128 bits: stable across builds and platforms, which is all
/// a content address needs (std's hasher promises neither).
pub fn key(bytes: &[u8]) -> u128 {
    bytes
        .iter()
        .fold(0x6c62272e07bb014262b821756295c58d, |h, &b| {
            (h ^ b as u128).wrapping_mul(0x0000000001000000000000000000013B)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    /// Scratch dir that goes away with the test, pass or fail.
    struct Tmp(PathBuf);

    impl Tmp {
        fn new(name: &str) -> Tmp {
            let dir =
                std::env::temp_dir().join(format!("snap-kvstore-{}-{name}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            Tmp(dir)
        }

        fn at(&self, file: &str) -> PathBuf {
            self.0.join(file)
        }

        fn open(&self) -> Store {
            Store::open(&self.0, "m1").unwrap()
        }

        fn size(&self, file: &str) -> u64 {
            fs::metadata(self.at(file)).unwrap().len()
        }
    }

    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// `n` bytes that differ per seed and per position.
    fn blob(seed: u8, n: usize) -> Vec<u8> {
        (0..n)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
            .collect()
    }

    /// Die before `sync`: what only lived in memory never reaches disk.
    fn crash(mut s: Store) {
        s.pending.clear();
    }

    fn append(path: PathBuf, bytes: &[u8]) {
        let mut f = OpenOptions::new().append(true).open(path).unwrap();
        f.write_all(bytes).unwrap();
    }

    /// Blobs 1..=3 of 500 bytes, synced, store closed: the baseline the
    /// damage tests start from. Returns the pack length.
    fn three(t: &Tmp) -> u64 {
        let mut s = t.open();
        for k in 1..=3 {
            s.put(k, k as usize, &blob(k as u8, 500)).unwrap();
        }
        s.sync().unwrap();
        s.bytes()
    }

    /// Keys 1..=3 as `three` wrote them, minus those in `gone`.
    fn assert_three(s: &Store, gone: &[u128]) {
        for k in 1..=3u128 {
            let want = (!gone.contains(&k)).then(|| blob(k as u8, 500));
            assert_eq!(s.read(k).unwrap(), want, "key {k}");
            assert_eq!(s.tokens(k), want.map(|_| k as usize), "tokens of {k}");
        }
    }

    #[test]
    fn put_read_roundtrip() {
        let t = Tmp::new("roundtrip");
        let mut s = t.open();
        let big = blob(7, 3 << 20);
        s.put(1, 10, &blob(1, 1000)).unwrap();
        s.put(2, 0, &[]).unwrap();
        s.put(3, 4096, &big).unwrap();
        // no sync yet: a put is readable at once
        assert_eq!(s.read(1).unwrap().unwrap(), blob(1, 1000));
        assert_eq!(s.read(2).unwrap().unwrap(), Vec::<u8>::new());
        assert_eq!(s.read(3).unwrap().unwrap(), big);
        assert_eq!(s.read(4).unwrap(), None);
        assert_eq!((s.len(), s.bytes()), (3, 1000 + (3 << 20)));
        let tokens: Vec<_> = (1..=4).map(|k| s.tokens(k)).collect();
        assert_eq!(tokens, [Some(10), Some(0), Some(4096), None]);
    }

    #[test]
    fn fresh_store_layout() {
        let t = Tmp::new("fresh");
        let dir = t.at("deep/er");
        let s = Store::open(&dir, "model-a/f16").unwrap();
        assert_eq!((s.len(), s.bytes()), (0, 0));
        assert_eq!(fs::read(dir.join("binding.txt")).unwrap(), b"model-a/f16");
        for f in ["pack.bin", "index.bin", "lock"] {
            assert_eq!(fs::metadata(dir.join(f)).unwrap().len(), 0, "{f}");
        }
    }

    #[test]
    fn index_record_layout() {
        let t = Tmp::new("layout");
        let mut s = t.open();
        let k = 0x0102030405060708090a0b0c0d0e0f10u128;
        s.put(k, 7, b"xyz").unwrap();
        s.put(2, 9, b"hello").unwrap();
        // records wait in memory until a sync
        assert_eq!(t.size("index.bin"), 0);
        s.sync().unwrap();
        s.sync().unwrap(); // nothing pending: nothing appended again
        assert_eq!(fs::read(t.at("pack.bin")).unwrap(), b"xyzhello");
        let idx = fs::read(t.at("index.bin")).unwrap();
        assert_eq!(idx.len(), 2 * 44);
        let (a, b) = idx.split_at(44);
        assert_eq!(a[..16], k.to_le_bytes());
        assert_eq!(a[16..24], 0u64.to_le_bytes());
        assert_eq!(a[24..32], 3u64.to_le_bytes());
        assert_eq!(a[32..36], 7u32.to_le_bytes());
        assert_eq!(a[36..], fnv64(&a[..36]).to_le_bytes());
        assert_eq!(b[..16], 2u128.to_le_bytes());
        assert_eq!(b[16..24], 3u64.to_le_bytes());
        assert_eq!(b[24..32], 5u64.to_le_bytes());
        assert_eq!(b[32..36], 9u32.to_le_bytes());
        assert_eq!(b[36..], fnv64(&b[..36]).to_le_bytes());
    }

    #[test]
    fn synced_puts_survive_a_crash() {
        let t = Tmp::new("crash");
        let mut s = t.open();
        for k in 0..500u128 {
            s.put(k, k as usize, &blob(k as u8, 50 + k as usize))
                .unwrap();
        }
        s.sync().unwrap();
        s.put(999, 1, b"never synced").unwrap();
        crash(s);
        let mut s = t.open();
        assert_eq!(s.len(), 500);
        for k in 0..500u128 {
            assert_eq!(s.read(k).unwrap().unwrap(), blob(k as u8, 50 + k as usize));
            assert_eq!(s.tokens(k), Some(k as usize));
        }
        // its blob is still in the pack; open cut it, so appends start clean
        assert_eq!(s.read(999).unwrap(), None);
        s.put(999, 2, b"again").unwrap();
        s.sync().unwrap();
        drop(s);
        let s = t.open();
        assert_eq!(s.len(), 501);
        assert_eq!(s.read(999).unwrap().unwrap(), b"again");
        assert_eq!(s.read(0).unwrap().unwrap(), blob(0, 50));
    }

    #[test]
    fn drop_syncs() {
        let t = Tmp::new("drop");
        let mut s = t.open();
        s.put(5, 3, b"abc").unwrap();
        drop(s);
        let s = t.open();
        assert_eq!(s.read(5).unwrap().unwrap(), b"abc");
        assert_eq!(s.tokens(5), Some(3));
    }

    #[test]
    fn binding_mismatch_is_refused() {
        let t = Tmp::new("binding");
        drop(Store::open(&t.0, "model-a/f16").unwrap());
        let e = Store::open(&t.0, "model-b/q8").err().unwrap().to_string();
        assert!(e.contains("model-a/f16") && e.contains("model-b/q8"), "{e}");
        // the refusal left the store alone and released its lock
        assert!(Store::open(&t.0, "model-a/f16").is_ok());
        assert_eq!(fs::read(t.at("binding.txt")).unwrap(), b"model-a/f16");
    }

    #[test]
    fn snapshots_without_a_binding_are_refused() {
        let t = Tmp::new("unbound");
        three(&t);
        fs::remove_file(t.at("binding.txt")).unwrap();
        let e = Store::open(&t.0, "m1").err().unwrap().to_string();
        assert!(e.contains("binding.txt"), "{e}");
        assert!(!t.at("binding.txt").exists());
        assert_eq!(t.size("index.bin"), 3 * REC as u64);
        // with nothing stored there is nothing to vouch for: a new store
        let empty = Tmp::new("unbound-empty");
        drop(empty.open());
        fs::remove_file(empty.at("binding.txt")).unwrap();
        assert_eq!(empty.open().len(), 0);
        assert_eq!(fs::read(empty.at("binding.txt")).unwrap(), b"m1");
    }

    #[test]
    fn second_open_fails_while_the_first_lives() {
        let t = Tmp::new("lock");
        let a = t.open();
        let e = Store::open(&t.0, "m1").err().unwrap().to_string();
        assert!(e.contains("in use"), "{e}");
        drop(a);
        assert!(Store::open(&t.0, "m1").is_ok());
    }

    #[test]
    fn torn_tail_is_dropped() {
        let t = Tmp::new("torn");
        let end = three(&t);
        // a crash mid-append: half a record in the index, and in the pack a
        // blob's worth of bytes that no record owns
        let mut rec = Vec::new();
        record(
            &mut rec,
            4,
            Loc {
                off: end,
                len: 500,
                tokens: 4,
            },
        );
        append(t.at("index.bin"), &rec[..20]);
        append(t.at("pack.bin"), &blob(4, 500));
        let mut s = t.open();
        assert_eq!((s.len(), s.bytes()), (3, end));
        assert_eq!(t.size("index.bin"), 3 * REC as u64);
        assert_eq!(t.size("pack.bin"), end);
        assert_three(&s, &[]);
        s.put(4, 4, &blob(4, 500)).unwrap();
        s.sync().unwrap();
        drop(s);
        let s = t.open();
        assert_eq!(s.len(), 4);
        assert_eq!(s.read(4).unwrap().unwrap(), blob(4, 500));
        assert_three(&s, &[]);
    }

    #[test]
    fn corrupt_last_record_is_dropped() {
        let t = Tmp::new("check");
        three(&t);
        let mut idx = fs::read(t.at("index.bin")).unwrap();
        *idx.last_mut().unwrap() ^= 0x80; // inside the last record's check
        fs::write(t.at("index.bin"), &idx).unwrap();
        let mut s = t.open();
        assert_eq!((s.len(), s.bytes()), (2, 1000));
        assert_eq!(t.size("pack.bin"), 1000);
        assert_three(&s, &[3]);
        s.put(3, 3, &blob(3, 500)).unwrap();
        s.sync().unwrap();
        drop(s);
        assert_three(&t.open(), &[]);
    }

    #[test]
    fn corruption_ends_the_index_there() {
        let t = Tmp::new("middle");
        three(&t);
        let mut idx = fs::read(t.at("index.bin")).unwrap();
        idx[REC + 3] ^= 1; // inside the second record's key
        fs::write(t.at("index.bin"), &idx).unwrap();
        let s = t.open();
        assert_eq!((s.len(), s.bytes()), (1, 500));
        assert_three(&s, &[2, 3]);
    }

    #[test]
    fn record_past_the_pack_end_is_dropped() {
        let t = Tmp::new("short-pack");
        let end = three(&t);
        let pack = OpenOptions::new()
            .write(true)
            .open(t.at("pack.bin"))
            .unwrap();
        pack.set_len(end - 1).unwrap();
        drop(pack);
        let s = t.open();
        assert_eq!((s.len(), s.bytes()), (2, 1000));
        assert_three(&s, &[3]);
    }

    #[test]
    fn duplicate_put_is_a_no_op() {
        let t = Tmp::new("dup");
        let mut s = t.open();
        s.put(9, 5, b"first").unwrap();
        s.put(9, 7, b"second!").unwrap();
        assert_eq!((s.len(), s.bytes(), s.tokens(9)), (1, 5, Some(5)));
        assert_eq!(s.read(9).unwrap().unwrap(), b"first");
        s.sync().unwrap();
        assert_eq!(t.size("index.bin"), REC as u64);
        assert_eq!(fs::read(t.at("pack.bin")).unwrap(), b"first");
    }

    #[test]
    fn first_record_wins_on_disk() {
        let t = Tmp::new("first");
        let mut s = t.open();
        s.put(1, 1, b"aaaa").unwrap();
        s.put(2, 2, b"bbbb").unwrap();
        drop(s);
        // a second record for key 1 that points at key 2's bytes
        let mut rec = Vec::new();
        record(
            &mut rec,
            1,
            Loc {
                off: 4,
                len: 4,
                tokens: 2,
            },
        );
        append(t.at("index.bin"), &rec);
        let s = t.open();
        assert_eq!(s.len(), 2);
        assert_eq!(s.read(1).unwrap().unwrap(), b"aaaa");
        assert_eq!(s.tokens(1), Some(1));
        assert_eq!(s.read(2).unwrap().unwrap(), b"bbbb");
    }

    #[test]
    fn read_of_a_shrunk_pack_is_an_error() {
        let t = Tmp::new("shrunk");
        let mut s = t.open();
        s.put(1, 1, &blob(1, 1000)).unwrap();
        s.sync().unwrap();
        let pack = OpenOptions::new()
            .write(true)
            .open(t.at("pack.bin"))
            .unwrap();
        pack.set_len(10).unwrap();
        assert!(s.read(1).is_err());
    }

    #[test]
    fn tokens_past_u32_are_refused() {
        let t = Tmp::new("tokens");
        let mut s = t.open();
        s.put(1, u32::MAX as usize, b"x").unwrap();
        assert_eq!(s.tokens(1), Some(u32::MAX as usize));
        #[cfg(target_pointer_width = "64")]
        {
            assert!(s.put(2, u32::MAX as usize + 1, b"y").is_err());
            assert_eq!((s.len(), s.bytes()), (1, 1));
        }
    }

    #[test]
    fn concurrent_reads_share_no_cursor() {
        let t = Tmp::new("threads");
        let mut s = t.open();
        let blobs: Vec<_> = (0..8u8).map(|k| blob(k, 64 << 10)).collect();
        for (k, b) in blobs.iter().enumerate() {
            s.put(k as u128, 1, b).unwrap();
        }
        std::thread::scope(|sc| {
            for n in 0..4 {
                let (s, blobs) = (&s, &blobs);
                sc.spawn(move || {
                    for i in 0..200 {
                        let k = (i * 3 + n) % 8;
                        assert_eq!(s.read(k as u128).unwrap().unwrap(), blobs[k]);
                    }
                });
            }
        });
    }

    #[test]
    fn prefetch_yields_in_order() {
        let t = Tmp::new("prefetch");
        let mut s = t.open();
        for k in 0..20u128 {
            s.put(k, 1, &blob(k as u8, 1000 * (k as usize + 1)))
                .unwrap();
        }
        // 100 is unknown, 7 and 100 repeat: asked order, asked count
        let asked = vec![5, 100, 0, 19, 100, 7, 7];
        let want: Vec<_> = asked.iter().map(|&k| (k, s.read(k).unwrap())).collect();
        assert_eq!(want[1], (100, None));
        for depth in [0, 1, 2, 64] {
            let got: Vec<_> = s
                .prefetch(asked.clone(), depth)
                .map(|(k, r)| (k, r.unwrap()))
                .collect();
            assert_eq!(got, want, "depth {depth}");
        }
        assert_eq!(s.prefetch(vec![], 4).count(), 0);
    }

    #[test]
    fn prefetch_does_not_borrow_the_store() {
        let t = Tmp::new("outlive");
        let mut s = t.open();
        for k in 0..4u128 {
            s.put(k, 1, &blob(k as u8, 4096)).unwrap();
        }
        let p = s.prefetch((0..4).collect(), 1);
        drop(s);
        let got: Vec<_> = p.map(|(k, r)| (k, r.unwrap().unwrap())).collect();
        let want: Vec<_> = (0..4u128).map(|k| (k, blob(k as u8, 4096))).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn dropping_a_prefetch_early_does_not_hang() {
        let t = Tmp::new("early");
        let mut s = t.open();
        for k in 0..32u128 {
            s.put(k, 1, &blob(k as u8, 1 << 20)).unwrap();
        }
        let untouched = s.prefetch((0..32).collect(), 1);
        let mut one = s.prefetch((0..32).collect(), 1);
        assert_eq!(one.next().unwrap().1.unwrap().unwrap(), blob(0, 1 << 20));
        // both readers fill their channel and block on a full one: the state
        // a hangup has to get them out of. The drops run on a thread so a
        // hang fails the test instead of stalling the whole run.
        let (done, waiting) = mpsc::channel();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            drop(untouched);
            drop(one);
            let _ = done.send(());
        });
        waiting
            .recv_timeout(Duration::from_secs(30))
            .expect("dropping a prefetch hung");
    }

    #[test]
    fn compact_keeps_live_and_shrinks() {
        let t = Tmp::new("compact");
        let mut s = t.open();
        for k in 0..6u128 {
            s.put(k, k as usize + 1, &blob(k as u8, 10_000)).unwrap();
        }
        s.sync().unwrap();
        // key 6 is not synced yet: compacting must not lose it
        s.put(6, 7, &blob(6, 10_000)).unwrap();
        let before = s.bytes();
        // 42 was never stored: ignored
        s.compact(&HashSet::from([1, 3, 6, 42])).unwrap();
        assert_eq!((s.len(), s.bytes()), (3, 30_000));
        assert!(s.bytes() < before);
        for k in [0, 2, 4, 5] {
            assert_eq!((s.read(k).unwrap(), s.tokens(k)), (None, None));
        }
        for k in [1, 3, 6] {
            assert_eq!(s.read(k).unwrap().unwrap(), blob(k as u8, 10_000));
            assert_eq!(s.tokens(k), Some(k as usize + 1));
        }
        assert_eq!((t.size("pack.bin"), t.size("index.bin")), (30_000, 3 * 44));
        assert!(!t.at("pack.tmp").exists() && !t.at("index.tmp").exists());
        // still appendable, and all of it survives a reopen
        s.put(7, 8, b"after").unwrap();
        drop(s);
        let s = t.open();
        assert_eq!((s.len(), s.bytes()), (4, 30_005));
        for k in [1, 3, 6] {
            assert_eq!(s.read(k).unwrap().unwrap(), blob(k as u8, 10_000));
        }
        assert_eq!(s.read(7).unwrap().unwrap(), b"after");
        assert_eq!(s.read(0).unwrap(), None);
    }

    /// Identity of a file across renames; 0 where the platform has none handy.
    fn inode(path: PathBuf) -> u64 {
        #[cfg(unix)]
        {
            std::os::unix::fs::MetadataExt::ino(&fs::metadata(path).unwrap())
        }
        #[cfg(not(unix))]
        {
            let _ = path;
            0
        }
    }

    #[test]
    fn compact_to_nothing_and_with_nothing_dead() {
        let t = Tmp::new("compact-edges");
        let mut s = t.open();
        for k in 1..=3 {
            s.put(k, 1, &blob(k as u8, 500)).unwrap();
        }
        // everything live: nothing to give back, so no rewrite
        let before = inode(t.at("pack.bin"));
        s.compact(&HashSet::from([1, 2, 3, 4])).unwrap();
        assert_eq!(inode(t.at("pack.bin")), before);
        assert_eq!((s.len(), s.bytes()), (3, 1500));
        s.compact(&HashSet::new()).unwrap();
        assert_eq!((s.len(), s.bytes()), (0, 0));
        assert_eq!((t.size("pack.bin"), t.size("index.bin")), (0, 0));
        // dead keys with no bytes in the pack still go
        s.put(8, 1, &[]).unwrap();
        s.put(9, 2, &[]).unwrap();
        s.compact(&HashSet::from([9])).unwrap();
        assert_eq!((s.len(), s.tokens(8), s.tokens(9)), (1, None, Some(2)));
        s.put(10, 1, b"fresh").unwrap();
        drop(s);
        let s = t.open();
        assert_eq!(s.len(), 2);
        assert_eq!(s.read(10).unwrap().unwrap(), b"fresh");
    }

    #[cfg(unix)]
    #[test]
    fn prefetch_reads_the_pack_it_was_asked_on() {
        let t = Tmp::new("prefetch-compact");
        let mut s = t.open();
        for k in 0..6u128 {
            s.put(k, 1, &blob(k as u8, 10_000)).unwrap();
        }
        // asked before the compaction, consumed after it: the old pack with
        // the old offsets, never the new pack with them
        let p = s.prefetch((0..6).collect(), 1);
        s.compact(&HashSet::from([1, 3])).unwrap();
        assert_eq!(s.len(), 2);
        let got: Vec<_> = p.map(|(k, r)| (k, r.unwrap().unwrap())).collect();
        let want: Vec<_> = (0..6u128).map(|k| (k, blob(k as u8, 10_000))).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn compaction_dead_between_its_renames_is_finished() {
        // compact a twin, then lay its files over the original the way a
        // death after the first rename leaves them: new pack in place, its
        // index still named index.tmp, the old index still index.bin
        let (t, twin) = (Tmp::new("roll"), Tmp::new("roll-twin"));
        three(&t);
        three(&twin);
        let mut s = twin.open();
        s.compact(&HashSet::from([2])).unwrap();
        drop(s);
        fs::copy(twin.at("pack.bin"), t.at("pack.bin")).unwrap();
        fs::copy(twin.at("index.bin"), t.at("index.tmp")).unwrap();
        let s = t.open();
        assert_eq!((s.len(), s.bytes()), (1, 500));
        assert_eq!(s.read(2).unwrap().unwrap(), blob(2, 500));
        assert_three(&s, &[1, 3]);
        assert!(!t.at("index.tmp").exists());
    }

    #[test]
    fn compaction_dead_before_its_renames_is_discarded() {
        let t = Tmp::new("unrolled");
        three(&t);
        // both temp files exist: nothing was swapped, the old pair stands
        fs::write(t.at("pack.tmp"), b"half a pack").unwrap();
        fs::write(t.at("index.tmp"), blob(0, REC)).unwrap();
        let s = t.open();
        assert_eq!(s.len(), 3);
        assert_three(&s, &[]);
        assert!(!t.at("pack.tmp").exists() && !t.at("index.tmp").exists());
    }

    #[test]
    fn handles_move_between_threads() {
        fn send<T: Send>() {}
        send::<Store>();
        send::<Prefetch>();
    }

    #[test]
    fn key_vectors() {
        // FNV-1a 128 reference values, computed independently of this code
        assert_eq!(key(b""), 0x6c62272e07bb014262b821756295c58d);
        assert_eq!(key(b"a"), 0xd228cb696f1a8caf78912b704e4a8964);
        assert_eq!(key(b"foobar"), 0x343e1662793c64bf6f0d3597ba446f18);
        let inputs: [&[u8]; 7] = [b"", b"a", b"b", b"ab", b"ba", b"\0", b"\0\0"];
        let keys: HashSet<_> = inputs.iter().map(|b| key(b)).collect();
        assert_eq!(keys.len(), inputs.len());
    }

    #[test]
    fn record_check_vectors() {
        assert_eq!(fnv64(b""), 0xcbf29ce484222325);
        assert_eq!(fnv64(b"a"), 0xaf63dc4c8601ec8c);
        assert_eq!(fnv64(b"foobar"), 0x85944171f73967e8);
    }
}
