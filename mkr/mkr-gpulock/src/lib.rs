//! One GPU job at a time (docs/design.md: The GPU lock; the Python side is mkmmd/gpulock.py). The card's memory is shared
//! by every job on it and what does not fit spills into system memory, which cannot be swapped, so a process that opens
//! the GPU calls [`hold`] first: an exclusive `flock` on `$MK_GPU_LOCK`, else `$MK_CACHE/gpu.lock`, else
//! `~/.cache/mk/gpu.lock`, waited for in turn and kept until the process exits (the kernel releases it however the
//! process ends). Jobs are served first come first served: one that finds the GPU busy (or others already waiting) puts
//! a ticket `<ns>-<pid>` into the queue folder beside the lock file (`gpu.lock.queue/`) and only the oldest live ticket
//! waits on the lock itself; a dead job's ticket is dropped by whoever sees it. Inside a job that already holds it
//! (`MK_GPU_LOCK_HELD=1`: mk render, mk post, `mk gpu run`, ...) it does nothing.

use std::ffi::OsString;
use std::fs::{File, OpenOptions, TryLockError};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Set to `1` by a job that holds the lock, for its children.
pub const ENV_HELD: &str = "MK_GPU_LOCK_HELD";
/// The lock file, when set.
pub const ENV_PATH: &str = "MK_GPU_LOCK";

static HELD: Mutex<Option<File>> = Mutex::new(None);

/// The lock file: `$MK_GPU_LOCK`, else `<cache>/gpu.lock` with the cache `$MK_CACHE` or `~/.cache/mk` (as
/// mkmmd.config.cache_dir).
pub fn path() -> PathBuf {
    resolve(std::env::var_os(ENV_PATH), std::env::var_os("MK_CACHE"), std::env::var_os("HOME"))
}

fn resolve(lock: Option<OsString>, cache: Option<OsString>, home: Option<OsString>) -> PathBuf {
    let home = PathBuf::from(home.unwrap_or_default());
    let expand = |p: OsString| -> PathBuf {
        let p = PathBuf::from(p);
        match p.strip_prefix("~") {
            Ok(rest) => home.join(rest),
            Err(_) => p,
        }
    };
    if let Some(p) = lock.filter(|p| !p.is_empty()) {
        return expand(p);
    }
    let cache = cache.filter(|p| !p.is_empty()).map(expand).unwrap_or_else(|| home.join(".cache").join("mk"));
    cache.join("gpu.lock")
}

/// Take the GPU for the rest of the process (see the crate doc), waiting for it in turn; says on stderr who it waits for.
pub fn hold() -> io::Result<()> {
    if std::env::var_os(ENV_HELD).is_some_and(|v| v == "1") {
        return Ok(());
    }
    let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
    if held.is_some() {
        return Ok(());
    }
    let path = path();
    let mut file = acquire_in_turn(&path)?;
    let what: Vec<String> = std::env::args().collect();
    let mut what = what.join(" ");
    what.truncate(400);
    file.set_len(0)?;
    file.seek(SeekFrom::Start(0))?;
    writeln!(file, "{} {}", std::process::id(), what)?;
    *held = Some(file);
    Ok(())
}

/// How often a job in the queue looks whether it is first.
const POLL: Duration = Duration::from_millis(200);

fn queue_of(path: &Path) -> PathBuf {
    let mut q = path.as_os_str().to_owned();
    q.push(".queue");
    PathBuf::from(q)
}

/// The tickets of the jobs waiting, oldest first; a dead job's ticket is removed.
fn line(queue: &Path) -> Vec<String> {
    let mut out: Vec<(u128, u32, String)> = Vec::new();
    let Ok(entries) = std::fs::read_dir(queue) else { return Vec::new() };
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some((ns, pid)) = name.split_once('-') else { continue };
        let (Ok(ns), Ok(pid)) = (ns.parse::<u128>(), pid.parse::<u32>()) else { continue };
        if !alive(pid) {
            let _ = std::fs::remove_file(e.path());
            continue;
        }
        out.push((ns, pid, name));
    }
    out.sort();
    out.into_iter().map(|(_, _, n)| n).collect()
}

/// Whether process `pid` still runs: in /proc and not a zombie (exited but not yet reaped by its parent).
fn alive(pid: u32) -> bool {
    match std::fs::read(format!("/proc/{pid}/stat")) {
        Ok(stat) => {
            let after = stat.iter().rposition(|&b| b == b')').map_or(&stat[..], |i| &stat[i + 1..]);
            after.split(|b| b.is_ascii_whitespace()).find(|f| !f.is_empty()) != Some(b"Z")
        }
        Err(_) => false,
    }
}

/// The lock, taken when free and nobody waits, else after the jobs that queued before this one.
pub fn acquire_in_turn(path: &PathBuf) -> io::Result<File> {
    let queue = queue_of(path);
    let file = open(path)?;
    std::fs::create_dir_all(&queue)?;
    if line(&queue).is_empty() {
        match file.try_lock() {
            Ok(()) => return Ok(file),
            Err(TryLockError::WouldBlock) => {}
            Err(TryLockError::Error(e)) => return Err(e),
        }
    }
    let ns = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let name = format!("{ns:020}-{}", std::process::id());
    let ticket = queue.join(&name);
    OpenOptions::new().write(true).create_new(true).open(&ticket)?;
    let waited = (|| -> io::Result<()> {
        let who = holder_text(path);
        let place = line(&queue).iter().position(|n| *n == name).map_or(1, |i| i + 1);
        eprintln!("mkr: waiting for the GPU, held by {} ({place} in line)", if who.is_empty() { "another job" } else { &who });
        let t0 = Instant::now();
        loop {
            if !ticket.exists() {
                // someone cleared the queue: back in at the same place
                OpenOptions::new().write(true).create(true).truncate(false).open(&ticket)?;
            }
            if line(&queue).first() == Some(&name) {
                file.lock()?; // first in line: the next turn is this job's
                break;
            }
            std::thread::sleep(POLL);
        }
        eprintln!("mkr: got the GPU after {:.0} s", t0.elapsed().as_secs_f64());
        Ok(())
    })();
    let _ = std::fs::remove_file(&ticket);
    waited.map(|()| file)
}

fn open(path: &PathBuf) -> io::Result<File> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    OpenOptions::new().read(true).write(true).create(true).truncate(false).open(path)
}

/// The lock if nobody holds it, else None.
pub fn try_acquire(path: &PathBuf) -> io::Result<Option<File>> {
    let file = open(path)?;
    match file.try_lock() {
        Ok(()) => Ok(Some(file)),
        Err(TryLockError::WouldBlock) => Ok(None),
        Err(TryLockError::Error(e)) => Err(e),
    }
}

/// What the holder wrote into the lock file ("<pid> <command>"), trimmed; empty when it said nothing.
fn holder_text(path: &PathBuf) -> String {
    let mut text = String::new();
    if let Ok(mut f) = File::open(path) {
        let _ = f.read_to_string(&mut text);
    }
    text.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_second_job_cannot_take_the_gpu_until_the_first_lets_go() {
        let dir = std::env::temp_dir().join(format!("mkr_gpulock_test_{}", std::process::id()));
        let path = dir.join("gpu.lock");
        let first = try_acquire(&path).unwrap().expect("a free lock is taken");
        assert!(try_acquire(&path).unwrap().is_none(), "a held lock is not taken again");
        drop(first);
        assert!(try_acquire(&path).unwrap().is_some(), "a released lock is free");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_job_that_queued_first_goes_first() {
        // a live job (a sleeping process) has a ticket older than ours: we wait for its turn even though nobody holds
        // the lock, and get the GPU once that job is gone (its ticket is dropped as stale)
        let dir = std::env::temp_dir().join(format!("mkr_gpulock_fifo_{}", std::process::id()));
        let path = dir.join("gpu.lock");
        let mut earlier = std::process::Command::new("sleep").arg("1.5").spawn().unwrap();
        std::fs::create_dir_all(queue_of(&path)).unwrap();
        File::create(queue_of(&path).join(format!("{:020}-{}", 1u128, earlier.id()))).unwrap();
        let reaper = std::thread::spawn(move || earlier.wait());
        let t0 = Instant::now();
        let lock = acquire_in_turn(&path).unwrap();
        assert!(t0.elapsed() >= Duration::from_millis(1200), "it waited for the job ahead of it");
        drop(lock);
        reaper.join().unwrap().unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_lock_file_is_where_mk_puts_it() {
        let home = Some(OsString::from("/home/u"));
        assert_eq!(resolve(None, None, home.clone()), PathBuf::from("/home/u/.cache/mk/gpu.lock"));
        assert_eq!(resolve(None, Some("~/c".into()), home.clone()), PathBuf::from("/home/u/c/gpu.lock"));
        assert_eq!(resolve(Some("/x/l".into()), Some("/c".into()), home), PathBuf::from("/x/l"));
    }
}
