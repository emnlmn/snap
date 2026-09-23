use std::path::PathBuf;

/// Locate the llama.cpp sources vendored inside llama-cpp-sys-2 in the cargo
/// registry, so the shim can compile against common/chat.h.
fn llama_cpp_dir() -> PathBuf {
    let home = std::env::var("CARGO_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap()).join(".cargo"));
    let registry_src = home.join("registry").join("src");
    let mut found = vec![];
    if let Ok(reg) = std::fs::read_dir(&registry_src) {
        for index in reg.flatten() {
            if let Ok(crates) = std::fs::read_dir(index.path()) {
                for c in crates.flatten() {
                    let name = c.file_name().to_string_lossy().to_string();
                    if name.starts_with("llama-cpp-sys-2-") {
                        let d = c.path().join("llama.cpp");
                        if d.join("common/chat.h").exists() {
                            found.push(d);
                        }
                    }
                }
            }
        }
    }
    found.sort();
    found.pop().unwrap_or_else(|| {
        panic!(
            "llama.cpp sources not found under {}",
            registry_src.display()
        )
    })
}

fn main() {
    let lc = llama_cpp_dir();
    println!("cargo:rerun-if-changed=csrc/chat_shim.cpp");
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("csrc/chat_shim.cpp")
        .include(lc.join("common"))
        .include(lc.join("include"))
        .include(lc.join("ggml/include"))
        .include(lc.join("vendor"))
        .warnings(false)
        .compile("snap_chat_shim");
    stage_shared_libs();
}

/// The sys crate builds ggml/llama as shared libraries under several triggers
/// (dynamic-backends, dynamic-link, LLAMA_BUILD_SHARED_LIBS=1), stages the
/// dylibs next to the executable only when its build script actually runs —
/// and never sets an rpath, so the binary only starts under
/// DYLD_/LD_LIBRARY_PATH. Point the loader at the executable's own directory
/// unconditionally (inert on static builds, rescues every shared-build path)
/// and stage the libs ourselves so a cached dep build can't strand the binary.
/// Backend modules for GGML_BACKEND_DL are copied likewise — upstream leaves
/// them in OUT_DIR where dlopen can't find them. The exe dir is what ships in
/// release tarballs.
///
/// On linux we force old-style DT_RPATH: the default DT_RUNPATH only applies
/// to the executable's own NEEDED entries, while the shipped libllama.so has
/// its own deps on libggml-*.so — only RPATH is inherited down the chain.
fn stage_shared_libs() {
    println!("cargo:rerun-if-env-changed=LLAMA_BUILD_SHARED_LIBS");
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "macos" => println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path"),
        "linux" => {
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
            println!("cargo:rustc-link-arg=-Wl,--disable-new-dtags");
        }
        _ => {} // windows: DLLs resolve from the exe dir natively
    }
    // OUT_DIR = target/[triple/]<profile>/build/snap-<hash>/out -> profile dir
    let Some(exe_dir) = std::env::var("OUT_DIR")
        .ok()
        .map(PathBuf::from)
        .and_then(|p| p.ancestors().nth(3).map(|a| a.to_path_buf()))
    else {
        return;
    };
    for out_dir in sys_out_dirs(&exe_dir) {
        // shared libs land in lib/ (bin/ on windows); backend modules in
        // backends/. Static builds produce only .a — filtered out.
        for sub in ["lib", "bin", "backends"] {
            if let Ok(entries) = std::fs::read_dir(out_dir.join(sub)) {
                for e in entries.flatten() {
                    let name = e.file_name().to_string_lossy().into_owned();
                    if is_shared_lib(&name) {
                        stage_file(&e.path(), &exe_dir.join(&name));
                    }
                }
            }
        }
    }
}

fn is_shared_lib(name: &str) -> bool {
    name.ends_with(".dylib") || name.ends_with(".dll") || name.contains(".so")
}

/// llama-cpp-sys-2's OUT_DIR: DEP_LLAMA_ROOT (links = "llama") when cargo
/// exposes it — the exact variant being linked. Else the sibling build dirs
/// next to our own (stale variants may leak old libs, harmless beside a
/// static binary that never loads them).
fn sys_out_dirs(exe_dir: &std::path::Path) -> Vec<PathBuf> {
    if let Ok(root) = std::env::var("DEP_LLAMA_ROOT") {
        return vec![PathBuf::from(root)];
    }
    let mut dirs = vec![];
    if let Ok(entries) = std::fs::read_dir(exe_dir.join("build")) {
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with("llama-cpp-sys-2-") {
                dirs.push(e.path().join("out"));
            }
        }
    }
    dirs
}

/// Copy a staged lib, preserving the versioned symlink chain
/// (libggml.dylib -> libggml.0.dylib -> libggml.0.24.0.dylib) on unix.
fn stage_file(src: &std::path::Path, dst: &std::path::Path) {
    #[cfg(unix)]
    if src.is_symlink() {
        if let Ok(target) = std::fs::read_link(src) {
            let _ = std::fs::remove_file(dst);
            if std::os::unix::fs::symlink(&target, dst).is_ok() {
                return;
            }
        }
    }
    if dst.symlink_metadata().is_ok() {
        let _ = std::fs::remove_file(dst);
    }
    if let Err(err) = std::fs::copy(src, dst) {
        println!(
            "cargo:warning=lib staging failed for {}: {err}",
            dst.display()
        );
    }
}
