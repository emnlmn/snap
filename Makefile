.PHONY: setup build test test-unit smoke serve lint check clean

# cargo lives in ~/.cargo/bin (rustup shim): resolve to an absolute path,
# the user shell PATH is unreliable inside make.
CARGO ?= $(shell command -v cargo 2>/dev/null || echo $(HOME)/.cargo/bin/cargo)
MODEL ?= minicpm5-2b
# backend features, e.g. `make build FEATURES="vulkan"` — see Cargo.toml [features]
FEATURES ?=

# Toolchain + build deps (llama.cpp is vendored; needs a C/C++ compiler
# and cmake — Xcode CLT on macOS, build-essential on Debian-ish).
setup:
	@command -v $(CARGO) >/dev/null || { \
		echo "rust not found: installing via rustup"; \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
	}
	@command -v cmake >/dev/null || { echo "cmake missing: brew install cmake | apt install cmake"; exit 1; }
	@command -v c++ >/dev/null || command -v clang++ >/dev/null || command -v g++ >/dev/null || { \
		echo "no C++ compiler: xcode-select --install | apt install build-essential"; exit 1; }
	@$(CARGO) --version

build:
	$(CARGO) build --release $(if $(FEATURES),--features $(FEATURES),)
	@echo "binary: ./target/release/snap"

# unit tests (fast, no model) + functional smoke (loads the GGUF)
test: test-unit smoke

test-unit:
	$(CARGO) test --release

# functional: real model, real request, checks output shape
smoke: build
	./target/release/snap -p eval/smoke.json --model $(MODEL)

serve: build
	./target/release/snap serve --model $(MODEL)

lint:
	@$(CARGO) fmt --version >/dev/null 2>&1 || { \
		echo "rustfmt/clippy not installed: rustup component add rustfmt clippy"; exit 1; }
	$(CARGO) fmt --check
	$(CARGO) clippy --release -- -D warnings

check: test-unit
	$(CARGO) check --release

clean:
	$(CARGO) clean
