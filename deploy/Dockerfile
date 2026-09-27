# syntax=docker/dockerfile:1
# snap demo image: build the release binary, ship a slim runtime.
# Built by Modal via deploy/modal_app.py — plain `docker build -f deploy/Dockerfile .` works too.
FROM rust:1-bookworm AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends cmake g++ \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /snap
COPY Cargo.toml Cargo.lock build.rs ./
COPY csrc ./csrc
COPY src ./src
RUN cargo build --release --locked

FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /snap/target/release/snap /usr/local/bin/snap
ENTRYPOINT ["snap"]
