# Build stage
FROM rust:1.82-slim-bookworm as builder
RUN apt-get update && apt-get install -y libssl-dev pkg-config
RUN rustup toolchain install nightly
RUN rustup default nightly

WORKDIR /app

# Copy workspace definition
COPY Cargo.toml Cargo.lock ./

# Create dummy crates to build dependencies first (caching layer)
RUN mkdir -p src && \
    echo "fn main() {}" > src/main.rs && \
    mkdir -p plinko/src && \
    echo "pub fn placeholder() {}" > plinko/src/lib.rs && \
    touch plinko/Cargo.toml

# Copy the actual Cargo.toml files
COPY plinko/Cargo.toml plinko/
# If additional workspace crates are added, copy their Cargo.toml files here.

# Build dependencies
RUN cargo build --release -p plinko --bins --jobs 2 || true

# Now copy the real source code
COPY plinko/src plinko/src
# Build the actual binaries
RUN cargo build --release -p plinko --bins --jobs 2

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binaries
COPY --from=builder /app/target/release/plinko_hints /usr/local/bin/
COPY --from=builder /app/target/release/bench_hints /usr/local/bin/

# Create data directories
RUN mkdir -p /data /public/deltas /public/snapshots

# Environment defaults
ENV PLINKO_STATE_DB_PATH=/data/database.bin
ENV PLINKO_STATE_ADDRESS_MAPPING_PATH=/data/account-mapping.bin
ENV PLINKO_STATE_PUBLIC_ROOT=/public
ENV PLINKO_STATE_HTTP_PORT=3002
ENV RUST_LOG=info

EXPOSE 3002

CMD ["plinko_hints"]
