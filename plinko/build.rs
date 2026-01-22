fn main() {
    println!("cargo:rerun-if-changed=cuda/hint_kernel.cu");

    // Only compile CUDA if nvcc is available AND cuda feature is enabled
    #[cfg(feature = "cuda")]
    if std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .is_ok()
    {
        let out_dir = std::env::var("OUT_DIR").unwrap();
        // Default to sm_80 (A100/H100 compatible), can override with CUDA_ARCH env var
        let arch = std::env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_80".to_string());

        println!(
            "cargo:warning=Compiling hint kernel for CUDA architecture: {}",
            arch
        );

        let status = std::process::Command::new("nvcc")
            .arg("-ptx")
            .arg("-O3")
            .arg(format!("-arch={}", arch))
            .arg("cuda/hint_kernel.cu")
            .arg("-o")
            .arg(format!("{}/hint_kernel.ptx", out_dir))
            .status()
            .unwrap();

        if !status.success() {
            panic!("Failed to compile CUDA kernel to PTX");
        }

        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-lib=dylib=cudart");
    } else {
        #[cfg(feature = "cuda")]
        println!("cargo:warning=nvcc not found, skipping CUDA kernel compilation");
    }
}
