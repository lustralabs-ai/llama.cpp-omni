# Building llama.cpp-omni for ROCm on Strix Halo (gfx1151)

This is the build used by `MiniCPM-o-Demo/omni.sh` (`start cpp`). The machine is a
GMKtec EVO-X2 with a Ryzen AI Max+ / Radeon 8060S iGPU (`gfx1151`).

## Toolchain

| Piece | Where it comes from |
|---|---|
| HIP compiler | `hipcc` / AMD clang 23 from the pip ROCm SDK (TheRock) in the `~/py` venv: `~/py/lib/python3.12/site-packages/_rocm_sdk_devel` |
| HIP CMake packages (`hip`, `hipblas`, `rocblas`) | `/opt/rocm/lib/cmake/...` |
| Runtime libs (`libamdhip64`, `hipblas`, `hipblaslt`, `rocblas`) | `/opt/rocm/core-10.1/lib` |
| Host C/C++ compiler | system `/usr/bin/cc`, `/usr/bin/c++` |
| Generator | Ninja, with ccache |

CMake picks the HIP compiler from `PATH`, so put the venv first when configuring.

## Configure

Run this from the repo root:

```sh
PATH=$HOME/py/bin:$PATH cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS=gfx1151 \
    -DGGML_HIP_ROCWMMA_FATTN=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF
```

Everything else is left at upstream defaults:
- shared libs;
- `GGML_NATIVE=ON`;
- `GGML_HIP_GRAPHS=ON`;
- `GGML_HIP_NO_VMM=ON`;
- `GGML_CUDA_FA=ON`;
- OpenMP on.

### Keep `GGML_HIP_ROCWMMA_FATTN=OFF`

On gfx1151 the rocWMMA flash-attention prefill gets slower as context depth grows. With `llama-bench` at pp80 and depth 7000, it runs at 137 t/s with rocWMMA on versus 629 t/s with it off.

Duplex mode prefills about 80 tokens every second at a growing `n_past`. With rocWMMA on, each step costs more until the GPU is over 100% busy, and audio drops out. With it off, a listen step stays flat at about 150 ms up to an `n_past` of 6k.

## Build

```sh
cmake --build build --target llama-omni-server -j 16
```

The result is `build/bin/llama-omni-server` plus `build/bin/libomni.so` and the ggml/llama shared libs. Most omni changes land in `libomni.so` (`tools/omni/`). Restart the server to pick them up:

```sh
./omni.sh stop cpp && ./omni.sh start cpp     # from MiniCPM-o-Demo
```

To check that the GPU is visible:

```sh
build/bin/llama-omni-server --list-devices    # should list ROCm0
```

## Runtime settings (set by omni.sh, not CMake)

- `GGML_CUDA_DISABLE_GRAPHS=1` turns off HIP graphs at runtime. The build keeps graphs compiled in, but the duplex graph shapes change every turn. That forces a recapture each turn, which stalls token2wav for 2–3 s.
- `ROCBLAS_USE_HIPBLASLT=1` sends rocBLAS GEMMs through hipBLASLt. Token2wav goes from 168 to 140 ms per 1 s of audio, and the vision encoder gains about 5 ms per frame. The Q8_0 LLM is unaffected because it uses ggml's own kernels. In the 10-turn video e2e at 1 frame/s (120 W mode), underruns of 0.3 s or more went from 4–12 per run to 0–1.
- `MTMD_BACKEND_DEVICE=ROCm0` runs the vision and audio encoders on the GPU.
- `OMNI_T2W_ASSET_DIR` and `OMNI_T2W_PYTHON` are needed for ref-audio voice cloning. The voice prompt bundles are cached in `~/.cache/llama-omni/voices`.

These two are optional, and omni.sh passes them through from your shell:
- `OMNI_LOG_VERBOSE=1` turns on the per-step trace: `[prof]` lines, encoder and TTS chunk lines, and per-window T2W lines. Without it, the log keeps only session lifecycle, init, KV slides, first-audio latency, warnings and errors.
- `OMNI_DEBUG_DUMP=1` keeps the debug dumps under `/tmp/omni_ws/<session_id>/`: `llm_debug/`, `tts_txt/` and `tts_wav/`. Without it, nothing is dumped, step inputs are deleted once encoded, and the session directory is removed when the session ends.

## Notes

- Q8_0 is the LLM quant in use. Keep the vision encoder at F16, because Q8_0 vision is slower on this GPU.
- How much memory the GPU can use is set by the amdttm kernel module's limit, not by this build. The DKMS amdgpu driver uses amdttm.
