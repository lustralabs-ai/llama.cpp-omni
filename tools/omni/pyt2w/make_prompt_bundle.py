#!/usr/bin/env python3
"""Build a token2wav prompt bundle (voice reference) from a wav file.

Writes the three files Token2Mel::load_prompt_bundle_dir() expects:
  spk_f32.bin             192 float32   (campplus speaker embedding)
  prompt_tokens_i32.bin   T+3 int32     (s3tokenizer tokens + first 3 repeated as lookahead)
  prompt_mel_btc_f32.bin  2T x 80 float32 (24 kHz log-mel, replicate-padded/cropped to 2T)

This mirrors Step-Audio2 Token2wav._prepare_prompt / setup_cache.

usage: make_prompt_bundle.py in.wav out_dir [token2wav_asset_dir]
The asset dir must contain speech_tokenizer_v2_25hz.onnx and campplus.onnx;
it defaults to $OMNI_T2W_ASSET_DIR.
"""
import os
import sys

import numpy as np
import onnxruntime
import s3tokenizer
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torchaudio.functional import melscale_fbanks

PRE_LOOKAHEAD = 3
UP_RATE = 2


def mel_24k(y: torch.Tensor) -> torch.Tensor:
    """CosyVoice mel_spectrogram(n_fft=1920, hop=480, 80 mels, fmax=8000, center=False) -> [T, 80]."""
    n_fft, hop = 1920, 480
    basis = melscale_fbanks(n_freqs=n_fft // 2 + 1, f_min=0.0, f_max=8000.0, n_mels=80,
                            sample_rate=24000, mel_scale="slaney", norm="slaney").T
    pad = (n_fft - hop) // 2
    y = torch.nn.functional.pad(y.view(1, 1, -1), (pad, pad), mode="reflect").view(1, -1)
    spec = torch.stft(y, n_fft, hop_length=hop, win_length=n_fft, window=torch.hann_window(n_fft),
                      center=False, normalized=False, onesided=True, return_complex=True)
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    spec = torch.log(torch.clamp(basis @ spec, min=1e-5))
    return spec[0].T.contiguous()


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    wav_path, out_dir = sys.argv[1], sys.argv[2]
    asset_dir = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("OMNI_T2W_ASSET_DIR", "")
    if not asset_dir:
        sys.exit("no token2wav asset dir (pass it or set OMNI_T2W_ASSET_DIR)")
    torch.set_num_threads(4)

    audio, sr = torchaudio.load(wav_path)
    audio = audio.mean(dim=0)
    audio_16k = torchaudio.functional.resample(audio, sr, 16000)
    audio_24k = torchaudio.functional.resample(audio, sr, 24000)

    tokenizer = s3tokenizer.load_model(os.path.join(asset_dir, "speech_tokenizer_v2_25hz.onnx")).eval()
    mels = s3tokenizer.log_mel_spectrogram(audio_16k)
    mels, mels_lens = s3tokenizer.padding([mels])
    with torch.no_grad():
        tokens, tokens_lens = tokenizer.quantize(mels, mels_lens)
    tokens = tokens[0, : int(tokens_lens[0])].to(torch.int32)

    feat = kaldi.fbank(audio_16k.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    opt = onnxruntime.SessionOptions()
    opt.intra_op_num_threads = 1
    spk_sess = onnxruntime.InferenceSession(os.path.join(asset_dir, "campplus.onnx"), sess_options=opt,
                                            providers=["CPUExecutionProvider"])
    spk = spk_sess.run(None, {spk_sess.get_inputs()[0].name: feat.unsqueeze(0).numpy()})[0]
    spk = np.asarray(spk, dtype=np.float32).reshape(-1)
    assert spk.size == 192, spk.shape

    mel = mel_24k(audio_24k)
    n_mel = tokens.numel() * UP_RATE
    if mel.shape[0] >= n_mel:
        mel = mel[:n_mel]
    else:
        mel = torch.cat([mel, mel[-1:].expand(n_mel - mel.shape[0], -1)], dim=0)

    tokens = torch.cat([tokens, tokens[:PRE_LOOKAHEAD]])

    os.makedirs(out_dir, exist_ok=True)
    tmp = out_dir.rstrip("/") + ".tmp"
    os.makedirs(tmp, exist_ok=True)
    spk.tofile(os.path.join(tmp, "spk_f32.bin"))
    tokens.numpy().astype(np.int32).tofile(os.path.join(tmp, "prompt_tokens_i32.bin"))
    mel.numpy().astype(np.float32).tofile(os.path.join(tmp, "prompt_mel_btc_f32.bin"))
    for f in os.listdir(tmp):
        os.replace(os.path.join(tmp, f), os.path.join(out_dir, f))
    os.rmdir(tmp)
    print(f"tokens={tokens.numel()} mel_frames={mel.shape[0]} spk_norm={np.linalg.norm(spk):.3f}")


if __name__ == "__main__":
    main()
