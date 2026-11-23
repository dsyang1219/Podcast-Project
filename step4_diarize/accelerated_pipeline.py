"""Accelerated pyannote speaker diarization.

Stock pyannote (3.1 / community-1) slides a 10 s segmentation window at a 1 s
step and then runs the speaker-embedding model once per (window, local speaker),
so each second of audio is embedded ~20-30 times. Two published changes remove
most of that cost while keeping the same models and clustering:

  * coarser segmentation stride (1 s -> 3 s)             Yamaguchi 2026, SDBench 2025
  * batched inference instead of batch size 1             Yamaguchi 2026 ("fp16 + bs64")
  * optional dominant-speaker-only embedding per window   SDBench "per-chunk" variant
  * relative minimum cluster size, mcs = round(f * n)      Yamaguchi 2026, f = 0.01

Yamaguchi (arXiv:2606.08505) reports stride 3 + per-chunk at 9.9x the speed of
the batched baseline with DER within 0.01 on AMI, a 0.075 -> 0.113 DER
regression on VoxConverse from speaker under-counting, and recovery to 0.079
with the relative minimum cluster size. SDBench (arXiv:2507.16136) reports
stride 4 at 15-38x with DER +0.01 to +0.06.

This module patches a pretrained pipeline *instance*; the models, the powerset
segmentation decoding, clustering and reconstruction are pyannote's own. The
pilot (pilot_eval.py) measures what each change costs on our audio before the
corpus run.

Requires a Hugging Face token with the pyannote model licences accepted
(pyannote/segmentation-3.0, pyannote/speaker-diarization-3.1 or
pyannote/speaker-diarization-community-1). Pass it as HF_TOKEN or --token.
"""
from __future__ import annotations

import math
import types
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio.pipelines.clustering import AgglomerativeClustering
from pyannote.core import SlidingWindowFeature

COMMUNITY_1 = "pyannote/speaker-diarization-community-1"
LEGACY_31 = "pyannote/speaker-diarization-3.1"


@dataclass
class AccelConfig:
    model: str = COMMUNITY_1
    stride: float = 0.3            # fraction of the segmentation window (10 s): 0.1 = stock, 0.3 = 3 s
    batch_size: int = 32           # segmentation and embedding batch size (stock: 1)
    embed_mode: str = "active"     # "full": pyannote's every (window, speaker) incl. silent ones;
                                   # "active": only speakers with speech in the window (same result, 2-3x fewer passes);
                                   # "dominant": one per window (SDBench per-chunk)
    mcs_fraction: float | None = 0.01   # relative minimum cluster size; None keeps pyannote's rule
    half: bool = True              # fp16 autocast for the embedding model (measured 1.9x on this GPU)
    device: str = "cuda"

    @classmethod
    def stock(cls, model: str = COMMUNITY_1, device: str = "cuda") -> "AccelConfig":
        """Pyannote as shipped, except for batching (which does not change results)."""
        return cls(model=model, stride=0.1, batch_size=32, embed_mode="full",
                   mcs_fraction=None, half=False, device=device)

    def tag(self) -> str:
        """Short label used in output file names and the pilot table."""
        s = f"s{self.stride:g}_b{self.batch_size}_{self.embed_mode}"
        s += f"_mcs{self.mcs_fraction:g}" if self.mcs_fraction else "_mcsStock"
        s += "_fp16" if self.half else "_fp32"
        return s


# --------------------------------------------------------------------------- patches

def _selective_get_embeddings(self: SpeakerDiarization, file, binary_segmentations: SlidingWindowFeature,
                              exclude_overlap: bool = False, hook=None):
    """Embed only the (window, speaker) pairs worth embedding.

    pyannote's stock loop runs the embedding model for every local speaker slot
    of every window, including slots with no speech at all (on a 2-speaker
    podcast at stride 3 s that is ~2,950 passes for ~1,130 with speech). Mode
    "active" skips the silent slots, which pyannote discards anyway
    (hard_clusters[inactive] = -2), so the result is unchanged. Mode
    "dominant" keeps one slot per window, the most active speaker (SDBench's
    per-chunk variant). Skipped slots get NaN embeddings; the clustering patch
    below sends them to the throw-away cluster.
    """
    mode = getattr(self, "_embed_select", "active")
    duration = binary_segmentations.sliding_window.duration
    num_chunks, num_frames, num_speakers = binary_segmentations.data.shape
    dim = self._embedding.dimension

    if exclude_overlap:
        min_num_samples = self._embedding.min_num_samples
        num_samples = duration * self._embedding.sample_rate
        min_num_frames = math.ceil(num_frames * min_num_samples / num_samples)
        clean = binary_segmentations.data * (np.sum(binary_segmentations.data, axis=2, keepdims=True) < 2)
    else:
        min_num_frames = -1
        clean = binary_segmentations.data

    masks = np.nan_to_num(binary_segmentations.data, nan=0.0).astype(np.float32)
    clean = np.nan_to_num(clean, nan=0.0).astype(np.float32)
    activity = masks.sum(axis=1)                       # (num_chunks, num_speakers)

    if mode == "dominant":
        items = [(c, int(activity[c].argmax())) for c in range(num_chunks) if activity[c].max() > 0]
    else:
        cs, ss = np.nonzero(activity > 0)
        items = list(zip(cs.tolist(), ss.tolist()))

    embeddings = np.full((num_chunks, num_speakers, dim), np.nan, dtype=np.float32)
    bs = self.embedding_batch_size
    total = math.ceil(len(items) / bs)
    if hook is not None:
        hook("embeddings", None, total=total, completed=0)

    use_half = getattr(self, "_embed_half", False)
    for b in range(total):
        batch = items[b * bs:(b + 1) * bs]
        waveforms, used_masks = [], []
        for c, s_ in batch:
            chunk = binary_segmentations.sliding_window[c]
            waveform, _ = self._audio.crop(file, chunk, mode="pad")
            m = clean[c, :, s_] if clean[c, :, s_].sum() > min_num_frames else masks[c, :, s_]
            waveforms.append(waveform[None])
            used_masks.append(torch.from_numpy(m)[None])
        emb = _embed_batch(self._embedding, torch.vstack(waveforms), torch.vstack(used_masks), use_half)
        for (c, s_), e in zip(batch, emb):
            embeddings[c, s_] = e
        if hook is not None:
            hook("embeddings", emb, total=total, completed=b + 1)

    bad = int(np.any(np.isnan(embeddings[tuple(np.array(items).T)]), axis=1).sum()) if items else 0
    if bad:
        warnings.warn(f"{bad} of {len(items)} computed embeddings are NaN (half={use_half}); they go to the discard cluster")
    if items and bad == len(items):
        raise RuntimeError("every embedding is NaN: the embedding model produced no usable output")
    return embeddings


def _embed_batch(embedding, waveforms: torch.Tensor, masks: torch.Tensor, half: bool) -> np.ndarray:
    """One embedding forward pass. With half=True the filterbank stays in fp32
    (its log-mel on 16-bit-scaled audio overflows fp16 and yields NaN) and only
    the ResNet runs under fp16 autocast, which is where the time is."""
    model = getattr(embedding, "model_", None)
    dev = embedding.device
    if model is None or not hasattr(model, "compute_fbank") or not hasattr(model, "resnet"):
        return np.asarray(embedding(waveforms, masks=masks), dtype=np.float32)
    with torch.inference_mode():
        fbank = model.compute_fbank(waveforms.to(dev))
        if half:
            with torch.autocast(dev.type, dtype=torch.float16):
                out = model.resnet(fbank, weights=masks.to(dev))[1]
        else:
            out = model.resnet(fbank, weights=masks.to(dev))[1]
    return out.float().cpu().numpy()


def _patch_nan_to_discard(clustering) -> None:
    """Send (window, speaker) pairs that have no embedding (NaN) to the
    throw-away cluster (-2) after clustering, for any clustering class.

    Works at the class level because `obj(...)` resolves `__call__` on the
    type, not the instance; the instance gets a one-off subclass.
    """
    base = type(clustering)
    if getattr(base, "_nan_patched", False):
        return

    def __call__(self, embeddings, segmentations=None, num_clusters=None, min_clusters=None,
                 max_clusters=None, **kwargs):
        hard, soft, centroids = base.__call__(self, embeddings, segmentations=segmentations,
                                              num_clusters=num_clusters, min_clusters=min_clusters,
                                              max_clusters=max_clusters, **kwargs)
        missing = np.any(np.isnan(embeddings), axis=2)
        hard = np.array(hard, copy=True)
        hard[missing] = -2
        return hard, soft, centroids

    patched = type(base.__name__ + "NaNSafe", (base,), {"__call__": __call__, "_nan_patched": True})
    clustering.__class__ = patched


def _patch_relative_mcs(clustering: AgglomerativeClustering, fraction: float) -> None:
    """mcs = round(f * n) (Yamaguchi 2026). pyannote's own rule is
    min(min_cluster_size, round(0.1 n)) capped at 12-20, which saturates at the
    cap for any recording longer than a few minutes; with f = 0.01 our value is
    always the smaller one, so it becomes the binding minimum."""
    original = clustering.cluster

    def cluster(self, embeddings, min_clusters=None, max_clusters=None, num_clusters=None):
        n = embeddings.shape[0]
        self.min_cluster_size = max(1, round(fraction * n))
        return original(embeddings, min_clusters=min_clusters, max_clusters=max_clusters,
                        num_clusters=num_clusters)

    clustering.cluster = types.MethodType(cluster, clustering)


# --------------------------------------------------------------------------- build

def build_pipeline(cfg: AccelConfig, token: str | None = None) -> SpeakerDiarization:
    pipeline = Pipeline.from_pretrained(cfg.model, token=token)
    if pipeline is None:
        raise RuntimeError(f"could not load {cfg.model}: accept its licence on Hugging Face and pass a token")
    if not isinstance(pipeline, SpeakerDiarization):
        raise TypeError(f"{cfg.model} is a {type(pipeline).__name__}, not a SpeakerDiarization pipeline")

    pipeline.to(torch.device(cfg.device))

    # stride: the Inference object was built at construction with step = 0.1 * duration
    duration = pipeline._segmentation.duration
    pipeline.segmentation_step = cfg.stride
    pipeline._segmentation.step = cfg.stride * duration

    pipeline.segmentation_batch_size = cfg.batch_size
    pipeline.embedding_batch_size = cfg.batch_size

    if cfg.embed_mode in ("active", "dominant"):
        pipeline._embed_select = cfg.embed_mode
        pipeline._embed_half = cfg.half
        pipeline.get_embeddings = types.MethodType(_selective_get_embeddings, pipeline)
        _patch_nan_to_discard(pipeline.clustering)
    elif cfg.embed_mode == "full":
        if cfg.half:
            warnings.warn("half precision is applied only in 'active'/'dominant' embed modes; 'full' runs fp32")
    else:
        raise ValueError(f"embed_mode must be 'full', 'active' or 'dominant', got {cfg.embed_mode!r}")

    if cfg.mcs_fraction:
        if isinstance(pipeline.clustering, AgglomerativeClustering):
            _patch_relative_mcs(pipeline.clustering, cfg.mcs_fraction)
        else:
            warnings.warn(f"{cfg.model} uses {type(pipeline.clustering).__name__}; "
                          "relative minimum cluster size applies to agglomerative clustering only, skipped")
    return pipeline


def describe(pipeline: SpeakerDiarization, cfg: AccelConfig) -> dict:
    """What actually got configured, for the per-episode JSON provenance block."""
    return {
        "model": cfg.model,
        "segmentation_window_s": float(pipeline._segmentation.duration),
        "segmentation_step_s": float(pipeline._segmentation.step),
        "batch_size": cfg.batch_size,
        "embed_mode": cfg.embed_mode,
        "half": cfg.half and cfg.embed_mode != "full",
        "clustering": type(pipeline.clustering).__name__.replace("NaNSafe", ""),
        "mcs_fraction": cfg.mcs_fraction,
        "device": cfg.device,
    }
