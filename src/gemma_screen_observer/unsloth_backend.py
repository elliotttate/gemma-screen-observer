"""Unsloth/transformers backend for Gemma 4 vision inference.

Uses HuggingFace transformers with Unsloth optimizations and direct CUDA
for maximum GPU utilization on NVIDIA GPUs. Supports 4-bit quantization
via bitsandbytes for reduced VRAM usage and faster inference.
"""

from __future__ import annotations

import base64
import io
import logging
import time

import torch
from PIL import Image

from .capture import Frame

logger = logging.getLogger(__name__)

# Global model/processor singletons (loaded once, reused across calls)
_model = None
_processor = None
_device = None


def _load_model(model_name: str, load_in_4bit: bool = True):
    """Load Gemma 4 model with Unsloth/transformers optimizations."""
    global _model, _processor, _device

    if _model is not None:
        return

    from transformers import AutoModelForMultimodalLM, AutoProcessor

    logger.info("Loading model %s (4bit=%s)...", model_name, load_in_4bit)
    t0 = time.perf_counter()

    _processor = AutoProcessor.from_pretrained(model_name)

    load_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    _model = AutoModelForMultimodalLM.from_pretrained(model_name, **load_kwargs)
    _device = next(_model.parameters()).device

    elapsed = time.perf_counter() - t0
    logger.info("Model loaded in %.1fs on %s", elapsed, _device)


def _frame_to_pil(frame: Frame) -> Image.Image:
    """Convert a Frame to a PIL Image suitable for the model."""
    img = frame.image
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Resize to reduce processing time — 640x360 is enough for scene understanding
    if img.width > 640:
        ratio = 640 / img.width
        img = img.resize((640, int(img.height * ratio)), Image.LANCZOS)
    return img


async def analyze_frame(
    frame: Frame,
    prompt: str,
    model_name: str = "google/gemma-4-e2b-it",
    max_tokens: int = 512,
    load_in_4bit: bool = True,
) -> str:
    """Analyze a single frame using the transformers backend.

    Runs synchronously on the GPU but is called from async context.
    """
    import asyncio
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _analyze_sync, frame, prompt, model_name, max_tokens, load_in_4bit
    )


def _analyze_sync(
    frame: Frame,
    prompt: str,
    model_name: str,
    max_tokens: int,
    load_in_4bit: bool,
) -> str:
    """Synchronous frame analysis."""
    _load_model(model_name, load_in_4bit)

    img = _frame_to_pil(frame)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = _processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(_model.device)

    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    elapsed = time.perf_counter() - t0

    # Decode only the new tokens
    input_len = inputs["input_ids"].shape[1]
    response = _processor.decode(outputs[0][input_len:], skip_special_tokens=True)

    output_tokens = outputs.shape[1] - input_len
    tps = output_tokens / elapsed if elapsed > 0 else 0
    logger.debug("Generated %d tokens in %.2fs (%.1f tok/s)", output_tokens, elapsed, tps)

    return response


async def analyze_two_frames(
    prev: Frame,
    current: Frame,
    prompt: str,
    model_name: str = "google/gemma-4-e2b-it",
    max_tokens: int = 512,
    load_in_4bit: bool = True,
) -> str:
    """Compare two frames using the transformers backend."""
    import asyncio
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _compare_sync, prev, current, prompt, model_name, max_tokens, load_in_4bit
    )


def _compare_sync(
    prev: Frame,
    current: Frame,
    prompt: str,
    model_name: str,
    max_tokens: int,
    load_in_4bit: bool,
) -> str:
    """Synchronous two-frame comparison."""
    _load_model(model_name, load_in_4bit)

    img_prev = _frame_to_pil(prev)
    img_curr = _frame_to_pil(current)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_prev},
                {"type": "image", "image": img_curr},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = _processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(_model.device)

    with torch.inference_mode():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    input_len = inputs["input_ids"].shape[1]
    return _processor.decode(outputs[0][input_len:], skip_special_tokens=True)


def unload_model():
    """Free GPU memory by unloading the model."""
    global _model, _processor, _device
    if _model is not None:
        del _model
        _model = None
        _processor = None
        _device = None
        torch.cuda.empty_cache()
        logger.info("Model unloaded, GPU memory freed")
