# SPDX-License-Identifier: Apache-2.0
"""
JANG model engine for loading quantized MoE+SSM hybrid models.

This engine wraps the jang-tools package to load JANG quantized models
with mixed-precision quantization (attention 6-8-bit, experts 2-4-bit).
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..api.tool_calling import convert_tools_for_template
from ..api.utils import clean_special_tokens, detect_and_strip_partial
from ..models.vlm import VLMModelAdapter
from .base import BaseEngine, GenerationOutput

logger = logging.getLogger(__name__)


class _TokenizerWrapper:
    """
    Wrapper for tokenizers that don't have an encode() method.

    Some VLM processors (like Qwen3VLProcessor) return a tokenizer that
    doesn't have encode() directly. This wrapper delegates to the underlying
    HF tokenizer's encode method.
    """

    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer

    def __getattr__(self, name: str) -> Any:
        # Delegate attribute access to the underlying tokenizer
        return getattr(self._tokenizer, name)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        # Try different ways to get encode
        if hasattr(self._tokenizer, "encode"):
            return self._tokenizer.encode(text)
        if hasattr(self._tokenizer, "tokenize"):
            # Fallback: use tokenize and map to ids
            tokens = self._tokenizer.tokenize(text)
            if hasattr(self._tokenizer, "convert_tokens_to_ids"):
                return self._tokenizer.convert_tokens_to_ids(tokens)
        # Try calling the tokenizer directly (e.g. HF processors)
        if callable(self._tokenizer):
            result = self._tokenizer(text)
            if isinstance(result, dict) and "input_ids" in result:
                return list(result["input_ids"])
        raise TypeError(
            f"Cannot encode text: tokenizer {type(self._tokenizer).__name__} "
            f"has no encode(), tokenize(), or __call__ returning input_ids"
        )


class JANGLoader(BaseEngine):
    """
    Engine for loading JANG quantized models via jang-tools.

    JANG models use mixed-precision quantization and require special handling:
    - Mixed bit widths per tensor (read from jang_config.json)
    - Nemotron-H weight renaming (fc1/fc2)
    - Auto bfloat16 for large expert models (512+ experts)
    - VLM support with vision encoder

    Args:
        model_name: HuggingFace model name or local path
        scheduler_config: Optional scheduler configuration
        stream_interval: Tokens to batch before streaming (1=every token)
        enable_thinking: Enable thinking mode for reasoning models
        model_settings: Optional per-model settings for post-load transforms
    """

    def __init__(
        self,
        model_name: str,
        scheduler_config: Any | None = None,
        stream_interval: int = 1,
        enable_thinking: bool | None = None,
        model_settings: Any | None = None,
    ):
        self._model_name = model_name
        self._scheduler_config = scheduler_config
        self._stream_interval = stream_interval
        self._enable_thinking = enable_thinking
        self._model_settings = model_settings

        self._model = None
        self._tokenizer = None
        self._engine = None
        self._loaded = False

        self._processor = None

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer."""
        return self._tokenizer

    @property
    def model_type(self) -> str | None:
        """Get the model type from config (e.g., 'gpt_oss', 'llama', 'qwen2')."""
        if self._model is None:
            return None
        try:
            if hasattr(self._model, 'config'):
                config = self._model.config
                if hasattr(config, 'model_type'):
                    model_type = config.model_type
                    return model_type if isinstance(model_type, str) else None
                elif isinstance(config, dict):
                    model_type = config.get('model_type')
                    return model_type if isinstance(model_type, str) else None
            if hasattr(self._model, 'args'):
                args = self._model.args
                if hasattr(args, 'model_type'):
                    model_type = args.model_type
                    return model_type if isinstance(model_type, str) else None
        except Exception as e:
            logger.debug(f"Error getting model_type: {e}")
        return None

    def _check_jang_tools_available(self) -> None:
        """Check if jang-tools is installed."""
        try:
            import jang_tools  # noqa: F401
        except ImportError:
            from ..exceptions import JANGDependencyError
            raise JANGDependencyError(
                "jang-tools package not found. Install with: pip install jang[mlx]",
                model_name=self._model_name,
            )

    def _get_config_dict(self) -> dict:
        """Get model config as a dict, handling both dict and object configs."""
        if self._model is None:
            return {}
        config = getattr(self._model, 'config', None)
        if config is None:
            return {}
        if isinstance(config, dict):
            return config
        # Config is an object — try to extract relevant fields
        result = {}
        for attr in ('architectures', 'num_local_experts', 'num_experts',
                      'n_routed_experts', 'hidden_size', 'model_type',
                      'text_config'):
            if hasattr(config, attr):
                result[attr] = getattr(config, attr)
        return result

    def _detect_nemotron_h(self) -> bool:
        """Check if model is Nemotron-H architecture."""
        if self._model is None:
            return False
        try:
            config = self._get_config_dict()
            for a in config.get('architectures', []):
                if 'Nemotron' in a:
                    return True
            return False
        except Exception:
            return False

    def _needs_bfloat16(self) -> bool:
        """Check if model needs bfloat16 (512+ experts, hidden>=4096)."""
        if self._model is None:
            return False
        try:
            config = self._get_config_dict()
            text_cfg = config.get('text_config', config)
            if isinstance(text_cfg, dict):
                cfg = text_cfg
            else:
                cfg = config
            n_experts = cfg.get('num_local_experts',
                        cfg.get('num_experts',
                        cfg.get('n_routed_experts', 0)))
            hidden_size = cfg.get('hidden_size', 0)
            if n_experts >= 512 and hidden_size >= 4096:
                return True
        except Exception as e:
            logger.debug(f"Error checking bfloat16 requirements: {e}")
        return False

    def _fix_nemotron_h_weights(self) -> None:
        """Fix Nemotron-H weights after JANG loading.

        JANG v2 stores Nemotron-H weights with different naming and quantized
        gate weights that mlx-lm's nemotron_h.py cannot handle directly.
        This applies three fixes:

        1. Rename switch_mlp.up_proj/down_proj -> switch_mlp.fc1/fc2
        2. Dequantize MoE gate weights (stored as quantized uint32, but
           mlx-lm expects nn.Linear with float weights)
        3. Drop mtp.* keys (multi-token prediction, unused at inference)
        """
        import mlx.nn as nn
        from mlx.utils import tree_flatten

        use_bfloat16 = self._needs_bfloat16()
        target_dtype = mx.bfloat16 if use_bfloat16 else mx.float16

        # Flatten current weights
        weights = dict(tree_flatten(self._model.parameters()))

        # --- 1. Rename up_proj/down_proj -> fc1/fc2 ---
        renames = {
            "switch_mlp.up_proj": "switch_mlp.fc1",
            "switch_mlp.down_proj": "switch_mlp.fc2",
        }
        renamed = {}
        rename_count = 0
        for k, v in weights.items():
            new_k = k
            for old, new in renames.items():
                if old in k:
                    new_k = k.replace(old, new)
                    rename_count += 1
                    break
            renamed[new_k] = v
        weights = renamed
        if rename_count > 0:
            logger.info(f"Nemotron-H: renamed {rename_count} switch_mlp weight keys (up_proj->fc1, down_proj->fc2)")

        # --- 2. Dequantize gate weights ---
        # Collect gate quantization parts: {prefix: {weight, scales, biases}}
        gate_parts: dict[str, dict[str, mx.array]] = {}
        non_gate_weights = {}
        for k, v in weights.items():
            if ".gate." in k:
                # Extract prefix (everything before .gate.)
                prefix = k[:k.index(".gate.") + len(".gate")]
                suffix = k[k.index(".gate.") + len(".gate."):]
                if prefix not in gate_parts:
                    gate_parts[prefix] = {}
                if suffix == "weight":
                    gate_parts[prefix]["weight"] = v
                elif suffix == "scales":
                    gate_parts[prefix]["scales"] = v
                elif suffix == "biases":
                    gate_parts[prefix]["biases"] = v
                else:
                    # Other gate sub-keys, keep as-is
                    non_gate_weights[k] = v
            else:
                non_gate_weights[k] = v

        weights = non_gate_weights
        dequant_count = 0
        for prefix, parts in gate_parts.items():
            gate_weight = parts.get("weight")
            scales = parts.get("scales")
            biases = parts.get("biases")

            if gate_weight is None:
                continue

            if scales is not None and biases is not None:
                # Gate is quantized — dequantize by trying bits in order
                # Gate is typically 8-bit (CRITICAL tier)
                dequantized = None
                for bits in [8, 6, 4, 3, 2]:
                    elem_per_u32 = 32 // bits
                    real_cols = gate_weight.shape[-1] * elem_per_u32
                    gs = real_cols // scales.shape[-1]
                    if gs > 0 and gs * scales.shape[-1] == real_cols:
                        dequantized = mx.dequantize(
                            gate_weight, scales, biases, gs, bits
                        )
                        dequantized = dequantized.astype(target_dtype)
                        logger.debug(
                            f"Nemotron-H: dequantized {prefix}.weight "
                            f"({bits}-bit, group_size={gs}) -> {dequantized.shape}"
                        )
                        break
                if dequantized is not None:
                    weights[f"{prefix}.weight"] = dequantized
                    dequant_count += 1
                else:
                    # Could not dequantize — keep original parts
                    logger.warning(
                        f"Nemotron-H: could not dequantize {prefix}, "
                        f"keeping original quantized weights"
                    )
                    weights[f"{prefix}.weight"] = gate_weight
                    weights[f"{prefix}.scales"] = scales
                    weights[f"{prefix}.biases"] = biases
            else:
                # Gate is not quantized, keep weight as-is
                weights[f"{prefix}.weight"] = gate_weight

        if dequant_count > 0:
            logger.info(
                f"Nemotron-H: dequantized {dequant_count} gate weights to {target_dtype}"
            )

        # --- 3. Drop mtp.* keys ---
        mtp_count = sum(1 for k in weights if k.startswith("mtp."))
        if mtp_count > 0:
            weights = {k: v for k, v in weights.items() if not k.startswith("mtp.")}
            logger.info(f"Nemotron-H: dropped {mtp_count} mtp.* keys")

        # Reload fixed weights into model (strict=False required per JANG guide)
        weight_list = list(weights.items())
        self._model.load_weights(weight_list, strict=False)

        # Clean up stale quantization attributes on gate modules (nn.Linear)
        # After dequantization, gate modules may still have scales/biases attrs
        # from the original quantized load — remove them so they don't waste memory
        for path, module in tree_flatten(
            self._model.leaf_modules(), is_leaf=nn.Module.is_module
        ):
            if ".gate" in path and isinstance(module, nn.Linear):
                for attr in ("scales", "biases"):
                    if hasattr(module, attr):
                        delattr(module, attr)

        logger.info("Nemotron-H: weight fixup complete")



    def _is_vlm_model(self) -> bool:
        """Check if this is a VLM model.

        Checks (in order):
        1. jang_config.json.architecture.has_vision (primary JANG metadata)
        2. preprocessor_config.json existence (VLM indicator)
        3. config.json.vision_config (existing MLX pattern)
        4. Model config for vision_config or VLM architectures (after load)
        """
        model_path = Path(self._model_name)

        # Check 1: JANG config architecture.has_vision (primary JANG metadata)
        try:
            for cfg_name in ("jang_config.json", "jjqf_config.json", "jang_cfg.json"):
                jang_config_path = model_path / cfg_name
                if jang_config_path.exists():
                    with open(jang_config_path) as f:
                        jang_config = json.load(f)
                    architecture = jang_config.get("architecture", {})
                    if isinstance(architecture, dict) and architecture.get("has_vision") is True:
                        return True
                    break  # Found JANG config, no need to check others
        except Exception:
            pass

        # Check 2: preprocessor_config.json existence (VLM indicator)
        try:
            preprocessor_path = model_path / "preprocessor_config.json"
            if preprocessor_path.exists():
                return True
        except Exception:
            pass

        # Check 3: config.json.vision_config (existing MLX pattern, fallback)
        try:
            config_path = model_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                if "vision_config" in config:
                    return True
                # Also check architectures for VLM hints
                arch = config.get("architectures", [])
                for a in arch:
                    if "VLM" in a or "VL" in a or "Vision" in a:
                        return True
        except Exception:
            pass

        # Check 4: Model config (after load)
        if self._model is not None:
            try:
                config = getattr(self._model, 'config', {})
                if isinstance(config, dict):
                    if "vision_config" in config:
                        return True
                    arch = config.get("architectures", [])
                    for a in arch:
                        if "VLM" in a or "VL" in a or "Vision" in a:
                            return True
            except Exception:
                pass

        return False

    async def start(self) -> None:
        """Start the engine (load JANG model if not loaded)."""
        if self._loaded:
            return

        from ..engine_core import AsyncEngineCore, EngineConfig
        from ..scheduler import SchedulerConfig

        # Check jang-tools is available
        self._check_jang_tools_available()

        # Validate that config.json and jang_config.json are consistent
        model_path = Path(self._model_name)
        config_path = model_path / "config.json"
        try:
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)

                # Check for common inconsistencies
                text_config = config.get("text_config", config)

                # Get hidden_size from text_config
                hidden_size = text_config.get("hidden_size", 0)
                num_attention_heads = text_config.get("num_attention_heads", 1)
                head_dim = text_config.get("head_dim", 0)

                # Verify consistency: hidden_size should equal num_attention_heads * head_dim
                if hidden_size > 0 and num_attention_heads > 0 and head_dim > 0:
                    expected_hidden = num_attention_heads * head_dim
                    if abs(hidden_size - expected_hidden) > 10:  # Allow small tolerance
                        logger.warning(
                            f"Potential config inconsistency: hidden_size={hidden_size} "
                            f"but num_attention_heads={num_attention_heads} * head_dim={head_dim} "
                            f"= {expected_hidden}. This may cause loading issues."
                        )

                # Check vocab_size consistency with embed_tokens
                vocab_size = text_config.get("vocab_size", 0)
                if vocab_size > 0:
                    logger.debug(f"Model vocab_size: {vocab_size}, hidden_size: {hidden_size}")
        except Exception as e:
            logger.debug(f"Config validation skipped: {e}")

        # Determine model type based on VLM indicators for jang-tools loader selection
        is_vlm = self._is_vlm_model()

        # Read the full config for jang-tools
        model_config = None
        try:
            if config_path.exists():
                with open(config_path) as f:
                    model_config = json.load(f)
                logger.info(
                    f"Model config loaded: model_type={model_config.get('model_type')}, "
                    f"architectures={model_config.get('architectures')}"
                )
                if "text_config" in model_config:
                    tc = model_config["text_config"]
                    logger.info(
                        f"text_config: hidden_size={tc.get('hidden_size')}, "
                        f"vocab_size={tc.get('vocab_size')}, "
                        f"num_hidden_layers={tc.get('num_hidden_layers')}"
                    )
        except Exception as e:
            logger.debug(f"Could not read full config: {e}")

        try:
            import jang_tools
            from jang_tools.loader import load_jang_model, load_jang_vlm_model

            # Determine correct loader based on VLM detection
            if is_vlm:
                logger.info(f"Loading JANG VLM model: {self._model_name}")
                self._model, self._processor = load_jang_vlm_model(self._model_name)
                # Extract tokenizer from processor for token counting
                if hasattr(self._processor, "tokenizer"):
                    self._tokenizer = self._processor.tokenizer
                else:
                    self._tokenizer = self._processor
                # Wrap model with VLMModelAdapter for BatchGenerator compatibility
                logger.info("Wrapping VLM model with VLMModelAdapter")
                self._model = VLMModelAdapter(self._model)
            else:
                logger.info(f"Loading JANG model: {self._model_name}")
                self._model, self._tokenizer = load_jang_model(self._model_name)
                self._processor = None

            # Wrap tokenizer if needed (some VLM processors don't have encode method)
            if self._tokenizer is not None and not hasattr(self._tokenizer, "encode"):
                if callable(self._tokenizer):
                    logger.debug("Wrapping callable tokenizer with encode() method")
                    self._tokenizer = _TokenizerWrapper(self._tokenizer)
                else:
                    logger.warning(
                        "Tokenizer has no encode() method and is not callable; "
                        "token counting may fail"
                    )

            # Apply Nemotron-H weight fixups before any dtype changes
            if self._detect_nemotron_h():
                logger.info("Detected Nemotron-H architecture, applying weight fixups")
                self._fix_nemotron_h_weights()

            # Auto-switch to bfloat16 for large expert models
            if self._needs_bfloat16():
                logger.info("Large expert model detected (512+ experts, hidden>=4096), using bfloat16")
                self._model.set_dtype(mx.bfloat16)

            # Create engine config (copy to avoid mutating the shared instance)
            scheduler_config = copy.copy(self._scheduler_config) if self._scheduler_config else SchedulerConfig()
            scheduler_config.model_name = self._model_name  # Ensure cache isolation per model
            engine_config = EngineConfig(
                model_name=self._model_name,
                scheduler_config=scheduler_config,
                stream_interval=self._stream_interval,
            )

            # Create async engine
            self._engine = AsyncEngineCore(
                model=self._model,
                tokenizer=self._tokenizer,
                config=engine_config,
            )

            await self._engine.engine.start()
            self._loaded = True
            logger.info(f"JANGLoader loaded: {self._model_name}")

        except ImportError as e:
            from ..exceptions import JANGDependencyError
            raise JANGDependencyError(
                f"Failed to import jang-tools: {e}. Install with: pip install jang[mlx]",
                model_name=self._model_name,
            )
        except Exception as e:
            from ..exceptions import JANGLoadError
            raise JANGLoadError(
                f"Failed to load JANG model {self._model_name}: {e}",
                model_name=self._model_name,
            ) from e

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        if self._engine:
            await self._engine.stop()
            self._engine.engine.close()
        self._engine = None
        self._model = None
        self._tokenizer = None
        self._loaded = False
        logger.info("JANGLoader stopped")

    def _preprocess_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Preprocess messages for model-specific formats."""
        try:
            from ..adapter.harmony import preprocess_harmony_messages
            if self.model_type == "gpt_oss":
                return preprocess_harmony_messages(messages)
        except ImportError:
            pass
        return messages

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Apply chat template to messages."""
        if hasattr(self._tokenizer, 'apply_chat_template'):
            is_partial = detect_and_strip_partial(messages)
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": not is_partial,
            }
            if is_partial:
                template_kwargs["continue_final_message"] = True
            if tools:
                template_kwargs["tools"] = tools
            if self._enable_thinking is not None:
                template_kwargs["enable_thinking"] = self._enable_thinking
            if chat_template_kwargs:
                template_kwargs.update(chat_template_kwargs)

            try:
                return self._tokenizer.apply_chat_template(messages, **template_kwargs)
            except TypeError:
                if chat_template_kwargs:
                    for key in chat_template_kwargs:
                        template_kwargs.pop(key, None)
                template_kwargs.pop("tools", None)
                template_kwargs.pop("enable_thinking", None)
                return self._tokenizer.apply_chat_template(messages, **template_kwargs)
            except Exception as e:
                logger.error(f"Chat template rendering failed: {e}")
                raise
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return prompt + "\nassistant:"

    def count_chat_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> int:
        """Count prompt tokens for chat messages."""
        messages = self._preprocess_messages(messages)
        template_tools = convert_tools_for_template(tools) if tools else None
        prompt = self._apply_chat_template(
            messages, template_tools, chat_template_kwargs=chat_template_kwargs
        )
        return len(self._tokenizer.encode(prompt))

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """Generate a complete response (non-streaming)."""
        if not self._loaded:
            await self.start()

        from ..request import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            stop=stop or [],
            thinking_budget=kwargs.get("thinking_budget", None),
        )

        output = await self._engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
        )

        text = clean_special_tokens(output.output_text)

        return GenerationOutput(
            text=text,
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
            finish_reason=output.finish_reason,
            tool_calls=output.tool_calls,
            cached_tokens=output.cached_tokens,
        )

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs,
    ) -> Any:
        """Stream generation token by token."""
        if not self._loaded:
            await self.start()

        from ..request import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            stop=stop or [],
            thinking_budget=kwargs.get("thinking_budget", None),
        )

        request_id = await self._engine.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
        )

        finished_normally = False
        try:
            async for output in self._engine.stream_outputs(request_id):
                text = clean_special_tokens(output.output_text)

                if output.finished:
                    finished_normally = True

                yield GenerationOutput(
                    text=text,
                    new_text=output.new_text,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    finished=output.finished,
                    finish_reason=output.finish_reason,
                    tool_calls=output.tool_calls,
                    cached_tokens=output.cached_tokens,
                )
        except GeneratorExit:
            logger.info(f"[stream_generate] GeneratorExit caught for request {request_id}")
        finally:
            if not finished_normally:
                logger.info(f"[stream_generate] Aborting request {request_id}")
                await self._engine.abort_request(request_id)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """Chat completion (non-streaming)."""
        if not self._loaded:
            await self.start()

        messages = self._preprocess_messages(messages)
        template_tools = convert_tools_for_template(tools) if tools else None
        ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        prompt = self._apply_chat_template(
            messages, template_tools, chat_template_kwargs=ct_kwargs
        )

        return await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            **kwargs,
        )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> Any:
        """Stream chat completion token by token."""
        if not self._loaded:
            await self.start()

        messages = self._preprocess_messages(messages)
        template_tools = convert_tools_for_template(tools) if tools else None
        ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        prompt = self._apply_chat_template(
            messages, template_tools, chat_template_kwargs=ct_kwargs
        )

        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            **kwargs,
        ):
            yield output

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        stats = {
            "engine_type": "jang",
            "model_name": self._model_name,
            "loaded": self._loaded,
            "stream_interval": self._stream_interval,
        }
        if self._engine:
            stats.update(self._engine.get_stats())
        return stats

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics."""
        if self._engine:
            return self._engine.get_cache_stats()
        return None
