# SPDX-License-Identifier: Apache-2.0
"""
JANG model engine for loading quantized MoE+SSM hybrid models.

This engine wraps the jang-tools package to load JANG quantized models
with mixed-precision quantization (attention 6-8-bit, experts 2-4-bit).
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from ..api.tool_calling import convert_tools_for_template
from ..api.utils import clean_special_tokens, detect_and_strip_partial
from .base import BaseEngine, GenerationOutput

logger = logging.getLogger(__name__)


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
        trust_remote_code: Whether to trust remote code
        scheduler_config: Optional scheduler configuration
        stream_interval: Tokens to batch before streaming (1=every token)
        enable_thinking: Enable thinking mode for reasoning models
        model_settings: Optional per-model settings for post-load transforms
    """

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = True,
        scheduler_config: Any | None = None,
        stream_interval: int = 1,
        enable_thinking: bool | None = None,
        model_settings: Any | None = None,
    ):
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._scheduler_config = scheduler_config
        self._stream_interval = stream_interval
        self._enable_thinking = enable_thinking
        self._model_settings = model_settings
        self._scheduler_config = scheduler_config
        self._stream_interval = stream_interval
        self._enable_thinking = enable_thinking
        self._model_settings = model_settings

        self._model = None
        self._tokenizer = None
        self._engine = None
        self._loaded = False

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

    def _detect_nemotron_h(self) -> bool:
        """Check if model is Nemotron-H architecture."""
        if self._model is None:
            return False
        try:
            config = getattr(self._model, 'config', {})
            if isinstance(config, dict):
                arch = config.get('architectures', [])
                for a in arch:
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
            config = getattr(self._model, 'config', {})
            if isinstance(config, dict):
                n_experts = config.get('num_local_experts', 0)
                hidden_size = config.get('hidden_size', 0)
                if n_experts >= 512 and hidden_size >= 4096:
                    return True
            # Try other config locations
            if hasattr(self._model, 'config'):
                cfg = self._model.config
                if hasattr(cfg, 'num_local_experts'):
                    if cfg.num_local_experts >= 512:
                        if hasattr(cfg, 'hidden_size') and cfg.hidden_size >= 4096:
                            return True
        except Exception as e:
            logger.debug(f"Error checking bfloat16 requirements: {e}")
        return False

    def _apply_weight_renaming(self) -> None:
        """Apply Nemotron-H weight renaming (up_proj->fc1, down_proj->fc2)."""
        if self._model is None:
            return
        try:
            import mlx.core as mx

            def _rename_weights():
                if not hasattr(self._model, 'modules'):
                    return

                renamed = []
                for module_name, module in self._model.modules():
                    for weight_name in ['up_proj', 'down_proj']:
                        if weight_name in module_name and weight_name not in ['gate_proj', 'win_proj', 'w1', 'w2', 'w3']:
                            new_name = module_name.replace(weight_name, 'fc1' if weight_name == 'up_proj' else 'fc2')
                            if hasattr(module, weight_name):
                                weight = getattr(module, weight_name)
                                setattr(module, new_name, weight)
                                delattr(module, weight_name)
                                renamed.append(f"{module_name}.{weight_name} -> {new_name}")

                if renamed:
                    logger.info(f"Nemotron-H: renamed {len(renamed)} weights")

            mx.eval(mx.jit(_rename_weights)())
        except Exception as e:
            logger.warning(f"Weight renaming failed: {e}")

    def _is_vlm_model(self) -> bool:
        """Check if this is a VLM model."""
        if self._model is None:
            return False
        try:
            config = getattr(self._model, 'config', {})
            if isinstance(config, dict):
                if 'vision_config' in config:
                    return True
                arch = config.get('architectures', [])
                for a in arch:
                    if 'VLM' in a or 'VL' in a or 'Vision' in a:
                        return True
            return False
        except Exception:
            return False

    async def start(self) -> None:
        """Start the engine (load JANG model if not loaded)."""
        if self._loaded:
            return

        import asyncio

        from ..engine_core import AsyncEngineCore, EngineConfig

        # Check jang-tools is available
        self._check_jang_tools_available()

        try:
            import jang_tools
            from jang_tools.loader import load_jang_model, load_jang_vlm_model

            # Load model based on type
            if self._is_vlm_model():
                logger.info(f"Loading JANG VLM model: {self._model_name}")
                self._model, self._tokenizer = await load_jang_vlm_model(
                    self._model_name,
                    trust_remote_code=self._trust_remote_code,
                )
            else:
                logger.info(f"Loading JANG model: {self._model_name}")
                self._model, self._tokenizer = await load_jang_model(
                    self._model_name,
                    trust_remote_code=self._trust_remote_code,
                )

            # Handle Nemotron-H architecture
            if self._detect_nemotron_h():
                logger.info("Detected Nemotron-H architecture, applying weight renaming")
                self._apply_weight_renaming()

            # Auto-switch to bfloat16 for large expert models
            if self._needs_bfloat16():
                logger.info("Large expert model detected (512+ experts, hidden>=4096), using bfloat16")
                import mlx.core as mx
                self._model = mx.cast(self._model, mx.bfloat16)

            # Create engine config
            scheduler_config = copy.copy(self._scheduler_config) if self._scheduler_config else None
            if scheduler_config:
                scheduler_config.model_name = self._model_name
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
