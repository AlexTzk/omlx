# SPDX-License-Identifier: Apache-2.0
"""Tests for JANG-specific discovery and live JANGLoader behavior."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.engine.jang import JANGLoader
from omlx.model_discovery import detect_model_type, discover_models


class _FakeArray:
    """Small array-like test double for patched mlx calls."""

    def __init__(self, shape):
        self.shape = shape

    def astype(self, _dtype):
        return self


class TestDetectModelTypeJangVlm:
    """Tests for production JANG VLM detection in model_discovery."""

    def test_detect_vlm_via_jang_has_vision(self, tmp_path):
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"version": "1.0", "architecture": {"has_vision": True}})
        )
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "jang_vlm"}))

        assert detect_model_type(tmp_path) == "vlm"

    def test_detect_text_only_jang_as_llm(self, tmp_path):
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"version": "1.0", "architecture": {"has_vision": False}})
        )
        (tmp_path / "preprocessor_config.json").write_text(
            json.dumps({"processor_type": "AutoProcessor"})
        )
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))

        assert detect_model_type(tmp_path) == "llm"

    def test_detect_vlm_via_preprocessor_only_for_jang(self, tmp_path):
        (tmp_path / "jjqf_config.json").write_text(json.dumps({"quantization": {"bits": 2}}))
        (tmp_path / "preprocessor_config.json").write_text(
            json.dumps({"processor_type": "AutoProcessor"})
        )
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))

        assert detect_model_type(tmp_path) == "vlm"

    def test_discover_jang_vlm_uses_jang_engine(self, tmp_path):
        model_dir = tmp_path / "qwen3.5-vlm-jang"
        model_dir.mkdir()
        (model_dir / "jang_config.json").write_text(
            json.dumps(
                {
                    "format": "jang",
                    "format_version": "2.0",
                    "architecture": {"has_vision": True},
                }
            )
        )
        (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
        (model_dir / "model.safetensors").write_bytes(b"0" * 1000)

        models = discover_models(tmp_path)
        assert models["qwen3.5-vlm-jang"].model_type == "vlm"
        assert models["qwen3.5-vlm-jang"].engine_type == "jang"


class TestJANGLoader:
    """Tests for JANGLoader behavior that is exercised in production."""

    def test_is_jang_v2_reads_format_version(self, tmp_path):
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"format": "jang", "format_version": "2.0"})
        )

        loader = JANGLoader(str(tmp_path))

        assert loader._is_jang_v2() is True

    def test_should_use_vlm_loader_when_discovery_detects_vlm(self, tmp_path):
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"format": "jang", "format_version": "2.0", "architecture": {}})
        )
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "mistral3", "vision_config": {"hidden_size": 1024}})
        )

        loader = JANGLoader(str(tmp_path))

        assert loader._should_use_vlm_loader() is True

    def test_should_not_use_vlm_loader_when_jang_explicitly_says_no_vision(self, tmp_path):
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"format": "jang", "format_version": "2.0", "architecture": {"has_vision": False}})
        )
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "mistral3", "vision_config": {"hidden_size": 1024}})
        )

        loader = JANGLoader(str(tmp_path))

        assert loader._should_use_vlm_loader() is False

    @pytest.mark.asyncio
    async def test_start_skips_nemotron_fixup_for_jang_v2(self, tmp_path):
        (tmp_path / "jang_config.json").write_text(
            json.dumps({"format": "jang", "format_version": "2.0"})
        )
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "nemotron_h",
                    "architectures": ["NemotronHForCausalLM"],
                    "hidden_size": 4096,
                    "num_attention_heads": 32,
                    "head_dim": 128,
                    "vocab_size": 131072,
                }
            )
        )

        loader = JANGLoader(str(tmp_path))
        fake_model = MagicMock()
        fake_model.config = {"architectures": ["NemotronHForCausalLM"]}
        fake_engine = MagicMock()
        fake_engine.engine.start = AsyncMock()

        with patch.object(loader, "_check_jang_tools_available"), patch.object(
            loader, "_should_use_vlm_loader", return_value=False
        ), patch(
            "jang_tools.loader.load_jang_model", return_value=(fake_model, MagicMock())
        ), patch.object(
            loader, "_fix_nemotron_h_weights"
        ) as fixup, patch.object(
            loader, "_needs_bfloat16", return_value=False
        ), patch(
            "omlx.engine_core.AsyncEngineCore", return_value=fake_engine
        ), patch(
            "omlx.engine_core.EngineConfig", side_effect=lambda **kwargs: SimpleNamespace(**kwargs)
        ), patch(
            "omlx.scheduler.SchedulerConfig", return_value=SimpleNamespace(model_name=None)
        ):
            await loader.start()

        fixup.assert_not_called()

    def test_fix_nemotron_h_weights_no_index_is_noop(self, tmp_path):
        loader = JANGLoader(str(tmp_path))
        loader._model = MagicMock()

        loader._fix_nemotron_h_weights()

        loader._model.load_weights.assert_not_called()

    def test_fix_nemotron_h_weights_dequantizes_gate_weights(self, tmp_path):
        loader = JANGLoader(str(tmp_path))
        loader._model = MagicMock()

        index = {
            "weight_map": {
                "backbone.layers.0.mixer.gate.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.0.mixer.gate.scales": "model-00001-of-00001.safetensors",
                "backbone.layers.0.mixer.gate.biases": "model-00001-of-00001.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

        gate_weight = _FakeArray((2, 32))
        scales = _FakeArray((2, 1))
        biases = _FakeArray((2, 1))
        dequantized = _FakeArray((2, 128))

        with patch.object(loader, "_needs_bfloat16", return_value=False), patch(
            "omlx.engine.jang.mx.load",
            return_value={
                "backbone.layers.0.mixer.gate.weight": gate_weight,
                "backbone.layers.0.mixer.gate.scales": scales,
                "backbone.layers.0.mixer.gate.biases": biases,
            },
        ) as mock_load, patch(
            "omlx.engine.jang.mx.dequantize", return_value=dequantized
        ) as mock_dequantize, patch(
            "omlx.engine.jang.mx.float16", "float16"
        ):
            loader._fix_nemotron_h_weights()

        mock_load.assert_called_once()
        mock_dequantize.assert_called_once_with(gate_weight, scales, biases, 128, 8)
        loader._model.load_weights.assert_called_once_with(
            [("backbone.layers.0.mixer.gate.weight", dequantized)],
            strict=False,
        )

    @pytest.mark.asyncio
    async def test_generate_uses_engine_and_cleans_output(self):
        loader = JANGLoader("/tmp/jang-model")
        loader._loaded = True
        loader._engine = MagicMock()
        loader._engine.generate = AsyncMock(
            return_value=SimpleNamespace(
                output_text="hello<|end|>",
                prompt_tokens=11,
                completion_tokens=7,
                finish_reason="stop",
                tool_calls=[],
                cached_tokens=3,
            )
        )

        with patch("omlx.engine.jang.clean_special_tokens", return_value="hello") as clean:
            output = await loader.generate("prompt", max_tokens=9, temperature=0.2)

        clean.assert_called_once_with("hello<|end|>")
        assert output.text == "hello"
        assert output.prompt_tokens == 11
        assert output.completion_tokens == 7
        sampling_params = loader._engine.generate.await_args.kwargs["sampling_params"]
        assert sampling_params.max_tokens == 9
        assert sampling_params.temperature == 0.2
