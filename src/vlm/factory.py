"""
VLM Factory - 统一的模型创建接口

支持的模型类型：
- Qwen3-VL
- Qwen2.5-VL
- Gemma-VL
- Claude API
- GPT-4V/OpenAI API
- Gemini API
"""

from typing import List
from .base import BaseVLM, VLMConfig, VLMType, Qwen3VL, Qwen2_5VL, GemmaVL, ClaudeAPI, GPT4VAPI, GeminiAPI, VLLMAPI


class VLMFactory:
    """VLM 工厂类 - 负责创建和管理 VLM 实例"""
    
    _vlm_classes = {
        VLMType.QWEN3_VL: Qwen3VL,
        VLMType.QWEN2_5_VL: Qwen2_5VL,
        VLMType.GEMMA_VL: GemmaVL,
        VLMType.CLAUDE_API: ClaudeAPI,
        VLMType.GPT4V_API: GPT4VAPI,
        VLMType.GEMINI_API: GeminiAPI,
        VLMType.VLLM_API: VLLMAPI,
    }

    @classmethod
    def _infer_vlm_type_from_name(cls, name_or_path: str):
        """Infer VLMType from model name/path string."""
        if not name_or_path:
            return None
        name = name_or_path.lower()

        if "qwen3" in name and "vl" in name:
            return VLMType.QWEN3_VL
        if ("qwen2.5" in name or "qwen2_5" in name or "qwen2-5" in name) and "vl" in name:
            return VLMType.QWEN2_5_VL
        if "gemma" in name and ("vision" in name or "vl" in name):
            return VLMType.GEMMA_VL
        if "claude" in name:
            return VLMType.CLAUDE_API
        if "gpt-4" in name or "gpt4" in name or name.startswith("gpt-") or name.startswith("o"):
            return VLMType.GPT4V_API
        if "gemini" in name:
            return VLMType.GEMINI_API
        return None
    
    @classmethod
    def create(cls, config: VLMConfig) -> BaseVLM:
        """
        创建 VLM 实例
        
        Args:
            config: VLMConfig 实例
            
        Returns:
            BaseVLM 实例
            
        Raises:
            ValueError: 如果 VLM 类型不被支持
        """
        if config.model_type in {VLMType.VLLM_API, VLMType.GPT4V_API, VLMType.CLAUDE_API, VLMType.GEMINI_API}:
            resolved_type = config.model_type
        else:
            inferred_type = cls._infer_vlm_type_from_name(config.model_path or config.model_name)
            resolved_type = inferred_type or config.model_type

        vlm_class = cls._vlm_classes.get(resolved_type)
        if not vlm_class:
            raise ValueError(
                f"Unknown VLM type: {resolved_type}\n"
                f"Available types: {cls.list_available()}"
            )
        config.model_type = resolved_type
        return vlm_class(config)
    
    @classmethod
    def register(cls, vlm_type: VLMType, vlm_class: type):
        """
        注册自定义 VLM 类
        
        Args:
            vlm_type: VLM 类型
            vlm_class: VLM 类（必须继承自 BaseVLM）
        """
        if not issubclass(vlm_class, BaseVLM):
            raise TypeError(f"{vlm_class} must be a subclass of BaseVLM")
        cls._vlm_classes[vlm_type] = vlm_class
    
    @classmethod
    def list_available(cls) -> List[str]:
        """列出所有可用的 VLM 类型"""
        return [vt.value for vt in cls._vlm_classes.keys()]
    
    @classmethod
    def get_class(cls, vlm_type: VLMType) -> type:
        """获取指定 VLM 类型的类"""
        return cls._vlm_classes.get(vlm_type)
