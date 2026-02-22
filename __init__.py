"""
ComfyUI_QSSIM — Quaternion SSIM 화질 평가 커스텀 노드
"""

from .qssim_node import QSSIMNode

NODE_CLASS_MAPPINGS = {
    "QSSIMNode": QSSIMNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QSSIMNode": "QSSIM Quality Score",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
