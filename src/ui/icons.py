"""内联 SVG 图标集。

Feather 风格描边图标，``fill="none" stroke="currentColor"``，随文字颜色自动变色；
纯内联字符串，离线可用，无需任何外部资源。
"""

from __future__ import annotations


def _icon(body: str, viewbox: str = "0 0 24 24") -> str:
    return (
        '<svg class="ui-ic" xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" '
        f'viewBox="{viewbox}" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        f"{body}</svg>"
    )


# 知识问答：放大镜
ICON_SEARCH = _icon(
    '<circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>'
)

# Agent 故障排查：脉搏图（诊断感）
ICON_PULSE = _icon('<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>')
