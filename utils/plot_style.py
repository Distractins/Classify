"""统一中文字体；服务器可安装 Noto CJK 或提供本地字体文件。"""
from pathlib import Path
import os
import warnings

_configured = False


def configure_chinese_font():
    global _configured
    if _configured:
        return
    import matplotlib
    from matplotlib import font_manager
    custom = os.environ.get("HMU_FONT_PATH")
    if custom:
        font_manager.fontManager.addfont(str(Path(custom).expanduser()))
        preferred = font_manager.FontProperties(fname=custom).get_name()
    else:
        preferred = None
    candidates = [preferred] if preferred else []
    candidates += ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans CJK JP",
                   "Source Han Sans SC", "WenQuanYi Zen Hei", "SimSun", "Arial Unicode MS"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = [name for name in candidates if name in available]
    if not selected:
        warnings.warn("未检测到中文字体，图中文字可能缺失。请安装 Noto CJK，或设置 HMU_FONT_PATH 指向中文字体文件。")
    matplotlib.rcParams["font.sans-serif"] = selected + ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    _configured = True
