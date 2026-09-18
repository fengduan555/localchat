"""临时恢复版启动器。

app.py 源码在 2026-09-17 22:24 被意外清空 (0 字节), git 未提交, staging 亦为空。
本文件通过 __pycache__/app.cpython-312.pyc (20:06 编译, 行为与丢失版本一致, 含
autostop / memory / relationship 全部特性) 的字节码恢复运行, 保证服务可用。

真正的可读源码重建完成后, 直接用源码替换本启动器即可。
"""
import importlib.util
import pathlib
import sys

_pyc = pathlib.Path(__file__).resolve().parent / "app_recovered.pyc"
_spec = importlib.util.spec_from_file_location("__main__", _pyc)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["__main__"] = _mod
_spec.loader.exec_module(_mod)
