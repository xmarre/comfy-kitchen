"""Regression coverage for setup.py's isolated CUDA CMake cache invalidation.

The helper is extracted from the AST to avoid executing setuptools.setup() or
probing a GPU merely to exercise the generated-build-directory policy.
"""

import ast
import os
import pathlib
import shutil


def _cache_reset():
    setup_path = pathlib.Path(__file__).resolve().parents[1] / "setup.py"
    ast_tree = ast.parse(setup_path.read_text(encoding="utf-8"))
    candidates = [
        node for node in ast_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "reset_cuda_cmake_cache_on_compiler_switch"
    ]
    assert len(candidates) == 1
    scope = {"pathlib": pathlib, "os": os, "shutil": shutil}
    code = compile(ast.Module(body=candidates, type_ignores=[]), str(setup_path), "exec")
    exec(code, scope)
    return scope[candidates[0].name]


def test_cuda_compiler_switch_removes_only_its_generated_backend_cache(tmp_path):
    reset = _cache_reset()
    root = tmp_path / "build" / "temp.linux-x86_64-cpython-312"
    cuda = root / "cuda"
    hip = root / "hip"
    cuda.mkdir(parents=True)
    hip.mkdir()
    (hip / "HIP_BUILD_SENTINEL").write_text("keep", encoding="utf-8")
    (cuda / "CMakeCache.txt").write_text(
        "CMAKE_CUDA_COMPILER:FILEPATH=/usr/bin/nvcc\n"
        "COMFY_CUDA_ARCHS:UNINITIALIZED=75-real;80-real;120f\n",
        encoding="utf-8",
    )
    (cuda / "CMakeFiles").mkdir()
    (cuda / "CMakeFiles" / "old.txt").write_text("old", encoding="utf-8")
    assert reset(cuda, pathlib.Path("/opt/conda/env/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc"))
    assert cuda.is_dir() and list(cuda.iterdir()) == []
    assert (hip / "HIP_BUILD_SENTINEL").read_text(encoding="utf-8") == "keep"


def test_unchanged_compiler_keeps_incremental_cmake_state(tmp_path):
    reset = _cache_reset()
    cache = tmp_path / "CMakeCache.txt"
    cache.write_text("CMAKE_CUDA_COMPILER:FILEPATH=/opt/cuda/bin/nvcc\n", encoding="utf-8")
    (tmp_path / "CMakeFiles").mkdir()
    assert not reset(tmp_path, pathlib.Path("/opt/cuda/bin/nvcc"))
    assert cache.exists() and (tmp_path / "CMakeFiles").is_dir()


def test_partial_cache_invalidation_removes_residual_cmakefiles(tmp_path):
    reset = _cache_reset()
    (tmp_path / "CMakeFiles").mkdir()
    (tmp_path / "CMakeFiles" / "stale.txt").write_text("old", encoding="utf-8")
    assert not reset(tmp_path, pathlib.Path("/opt/cuda/bin/nvcc"))
    assert not (tmp_path / "CMakeFiles").exists()


def test_cache_without_cuda_compiler_is_preserved(tmp_path):
    reset = _cache_reset()
    cache = tmp_path / "CMakeCache.txt"
    cache.write_text("COMFY_CUDA_ARCHS:UNINITIALIZED=120f\n", encoding="utf-8")
    assert not reset(tmp_path, pathlib.Path("/opt/cuda/bin/nvcc"))
    assert cache.exists()
