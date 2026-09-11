from __future__ import annotations

import ast
from pathlib import Path


_LEGACY_TAIL = [
    "rope_eps",
    "tail",
    "block_len",
    "coarse_gate",
    "token_aug",
]


def _chunked_parameters(path: str) -> list[str]:
    tree = ast.parse(Path(path).read_text())
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "sol_attn_chunked"
    ]
    assert len(matches) == 1
    fn = matches[0]
    return [arg.arg for arg in (*fn.args.posonlyargs, *fn.args.args)]


def test_chunked_key_bias_is_appended_after_legacy_positional_tail():
    for path in (
        "comfy_kitchen/backends/cuda/__init__.py",
        "comfy_kitchen/backends/hip/__init__.py",
    ):
        params = _chunked_parameters(path)
        assert params[-6:] == [*_LEGACY_TAIL, "key_bias"]
