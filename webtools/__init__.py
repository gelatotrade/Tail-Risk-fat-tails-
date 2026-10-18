"""
webtools - the bridge between the tail-risk framework and the browser.

Every feature of ``src/`` is exposed as a *tool*: a small module in
``webtools/tools/`` that declares a ``SPEC`` (title, parameters, what data it
needs) and a ``run(params, data, out)`` function.  The same code runs in
CPython (tests, ``python -m webtools``) and in Pyodide inside the browser's
Web Worker, which is how the GitHub Pages site executes the real framework
without a server.
"""

from .core import Result, ToolError, run_tool, tool_specs

__all__ = ["Result", "ToolError", "run_tool", "tool_specs"]
