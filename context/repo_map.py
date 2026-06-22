"""
context/repo_map.py

Repo-map: compresses the entire repo structure into a summary string for injection into the system prompt.

Core approach (simplified version of Aider's repo-map):
1. Scan source files with tree-sitter to extract function/class definitions
2. Fall back to regex for languages not supported by or not installed for tree-sitter
3. Sort by "importance": top-level definitions > methods; smaller files are more likely to be core
4. Trim to a token budget and generate the summary string

## Multi-language support

tree-sitter requires a separate language package for each language:

    pip install tree-sitter-python       # Python (required)
    pip install tree-sitter-javascript   # JavaScript
    pip install tree-sitter-typescript   # TypeScript
    pip install tree-sitter-go           # Go
    pip install tree-sitter-rust         # Rust
    pip install tree-sitter-java         # Java

Unsupported languages silently degrade to regex parsing without errors.
To add a new language, add one line to _LANG_REGISTRY.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Language registry
# Format: file extension → (pip package name, module attribute name)
# Imported on demand at runtime; import failures are silently ignored and fall back to regex
# ---------------------------------------------------------------------------

_LANG_REGISTRY: dict[str, tuple[str, str]] = {
    ".py":  ("tree_sitter_python",     "language"),
    ".js":  ("tree_sitter_javascript", "language"),
    ".ts":  ("tree_sitter_typescript", "language_typescript"),
    ".tsx": ("tree_sitter_typescript", "language_tsx"),
    ".go":  ("tree_sitter_go",         "language"),
    ".rs":  ("tree_sitter_rust",       "language"),
    ".java":("tree_sitter_java",       "language"),
    ".cpp": ("tree_sitter_cpp",        "language"),
    ".c":   ("tree_sitter_c",          "language"),
    ".rb":  ("tree_sitter_ruby",       "language"),
}

# AST node type → symbol kind mapping (generic names across languages)
_FUNC_NODES: frozenset[str] = frozenset({
    "function_definition",       # Python, Go, C, C++
    "async_function_definition", # Python async def
    "function_declaration",      # JS, TS, Java
    "method_declaration",        # Java
    "method_definition",         # JS class method
    "function_item",             # Rust fn
    "arrow_function",            # JS arrow (skipped; usually anonymous)
})
_CLASS_NODES: frozenset[str] = frozenset({
    "class_definition",   # Python
    "class_declaration",  # JS, TS, Java
    "struct_item",        # Rust struct
    "impl_item",          # Rust impl
    "interface_declaration",  # TS/Java
})

# Directories to skip
_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    ".mypy_cache", ".pytest_cache", "dist", "build",
})

# Regex fallback: matches definition statements in common languages
_SYMBOL_RE = re.compile(
    r"^[ \t]*(def|class|function|func|fn|pub fn|async fn|async def"
    r"|public|private|protected|static)\s+(\w+)",
    re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Query-aware ranking helpers
#
# To make the repo-map relevant to the *current task* (not just globally
# "important" files), we score each file by lexical overlap between the task
# description and the file's path + symbol names. This is intentionally cheap
# (no embeddings / no network): a missing or empty query degrades silently to
# the original structural-only ordering.
# ---------------------------------------------------------------------------

# Common English + boilerplate words that carry no signal for ranking.
_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "with", "this", "that", "from", "into", "use", "using",
    "add", "fix", "bug", "issue", "make", "want", "need", "please", "code",
    "file", "files", "function", "method", "class", "test", "tests", "should",
    "would", "could", "when", "then", "else", "have", "has", "not", "but",
    "can", "all", "any", "new", "old", "get", "set", "run", "via", "are",
})

# Split an identifier token into camelCase / PascalCase / snake_case sub-words.
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z0-9]+|[A-Z]+|[0-9]+")


def _tokenize(text: str) -> set[str]:
    """
    Lowercase, split on non-alphanumerics, further split camelCase / snake_case,
    drop stopwords and tokens shorter than 3 chars. Returns a set of terms.

    Example:
        "fix the trimHistory token budget" -> {"trim", "history", "token", "budget"}
    """
    terms: set[str] = set()
    for raw in re.findall(r"[A-Za-z0-9]+", text or ""):
        parts = _CAMEL_RE.findall(raw) or [raw]
        for p in parts:
            p = p.lower()
            if len(p) >= 3 and p not in _STOPWORDS:
                terms.add(p)
    return terms

# Cache of loaded tree-sitter Language objects (avoids repeated imports)
_lang_cache: dict[str, object] = {}   # ext → Language or None


def _get_language(ext: str):
    """
    Get the tree-sitter Language object for a given file extension.
    Returns None when the package is not installed; callers fall back to regex.
    """
    if ext in _lang_cache:
        return _lang_cache[ext]

    entry = _LANG_REGISTRY.get(ext)
    if entry is None:
        _lang_cache[ext] = None
        return None

    module_name, attr_name = entry
    try:
        import importlib
        from tree_sitter import Language
        mod = importlib.import_module(module_name)
        lang_fn = getattr(mod, attr_name)
        lang = Language(lang_fn())
        _lang_cache[ext] = lang
        return lang
    except Exception:
        _lang_cache[ext] = None
        return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Symbol:
    """A single extracted symbol (function or class definition)."""
    name: str
    kind: str           # "function" | "class" | "method"
    line: int
    file: Path
    indent: int = 0

    @property
    def is_toplevel(self) -> bool:
        return self.indent == 0


@dataclass
class FileInfo:
    """Metadata and symbol list for one file."""
    path: Path
    size: int
    symbols: list[Symbol] = field(default_factory=list)
    relevance: float = 0.0   # query relevance for the current task (0 = no query / no match)

    @property
    def rel_path(self) -> str:
        return str(self.path)

    def importance_score(self) -> float:
        top_level = sum(1 for s in self.symbols if s.is_toplevel)
        size_penalty = self.size / 10_000
        return top_level - size_penalty

    def relevance_score(self, query_terms: set[str]) -> float:
        """
        Lexical relevance of this file to the task query.

        A path match (e.g. query term "budget" appearing in "token_budget.py")
        is a strong signal, so it is weighted higher than a match on a symbol
        name. Each query term counts at most once to avoid a file with many
        repetitive symbols dominating.
        """
        if not query_terms:
            return 0.0
        path_terms = _tokenize(self.rel_path)
        symbol_terms: set[str] = set()
        for s in self.symbols:
            symbol_terms |= _tokenize(s.name)

        score = 0.0
        for term in query_terms:
            if term in path_terms:
                score += 3.0       # filename/path match: strong
            elif term in symbol_terms:
                score += 1.0       # symbol-name match: weaker
        return score


# ---------------------------------------------------------------------------
# RepoMap
# ---------------------------------------------------------------------------

class RepoMap:
    """
    Scan the repo and generate a summary string.

    Usage:
        rm = RepoMap(repo_path="/path/to/repo")
        summary = rm.build(budget=8000)
    """

    def __init__(self, repo_path: str | Path) -> None:
        self._root = Path(repo_path).resolve()

    def build(self, budget: int = 8000, query: str | None = None) -> str:
        """
        Build the repo-map summary string.

        Args:
            budget: token budget for the summary (approximated as budget * 4 chars).
            query:  optional task description. When provided, files lexically
                    relevant to the task are ranked first (and marked), so the
                    most useful files survive the budget cut. When omitted, falls
                    back to the original structural-importance ordering.
        """
        files = self._scan()
        if not files:
            return "(empty repository)"

        query_terms = _tokenize(query) if query else set()
        for fi in files:
            fi.relevance = fi.relevance_score(query_terms)

        # Primary key: query relevance (relevant files first); tie-break by
        # structural importance. With no query, relevance is 0 for all files and
        # this reduces to the original importance-only ordering.
        files.sort(key=lambda f: (f.relevance, f.importance_score()), reverse=True)

        lines: list[str] = []
        char_count = 0
        max_chars = budget * 4

        if query_terms and any(f.relevance > 0 for f in files):
            lines.append("# Files most relevant to the task are listed first (marked ★).")

        for fi in files:
            block = self._format_file(fi)
            if char_count + len(block) > max_chars:
                remaining = len(files) - files.index(fi)
                lines.append(f"... ({remaining} more files not shown)")
                break
            lines.append(block)
            char_count += len(block)

        return "\n".join(lines)

    def _scan(self) -> list[FileInfo]:
        results: list[FileInfo] = []
        for path in sorted(self._root.rglob("*")):
            if any(part in _SKIP_DIRS for part in path.parts):
                continue
            if not path.is_file():
                continue
            size = path.stat().st_size
            if size > 500_000:
                continue

            fi = FileInfo(path=path.relative_to(self._root), size=size)
            ext = path.suffix.lower()

            if ext in _LANG_REGISTRY or ext in {".py", ".js", ".ts", ".go", ".rs"}:
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                    fi.symbols = _extract_symbols(content, fi.path, ext)
                except OSError:
                    pass

            results.append(fi)
        return results

    def _format_file(self, fi: FileInfo) -> str:
        sym_count = len(fi.symbols)
        marker = "★ " if fi.relevance > 0 else ""
        header = f"{marker}{fi.rel_path}"
        if sym_count:
            header += f" ({sym_count} symbol{'s' if sym_count != 1 else ''})"

        if not fi.symbols:
            return header + "\n"

        sym_lines = [header + ":"]
        for sym in fi.symbols:
            prefix = "    " if not sym.is_toplevel else "  "
            sym_lines.append(f"{prefix}{sym.kind} {sym.name} (line {sym.line})")
        return "\n".join(sym_lines) + "\n"


# ---------------------------------------------------------------------------
# Symbol extraction (exported for testing)
# ---------------------------------------------------------------------------

def _extract_symbols(content: str, filepath: Path, ext: str) -> list[Symbol]:
    """Choose the parsing method based on extension: tree-sitter (if installed) or regex fallback."""
    lang = _get_language(ext)
    if lang is not None:
        return _extract_with_treesitter(content, filepath, lang)
    return _extract_symbols_regex(content, filepath)


def _extract_with_treesitter(content: str, filepath: Path, lang) -> list[Symbol]:
    """Extract symbols with tree-sitter; fall back to regex on failure."""
    try:
        from tree_sitter import Parser
        parser = Parser(lang)
        tree = parser.parse(content.encode("utf-8", errors="replace"))
        return _walk_tree(tree.root_node, filepath)
    except Exception:
        return _extract_symbols_regex(content, filepath)


def _walk_tree(node, filepath: Path) -> list[Symbol]:
    """Recursively walk a tree-sitter AST and extract function and class definitions."""
    results: list[Symbol] = []
    ntype = node.type

    if ntype in _FUNC_NODES and ntype != "arrow_function":
        name_node = node.child_by_field_name("name")
        if name_node:
            indent = node.start_point[1]
            kind = "method" if indent > 0 else "function"
            results.append(Symbol(
                name=name_node.text.decode("utf-8", errors="replace"),
                kind=kind,
                line=node.start_point[0] + 1,
                file=filepath,
                indent=indent,
            ))
    elif ntype in _CLASS_NODES:
        name_node = node.child_by_field_name("name")
        if name_node:
            indent = node.start_point[1]
            results.append(Symbol(
                name=name_node.text.decode("utf-8", errors="replace"),
                kind="class",
                line=node.start_point[0] + 1,
                file=filepath,
                indent=indent,
            ))

    for child in node.children:
        results.extend(_walk_tree(child, filepath))

    return results


# Keep original function name for test compatibility
def _extract_python_symbols(content: str, filepath: Path) -> list[Symbol]:
    """Backward-compatible alias; test files call this name."""
    return _extract_symbols(content, filepath, ".py")


def _extract_symbols_regex(content: str, filepath: Path) -> list[Symbol]:
    """Regex fallback; supports multiple languages."""
    symbols: list[Symbol] = []
    for lineno, line in enumerate(content.splitlines(), start=1):
        m = _SYMBOL_RE.match(line)
        if not m:
            continue
        keyword = m.group(1)
        name = m.group(2)
        # Skip Java/JS modifier false matches (public/private followed by a type, not a name)
        if keyword in ("public", "private", "protected", "static"):
            continue
        indent = len(line) - len(line.lstrip())
        if keyword == "class":
            kind = "class"
        elif indent > 0:
            kind = "method"
        else:
            kind = "function"
        symbols.append(Symbol(
            name=name, kind=kind, line=lineno,
            file=filepath, indent=indent,
        ))
    return symbols
