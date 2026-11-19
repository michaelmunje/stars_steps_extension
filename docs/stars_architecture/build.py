"""Build stars_architecture.tex from stars_architecture.src.tex.

Every ``%%SNIP path start end caption%%`` line is replaced by the code of ``path`` lines ``start..end`` at
the pinned commit (read with ``git show``, so the snippets are exactly the committed code) and a GitHub
permalink to those lines. Then compile with tectonic.

    python docs/stars_architecture/build.py && tectonic docs/stars_architecture/stars_architecture.tex
"""
import re
import subprocess
import textwrap
from pathlib import Path

COMMIT = "5d37a3a459c37f0629b9c0044c9cba02300095fb"
REPO = "https://github.com/michaelmunje/stars_steps_extension"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASCII = {"—": "--", "–": "-", "→": "->", "×": "x", "≤": "<=", "≥": ">=", "‘": "'",
         "’": "'", "“": '"', "”": '"', "μ": "mu", "σ": "sigma", "≈": "~", "…": "..."}


def snippet(path: str, start: int, end: int, caption: str) -> str:
    source = subprocess.run(["git", "show", f"{COMMIT}:{path}"], cwd=ROOT, check=True, capture_output=True,
                            text=True).stdout.splitlines()
    code = textwrap.dedent("\n".join(source[start - 1:end]))
    for a, b in ASCII.items():
        code = code.replace(a, b)
    url = f"{REPO}/blob/{COMMIT}/{path}#L{start}-L{end}"
    name = path.split("/")[-1].replace("_", "\\_")
    return (f"\\par\\smallskip\\noindent{{\\small\\textbf{{{caption}}}\\hfill\\href{{{url}}}"
            f"{{\\texttt{{{name}:{start}--{end}}}}}}}\\par\\nopagebreak\n"
            f"\\begin{{lstlisting}}[aboveskip=2pt]\n{code}\n\\end{{lstlisting}}")


def main() -> None:
    text = (HERE / "stars_architecture.src.tex").read_text()
    pattern = re.compile(r"^%%SNIP (\S+) (\d+) (\d+) (.+?)%%$", re.M)
    text = pattern.sub(lambda m: snippet(m[1], int(m[2]), int(m[3]), m[4]), text)
    text = text.replace("@@COMMIT@@", COMMIT).replace("@@SHORT@@", COMMIT[:7]).replace("@@REPO@@", REPO)
    (HERE / "stars_architecture.tex").write_text(text)


if __name__ == "__main__":
    main()
