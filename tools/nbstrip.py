#!/usr/bin/env python3
"""Git *clean* filter: strip Jupyter notebook outputs before they are stored.

Reads a notebook from stdin and writes a cleaned notebook to stdout, so version
control stays small and diffs stay readable. Working-tree files are left
untouched (a clean filter only affects what git records in the index/history).

Dependency-free (stdlib only). Wired up via `.gitattributes`
(`*.ipynb filter=nbstripout`) + `tools/setup-git-filters.sh`.
"""
import sys
import json


def strip(nb):
    nb.pop("signature", None)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        meta = cell.get("metadata", {})
        for k in ("execution", "collapsed", "scrolled"):
            meta.pop(k, None)
    # drop the kernel patch-version, which churns across machines
    lang = nb.get("metadata", {}).get("language_info", {})
    lang.pop("version", None)
    return nb


def main():
    raw = sys.stdin.read()
    try:
        nb = json.loads(raw)
    except Exception:
        sys.stdout.write(raw)   # not valid JSON — pass through unchanged
        return
    json.dump(strip(nb), sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
