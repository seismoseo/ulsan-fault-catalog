#!/bin/sh
# Configure the local git "clean" filter that strips Jupyter notebook outputs
# on commit. Filter config lives in .git/config (not committed), so run this
# ONCE per clone:
#
#     bash tools/setup-git-filters.sh
#
set -e
repo_root=$(git rev-parse --show-toplevel)
git config filter.nbstripout.clean "python '$repo_root/tools/nbstrip.py'"
git config filter.nbstripout.smudge cat
echo "OK: notebook output-stripping filter configured (clean = tools/nbstrip.py)."
echo "    Notebook outputs are now removed automatically on 'git add' / commit;"
echo "    your working copies keep their rendered outputs."
