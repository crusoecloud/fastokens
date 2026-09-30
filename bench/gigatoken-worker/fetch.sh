#!/bin/sh
# Unpack gigatoken 0.10.0's source release (the one its PyPI wheel is built from)
# into vendor/, checking its hash.
set -eu
cd "$(dirname "$0")"
url=https://files.pythonhosted.org/packages/33/8a/fa097b404650a9eaea59de80bc8c33ad8ccb6b4a17ff6f33f213ec091057/gigatoken-0.10.0.tar.gz
sha=562fd4284eacdebd8a8043ce21ddcdacb61210a036e6e700cd13bdca2b2af7ac
mkdir -p vendor
[ -f vendor/gigatoken-0.10.0.tar.gz ] || curl -fsSL -o vendor/gigatoken-0.10.0.tar.gz "$url"
echo "$sha  vendor/gigatoken-0.10.0.tar.gz" | sha256sum -c -
tar xzf vendor/gigatoken-0.10.0.tar.gz -C vendor
