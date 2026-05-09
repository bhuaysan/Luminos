#!/usr/bin/env bash
# Build Luminos-x86_64.AppImage
# Usage: ./build_appimage.sh [--clean]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APPDIR="dist/Luminos.AppDir"
OUTPUT="Luminos-x86_64.AppImage"
APPIMAGETOOL="appimagetool.AppImage"

if [[ "${1:-}" == "--clean" ]]; then
    rm -rf dist/luminos dist/Luminos.AppDir build
    echo "Cleaned build artifacts."
fi

# ── 1. Python dependencies ────────────────────────────────────────────────────
echo "Installing / verifying dependencies …"
.venv/bin/pip install --quiet pyinstaller

# ── 2. PyInstaller bundle ─────────────────────────────────────────────────────
echo "Running PyInstaller …"
.venv/bin/pyinstaller luminos.spec --clean --noconfirm

# ── 3. AppDir scaffolding ─────────────────────────────────────────────────────
echo "Building AppDir …"
mkdir -p "$APPDIR"
cp -a dist/luminos/. "$APPDIR/"

# Icon (256×256 PNG)
magick -background none data/luminos.svg -resize 256x256 "$APPDIR/luminos.png"
ln -sf luminos.png "$APPDIR/.DirIcon"

# Desktop entry
cp luminos.desktop "$APPDIR/luminos.desktop"
sed -i 's|^Exec=.*|Exec=luminos %F|' "$APPDIR/luminos.desktop"

# AppStream metadata
mkdir -p "$APPDIR/usr/share/metainfo"
cp dist/Luminos.AppDir/usr/share/metainfo/luminos.appdata.xml \
   "$APPDIR/usr/share/metainfo/luminos.appdata.xml" 2>/dev/null || true

# AppRun launcher
cat > "$APPDIR/AppRun" << 'APPRUN'
#!/bin/bash
SELF="$(readlink -f "$0")"
HERE="$(dirname "$SELF")"
export QT_PLUGIN_PATH="${HERE}/_internal/PySide6/Qt/plugins"
export QT_QPA_PLATFORM_PLUGIN_PATH="${HERE}/_internal/PySide6/Qt/plugins/platforms"
if [ -z "$DISPLAY" ] && [ -n "$WAYLAND_DISPLAY" ]; then
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-wayland}"
else
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
fi
exec "${HERE}/luminos" "$@"
APPRUN
chmod +x "$APPDIR/AppRun"

# ── 4. appimagetool ───────────────────────────────────────────────────────────
if [[ ! -x "$APPIMAGETOOL" ]]; then
    echo "Downloading appimagetool …"
    wget -q --show-progress \
        "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage" \
        -O "$APPIMAGETOOL"
    chmod +x "$APPIMAGETOOL"
fi

# ── 5. Package ────────────────────────────────────────────────────────────────
echo "Packaging AppImage …"
ARCH=x86_64 ./"$APPIMAGETOOL" "$APPDIR" "$OUTPUT"

echo ""
echo "Done: $(ls -lh "$OUTPUT" | awk '{print $5, $9}')"
