#!/usr/bin/env bash
# Installs Luminos als Desktop-Applikation (KDE / GNOME / XDG-kompatibel).
# Kein sudo nötig — alles landet unter ~/.local/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
PYTHON="$VENV/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "Fehler: Kein Python-Interpreter in $VENV gefunden."
    echo "Bitte zuerst: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

# 1 — Paket im editierbaren Modus installieren (erzeugt .venv/bin/luminos)
echo "→ Installiere Luminos-Paket (editable)…"
"$PYTHON" -m pip install -e "$SCRIPT_DIR" --quiet
LUMINOS_BIN="$VENV/bin/luminos"

# 2 — Wrapper-Skript in ~/.local/bin anlegen
mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/luminos" <<EOF
#!/usr/bin/env bash
exec "$LUMINOS_BIN" "\$@"
EOF
chmod +x "$HOME/.local/bin/luminos"
echo "→ Startskript: ~/.local/bin/luminos"

# 3 — Icon installieren
ICON_DIR="$HOME/.local/share/icons/hicolor/scalable/apps"
mkdir -p "$ICON_DIR"
cp "$SCRIPT_DIR/data/luminos.svg" "$ICON_DIR/luminos.svg"
echo "→ Icon:        $ICON_DIR/luminos.svg"

# 4 — Desktop-Eintrag installieren
DESKTOP_DIR="$HOME/.local/share/applications"
mkdir -p "$DESKTOP_DIR"
cp "$SCRIPT_DIR/luminos.desktop" "$DESKTOP_DIR/luminos.desktop"
echo "→ Desktop:     $DESKTOP_DIR/luminos.desktop"

# 5 — Datenbanken aktualisieren
update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true
gtk-update-icon-cache -f -t "$HOME/.local/share/icons/hicolor" 2>/dev/null || true

# KDE Plasma: sycoca neu aufbauen (macht den Eintrag im Starter sofort sichtbar)
if command -v kbuildsycoca6 &>/dev/null; then
    kbuildsycoca6 --noincremental 2>/dev/null || true
elif command -v kbuildsycoca5 &>/dev/null; then
    kbuildsycoca5 --noincremental 2>/dev/null || true
fi

echo ""
echo "✓ Fertig. Luminos ist jetzt im Anwendungsstarter verfügbar."
echo "  Falls der Eintrag fehlt: KDE-Sitzung neu starten oder 'kbuildsycoca6' ausführen."
