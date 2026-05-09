<div align="center">
  <img src="data/luminos.svg" width="96" alt="Luminos logo" />
  <h1>Luminos</h1>
  <p>Film negative converter for Linux — RAW &amp; TIFF scans to high-quality positives</p>

  ![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
  ![PySide6](https://img.shields.io/badge/PySide6-Qt6-41cd52?logo=qt&logoColor=white)
  ![Platform](https://img.shields.io/badge/Platform-Linux-orange?logo=linux&logoColor=white)
  ![License](https://img.shields.io/badge/License-MIT-green)
</div>

---

Luminos is a native Linux desktop application focused on one task: converting analog film negative scans into positive images. It handles C-41 colour negatives with automatic orange-mask compensation and black-and-white negatives, and ships with 20 built-in film profiles for the most common emulsions.

## Features

### Conversion
- **C-41 colour negatives** — automatic orange-mask detection and compensation
- **Black-and-white negatives** — luma-based inversion with auto-stretch
- **RAW support** — NEF, CR2, CR3, ARW, DNG, ORF, RAF, RW2 via rawpy
- **TIFF support** — 16-bit linear scans via tifffile / pyvips

### Editing
| Control | Range |
|---|---|
| Exposure | ±3 EV |
| Contrast | ±100 |
| Highlights / Shadows | ±100 |
| Black point / White point | 0–100 |
| Temperature / Tint | ±100 |
| Vibrance / Saturation | ±100 |
| Sharpening | 0–100 (unsharp mask) |
| Noise reduction | 0–100 (Gaussian) |
| Vignette | ±100 |
| Film grain | 0–100 |
| Split toning | Shadow/Highlight hue + strength + balance |
| Tone curves | Master + per-channel RGB spline editor |

### Workflow
- Live histogram (RGB + luminance, linear/log scale)
- Before/After and split-view comparison
- Clipping overlay (highlight reds, shadow blues)
- Crop with aspect-ratio lock + rule-of-thirds grid
- Rotation: 90° steps and ±15° fine adjustment
- WB pipette (click to sample) and one-click auto white balance
- Auto exposure/levels suggestion
- Undo/redo — 50 steps for edits, separate stack for crop & rotation
- Drag & drop import
- Batch export with per-image settings
- Session save/load (`.luminos` files)
- sRGB ICC profile embedded in all exports
- EXIF metadata preserved on JPEG export

### Export formats
| Format | Depth | Notes |
|---|---|---|
| TIFF | **16-bit** | Lossless; pyvips backend for large files |
| PNG | 8-bit | Lossless; ICC + metadata chunks |
| JPEG | 8-bit | Quality 1–95; ICC + EXIF merge |

---

## Built-in Film Profiles

**Colour (C-41)**

| Kodak | Fuji | Other |
|---|---|---|
| Portra 160 / 400 / 800 | Superia 200 / 400 | Agfa Vista 200 |
| Ektar 100 | Pro 400H | CineStill 800T |
| Gold 200 | | |
| ColorPlus 200 | | |

**Black & White**

Ilford HP5 Plus · FP4 Plus · Pan F Plus · Delta 100 / 400  
Kodak Tri-X 400 · T-MAX 100 / 400  
Fomapan 100

Custom profiles can be placed in `~/.local/share/luminos/profiles/` — see [Custom Profiles](#custom-profiles).

---

## Requirements

| Dependency | Version | Notes |
|---|---|---|
| Python | ≥ 3.11 | |
| PySide6 | ≥ 6.7 | Qt6 widgets |
| rawpy | ≥ 0.21 | RAW decoding |
| NumPy | ≥ 1.26 | |
| Pillow | ≥ 10.0 | PNG/JPEG export, ICC profiles |
| tifffile | ≥ 2023.1 | 16-bit TIFF I/O |
| pyvips | optional | Faster TIFF I/O for large files |
| piexif | optional | JPEG EXIF merge |

---

## Installation

### Option A — AppImage (no dependencies needed)

Download the latest `Luminos-x86_64.AppImage` from the [Releases](../../releases) page, then:

```bash
chmod +x Luminos-x86_64.AppImage
./Luminos-x86_64.AppImage
```

> Requires a 64-bit Linux with glibc ≥ 2.42 (Ubuntu 24.04+, Fedora 40+, and equivalents).

### Option B — From source

```bash
git clone https://github.com/example/luminos.git
cd luminos

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Optional extras
.venv/bin/pip install pyvips piexif
```

**Desktop integration** (application menu entry, no sudo required):

```bash
bash install-desktop.sh
```

This installs a launcher to `~/.local/bin/luminos`, the icon to `~/.local/share/icons/`, and the `.desktop` entry to `~/.local/share/applications/`.

---

## Usage

### GUI

```bash
# From venv
.venv/bin/python luminos_gui.py

# After install-desktop.sh
luminos

# AppImage
./Luminos-x86_64.AppImage
```

### CLI

For headless/batch use without a display:

```bash
python luminos_cli.py input.tif output.tif
python luminos_cli.py scan.NEF positive.tif --exposure 0.5 --wb 1.1 1.0 0.85
```

| Flag | Description |
|---|---|
| `--exposure STOPS` | Exposure adjustment in EV stops (default: 0.0) |
| `--wb R G B` | White balance RGB multipliers (default: 1.0 1.0 1.0) |

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `←` / `→` | Previous / next image |
| `F` | Fit to window |
| `1` | 100% zoom |
| `+` / `-` | Zoom in / out |
| `\` | Toggle before/after |
| `P` | Toggle WB pipette |
| `C` | Toggle crop mode |
| `Enter` / `Escape` | Confirm / exit crop |
| `R` | Reset all sliders |
| `Z` | Reset crop & rotation |
| `Ctrl+Z` / `Ctrl+Y` | Undo / Redo |
| `Ctrl+O` | Import files |
| `Ctrl+E` | Export active image |
| `Ctrl+Shift+E` | Batch export |
| `Ctrl+S` | Save session |

---

## Custom Profiles

Profiles are JSON files. Place them in `~/.local/share/luminos/profiles/` to override or extend the built-ins.

```jsonc
{
  "name": "My Custom Film",
  "type": "color_negative",       // or "bw_negative"
  "orange_mask": [0.82, 0.45, 0.20],
  "exposure_stops": 0.3,
  "white_balance": [1.05, 1.0, 0.92],
  "contrast": 10,
  "saturation": 1.1,
  "highlights": -5,
  "shadows": 8,
  "curve_points": {
    "master": [[0,0], [0.25,0.22], [0.75,0.80], [1,1]]
  }
}
```

A user profile whose `"name"` matches a built-in replaces it automatically.

---

## Building the AppImage

```bash
bash build_appimage.sh          # incremental build
bash build_appimage.sh --clean  # full clean rebuild
```

Requires: `python3`, a working `.venv`, and internet access to download `appimagetool` on the first run.

---

## Project Structure

```
luminos/
├── core/        # Pipeline, inversion, colour, tone curves
├── io/          # RAW loader, TIFF loader, export, EXIF
├── profiles/    # Profile loader, schema validator, built-in JSONs
└── ui/          # Main window, widgets, workers, histogram, dialogs
luminos_gui.py   # GUI entry point
luminos_cli.py   # CLI entry point
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## AI Disclosure

This project was developed with the assistance of [Claude](https://claude.ai) (Anthropic).
Code, documentation, and packaging were written in collaboration with an AI assistant.
All output was reviewed and directed by a human.
