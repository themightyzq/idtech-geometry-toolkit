# idTech Geometry Toolkit

A PyQt5 desktop application for generating brush geometry for idTech engines (Quake, Doom 3).

## Features

- **43 Geometric Modules** — Rooms, halls, multi-floor rooms, arches, staircases, pillars, and more
- **BSP-Based Dungeon Generation** — Random layouts with guaranteed connectivity
- **Layout Editor** — 2D grid placement with drag-and-drop, portal auto-connect, and snap-to-portal
- **Quad View** — 4-pane layout (Top/Front/Side/3D) with orthographic Z-height visualization
- **Flow Views** — Front (XZ) and Side (YZ) schematics showing module positions and portal Z-levels
- **Real-Time 3D Preview** — FPS-style camera with textured rendering and 5 render modes
- **Multi-Floor Support** — Vertical dungeons with stair connections and 8 multi-floor room types
- **Generation Templates** — Arena, Maze, Fortress, Cathedral presets
- **5 Texture Themes** — Base, Medieval, Tech, Gothic, Runic
- **Secret Areas** — Walk-through CLIP walls and hidden chambers
- **Polygonal Rooms** — 3 to 16-sided room shapes

## Export Formats

| Format | Description | Use With |
|--------|-------------|----------|
| **idTech 1** | 3-point plane format | Quake, Half-Life, Source (via ericw-tools) |
| **idTech 4** | brushDef3 normal+distance | Doom 3, Quake 4, Prey (via dmap) |
| **OBJ/MTL** | Wavefront mesh | Blender, 3D modeling software |

## Installation

### Requirements
- Python 3.9+
- PyQt5
- PyOpenGL
- NumPy
- Pillow

### Setup

```bash
# Clone the repository
git clone https://github.com/themightyzq/idtech-geometry-toolkit.git
cd idtech-geometry-toolkit

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Launch the Application

**macOS:**
```bash
./Launch_LevelGenerator.command
```

**All Platforms:**
```bash
source venv/bin/activate
python3 main.py
```

### Basic Workflow

1. **Module Mode** — Preview individual geometric pieces
   - Select a module from the palette (Sanctuary, StraightHall, Arch, etc.)
   - Adjust parameters in the right panel
   - View in real-time 3D preview or Quad View (Shift+Q)

2. **Layout Mode** — Build complete levels
   - Click modules in the palette to place on the 2D grid
   - Drag to reposition, press R to rotate
   - Portals auto-connect when adjacent modules face each other
   - Right-click connections to toggle secret passages (CLIP walls)
   - Use Quad View (Shift+Q) to see Front/Side Z-height flow views
   - Click "Build Geometry" to generate 3D geometry

3. **Random Dungeon** — Auto-generate layouts
   - Set room count, floor count, and complexity
   - Choose a template (Arena, Maze, Fortress, Cathedral)
   - Click "Random Dungeon" for instant levels

4. **Export** — Save your map
   - File > Export (Ctrl+E)
   - Choose format: idTech 1, idTech 4, or OBJ

## Keyboard Shortcuts

### Global

| Shortcut | Action |
|----------|--------|
| **Ctrl+1** | Switch to Layout mode |
| **Ctrl+2** | Switch to Module mode |
| **Ctrl+G** | Build geometry |
| **Ctrl+R** | Random dungeon (Layout mode) |
| **Ctrl+E** | Export .map file |
| **Ctrl+O** | Open output folder |
| **Shift+Q** | Toggle Quad View |
| **Escape** | Cancel generation / restore quad view |
| **F1** | Help |

### Layout Editor

| Shortcut | Action |
|----------|--------|
| **Click** | Place or select module |
| **Shift+Click** | Place and continue placing |
| **Drag** | Move selected module |
| **R** | Rotate 90° (placement or selected) |
| **Delete** | Delete selected |
| **F** | Fit view to content |
| **G** | Focus on selected module |
| **Ctrl+Z / Ctrl+Shift+Z** | Undo / Redo |
| **Ctrl+D** | Duplicate selected |
| **Ctrl+F** | Toggle flow visualization |
| **Ctrl+3** | Toggle 3D preview |
| **Ctrl+N** | New layout |
| **Ctrl+S** | Save layout |

### 3D Preview

| Shortcut | Action |
|----------|--------|
| **W/A/S/D** | Fly forward/left/back/right |
| **Q/E** | Fly down/up |
| **Shift** | Sprint (2x speed) |
| **Right-drag** | Mouselook |
| **Middle-drag** | Pan |
| **Scroll** | Zoom |
| **RMB+Scroll** | Adjust camera speed |
| **F** | Fit geometry in view |
| **Home** | Reset view and speed |
| **H** | Toggle control hints |
| **1-6** | Preset camera angles |

### Quad View

| Shortcut | Action |
|----------|--------|
| **Alt+7** | Maximize Top (XY) pane |
| **Alt+1** | Maximize Front (XZ) pane |
| **Alt+3** | Maximize Side (YZ) pane |
| **Alt+0** | Maximize 3D pane |
| **Escape** | Restore quad view |
| **F** | Fit view to content |
| **G** | Focus on selected module |

### Flow Views (Front/Side)

| Shortcut | Action |
|----------|--------|
| **Click** | Select module |
| **Drag** | Reposition (horizontal = cell, vertical = Z-offset) |
| **Middle/Right-drag** | Pan view |
| **Scroll** | Zoom |
| **F** | Fit view to layout |
| **G** | Focus on selected module |
| **Escape** | Cancel drag / clear selection |

## Module Library

### Rooms (19)
Sanctuary, Tomb, Tower, Chamber, Storage, GreatHall, Prison, Armory, Cistern, Stronghold, Courtyard, Arena, Laboratory, Vault, Barracks, Shrine, Pit, Antechamber, SecretChamber

### Multi-Floor Rooms (8)
Amphitheater, CatwalkChamber, BalconyRoom, SunkenChamber, LibraryArchive, Grotto, RadialShrine, Forge

### Halls (6)
StraightHall, TJunction, Crossroads, SquareCorner, VerticalStairHall, SecretHall

### Structural (6)
StraightStaircase, SpiralStaircase, Arch, Pillar, Buttress, Battlement

### Connective (4)
Bridge, Platform, Rampart, Gallery

## Map Compilation

Generated `.map` files require compilation before use in-game.

### idTech 1 (Quake)

Download [ericw-tools](https://github.com/ericwa/ericw-tools/releases):

```bash
qbsp my_map.map
vis my_map.bsp
light my_map.bsp
```

### idTech 4 (Doom 3)

Load the `.map` file in DarkRadiant or the Doom 3 editor, then run:
```
dmap my_map
```

## License

MIT License - Copyright 2026 Zachary Quarles

See [LICENSE](LICENSE) for details.

## Acknowledgments

- idTech engine formats by id Software
- [ericw-tools](https://github.com/ericwa/ericw-tools) for Quake map compilation
- [TrenchBroom](https://trenchbroom.github.io/) for map editing inspiration
