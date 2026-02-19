# idTech Geometry Toolkit

A PyQt5 desktop application for generating brush geometry for idTech engines (Quake, Doom 3).

![Screenshot placeholder](screenshot.png)

## Features

- **34 Geometric Modules** - Rooms, halls, arches, staircases, pillars, and more
- **BSP-Based Dungeon Generation** - Random layouts with guaranteed connectivity
- **Layout Editor** - Manual 2D grid placement with drag-and-drop modules
- **Real-Time 3D Preview** - FPS-style camera with textured rendering
- **Multi-Floor Support** - Vertical dungeons with stair connections
- **Generation Templates** - Arena, Maze, Fortress, Cathedral presets
- **5 Texture Themes** - Base, Medieval, Tech, Gothic, Runic
- **Secret Areas** - Walk-through walls and hidden chambers
- **Polygonal Rooms** - 3 to 16-sided room shapes

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
git clone https://github.com/yourusername/idtech-geometry-toolkit.git
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

1. **Module Mode** - Preview individual geometric pieces
   - Select a module from the palette (Sanctuary, StraightHall, Arch, etc.)
   - Adjust parameters in the right panel
   - View in real-time 3D preview

2. **Layout Mode** - Build complete levels
   - Click modules in the palette to place on the 2D grid
   - Drag to reposition, right-click for options
   - Use "Auto-Connect" to link adjacent rooms
   - Click "Generate" to create 3D geometry

3. **Random Dungeon** - Auto-generate layouts
   - Set room count, connectivity, and theme
   - Choose a template (Arena, Maze, Fortress, Cathedral)
   - Click "Random Dungeon" for instant levels

4. **Export** - Save your map
   - File > Export (Ctrl+E)
   - Choose format: idTech 1, idTech 4, or OBJ

## 3D Preview Controls

| Control | Action |
|---------|--------|
| **W/A/S/D** | Move forward/left/back/right |
| **Mouse Drag** | Look around |
| **Scroll Wheel** | Adjust movement speed |
| **R** | Reset camera position |
| **F** | Toggle wireframe |
| **T** | Toggle textures |

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

## Module Library

### Rooms (19)
Sanctuary, Tomb, Tower, Chamber, Storage, GreatHall, Prison, Armory, Cistern, Stronghold, Courtyard, Arena, Laboratory, Vault, Barracks, Shrine, Pit, Antechamber, SecretChamber

### Halls (6)
StraightHall, TJunction, Crossroads, SquareCorner, VerticalStairHall, SecretHall

### Structural (5)
StraightStaircase, Arch, Pillar, Buttress, Battlement

### Connective (4)
Bridge, Platform, Rampart, Gallery

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+N | New Layout |
| Ctrl+O | Open Layout |
| Ctrl+S | Save Layout |
| Ctrl+E | Export Map |
| Ctrl+G | Generate Geometry |
| Ctrl+R | Random Dungeon |
| Ctrl+3 | Toggle 3D Preview |
| F1 | Help |
| Delete | Remove Selected |

## License

MIT License - Copyright 2026 Zachary Quarles

See [LICENSE](LICENSE) for details.

## Acknowledgments

- idTech engine formats by id Software
- [ericw-tools](https://github.com/ericwa/ericw-tools) for Quake map compilation
- [TrenchBroom](https://trenchbroom.github.io/) for map editing inspiration
