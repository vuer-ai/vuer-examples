# COLMAP Visualization with Vuer

Visualize a COLMAP sparse reconstruction: 3D point cloud + color-coded camera frustums.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py --data-dir /path/to/colmap/project
```

Then open `http://localhost:8012` in your browser.

## Expected Directory Structure

```
project/
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```
