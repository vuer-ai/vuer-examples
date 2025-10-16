# Vuer Examples

A collection of examples demonstrating the capabilities of [Vuer](https://github.com/vuer-ai/vuer), a powerful framework for building 3D visualization applications.

## About

This repository contains individual example projects, each in its own git repository and included here as git submodules. Each example is self-contained with its own dependencies and documentation.

## Quick Start

### Clone with all examples

```bash
git clone --recursive https://github.com/your-org/vuer-examples.git
cd vuer-examples
```

If you've already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

### Running an example

Each example is a standalone project. To run an example:

```bash
cd vuer-example-01_trimesh
pip install -r requirements.txt
python main.py
```

Then open your browser to `http://localhost:8012` (or the port specified in the example).

## Examples Index

Each example demonstrates different features of Vuer:

### Basic Examples
- **01_trimesh** - Loading and rendering mesh files (OBJ, TriMesh)
- **02_pointcloud** - Working with point cloud data
- **02_pointcloud_pcd** - Loading PCD files
- **02_pointcloud_ply** - Loading PLY files
- **03_urdf** - Loading and displaying URDF robot models
- **04_imperative_api** - Using Vuer's imperative API

### Rendering & Visualization
- **05_collecting_render** - Collecting rendered frames
- **05_pointcloud_animation** - Animating point clouds
- **06_depth_texture** - Working with depth textures
- **07_background_image** - Setting background images
- **08_experimental_depth_image** - Advanced depth rendering

### Scene Elements
- **11_coordinates_markers** - Displaying coordinate frames
- **12_camera** - Camera controls and positioning
- **13_plane_primitive** - Using primitive shapes
- **14_obj** - Loading OBJ files with materials
- **15_spline_frustum** - Camera frustums and splines
- **16_arrows** - Arrow visualizations
- **17_sky_ball** - Skybox environments

### Interaction & Controls
- **18_movable** - Interactive movable objects
- **19_hand_tracking** - VR hand tracking
- **20_motion_controllers** - VR motion controller support
- **25_body_tracking** - Full body tracking

### Advanced Examples
- **21_3D_movie** - Playing 3D video content
- **22_3d_text** - Rendering 3D text
- **23_spark** - Spark/particle effects
- **24_mujoco_interactive_simulator** - Interactive MuJoCo physics simulation

## Development Workflow

### Contributing a new example

1. Create a new repository for your example:
   ```bash
   mkdir vuer-example-my-feature
   cd vuer-example-my-feature
   git init
   ```

2. Create the example structure:
   ```
   vuer-example-my-feature/
   ├── main.py           # Main example code
   ├── README.md         # Documentation
   ├── requirements.txt  # Python dependencies
   ├── assets/           # Optional: asset files
   └── .gitignore
   ```

3. Push to GitHub:
   ```bash
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/your-org/vuer-example-my-feature.git
   git push -u origin main
   ```

4. Add as a submodule to this repository:
   ```bash
   cd /path/to/vuer-examples
   git submodule add https://github.com/your-org/vuer-example-my-feature.git
   git commit -m "Add my-feature example"
   ```

### Migrating existing examples

If you have existing examples in `vuer/docs/examples`, use the provided migration script:

```bash
# Test first with dry-run
python setup_example_repos.py --dry-run

# Create individual repos for all examples
python setup_example_repos.py

# Or just one specific example
python setup_example_repos.py --example 01_trimesh
```

After creating the individual repos and pushing them to GitHub:

```bash
# Edit add_submodules.sh to set your GitHub org
vim add_submodules.sh  # Set GITHUB_ORG

# Add all as submodules
./add_submodules.sh

# Or add specific examples
./add_submodules.sh 01_trimesh 02_pointcloud
```

## Example Template

Here's a minimal example structure:

```python
# main.py
from vuer import Vuer
from vuer.schemas import DefaultScene, Sphere

app = Vuer()

@app.spawn(start=True)
async def main(session):
    session.upsert @ DefaultScene(
        Sphere(
            key="my-sphere",
            args=[0.5, 32, 32],
            position=[0, 1, 0],
            material={"color": "red"}
        )
    )

    # Keep the session alive
    while True:
        await session.sleep(0.1)
```

```txt
# requirements.txt
vuer>=0.1.0
```

## Resources

- [Vuer Documentation](https://vuer.ai/docs)
- [Vuer GitHub](https://github.com/vuer-ai/vuer)
- [Vuer Discord Community](https://discord.gg/vuer)

## License

Each example maintains its own license. Please refer to individual example repositories for licensing information.

## Support

For questions or issues:
- Open an issue in the specific example repository
- Open an issue in the main [Vuer repository](https://github.com/vuer-ai/vuer/issues)
- Join the [Vuer Discord community](https://discord.gg/vuer)
