#!/usr/bin/env python3
"""
Script to convert vuer examples into individual git repositories
and add them as submodules to the vuer-examples repo.

Usage:
    python setup_example_repos.py [--dry-run] [--example EXAMPLE_NAME]
"""

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple


class ExampleRepo:
    """Manages creation of individual example repositories"""

    def __init__(self, source_dir: Path, target_base_dir: Path, dry_run: bool = False):
        self.source_dir = source_dir
        self.target_base_dir = target_base_dir
        self.dry_run = dry_run

    def find_examples(self) -> List[Tuple[str, Path, Optional[Path]]]:
        """
        Find all example files in the source directory.
        Returns list of (example_name, py_file, md_file) tuples.
        """
        examples = []
        py_files = sorted(self.source_dir.glob("*.py"))

        for py_file in py_files:
            # Skip utility files
            if py_file.name.startswith("_"):
                continue

            # Get corresponding .md file if it exists
            md_file = py_file.with_suffix(".md")
            if not md_file.exists():
                md_file = None

            # Extract example name (e.g., "01_trimesh" from "01_trimesh.py")
            example_name = py_file.stem

            examples.append((example_name, py_file, md_file))

        return examples

    def find_example_assets(self, py_file: Path) -> List[Path]:
        """
        Parse the Python file to find referenced assets.
        Returns list of asset paths relative to the examples directory.
        """
        assets = []
        content = py_file.read_text()

        # Look for common asset patterns
        # Pattern 1: assets_folder / "filename"
        asset_pattern = re.compile(r'assets_folder\s*/\s*["\']([^"\']+)["\']')
        for match in asset_pattern.finditer(content):
            asset_file = match.group(1)
            asset_path = self.source_dir / "assets" / asset_file
            if asset_path.exists():
                assets.append(asset_path)

        # Pattern 2: Direct references to assets/ or figures/
        direct_pattern = re.compile(r'["\'](?:assets|figures)/([^"\']+)["\']')
        for match in direct_pattern.finditer(content):
            asset_file = match.group(1)
            for subdir in ["assets", "figures"]:
                asset_path = self.source_dir / subdir / asset_file
                if asset_path.exists() and asset_path not in assets:
                    assets.append(asset_path)

        return assets

    def create_example_repo(self, example_name: str, py_file: Path,
                           md_file: Optional[Path]) -> Path:
        """
        Create a new repository for a single example.
        Returns the path to the created repository.
        """
        # Create repo directory
        repo_dir = self.target_base_dir / f"vuer-example-{example_name}"

        if self.dry_run:
            print(f"[DRY RUN] Would create repo at: {repo_dir}")
            return repo_dir

        # Create directory
        repo_dir.mkdir(parents=True, exist_ok=True)

        # Copy main.py (rename from example file)
        main_py = repo_dir / "main.py"
        shutil.copy2(py_file, main_py)

        # Modify main.py to remove doc wrapper if present
        self._clean_python_file(main_py)

        # Copy README.md (from .md file)
        if md_file:
            readme = repo_dir / "README.md"
            # Copy and clean the README
            self._create_clean_readme(md_file, readme, example_name)
        else:
            # Create a basic README
            readme = repo_dir / "README.md"
            readme.write_text(f"# {example_name}\n\nVuer example: {example_name}\n")

        # Copy assets
        assets = self.find_example_assets(py_file)
        if assets:
            assets_dir = repo_dir / "assets"
            assets_dir.mkdir(exist_ok=True)
            for asset in assets:
                shutil.copy2(asset, assets_dir / asset.name)

        # Create requirements.txt
        self._create_requirements(repo_dir, py_file)

        # Create .gitignore
        self._create_gitignore(repo_dir)

        # Initialize git repo
        self._init_git_repo(repo_dir, example_name)

        print(f"✓ Created example repo: {repo_dir}")
        return repo_dir

    def _create_clean_readme(self, md_file: Path, readme: Path, example_name: str):
        """Create a clean README without the Python code block"""
        content = md_file.read_text()
        lines = content.split('\n')
        cleaned_lines = []
        in_code_block = False

        for line in lines:
            # Detect start of Python code block
            if line.strip() == '```python':
                in_code_block = True
                continue

            # Detect end of code block
            if in_code_block and line.strip() == '```':
                in_code_block = False
                continue

            # Skip lines inside code block
            if in_code_block:
                continue

            cleaned_lines.append(line)

        # Add usage instructions at the end
        cleaned_lines.extend([
            '',
            '## Usage',
            '',
            '```bash',
            'pip install -r requirements.txt',
            'python main.py',
            '```',
            '',
            f'Then open your browser to `http://localhost:8012`',
        ])

        readme.write_text('\n'.join(cleaned_lines))

    def _clean_python_file(self, py_file: Path):
        """Remove cmx doc wrapper from Python file"""
        content = py_file.read_text()

        # Remove MAKE_DOCS checks and doc decorators
        lines = content.split('\n')
        cleaned_lines = []
        in_doc_block = False
        in_with_block = False

        for line in lines:
            # Skip imports related to documentation
            if 'from cmx import doc' in line or 'MAKE_DOCS' in line:
                continue

            # Skip MAKE_DOCS variable
            if line.strip().startswith('MAKE_DOCS ='):
                continue

            # Skip doc decorator lines
            if line.strip().startswith('doc @'):
                in_doc_block = True
                continue

            # Skip doc string blocks
            if in_doc_block:
                if '"""' in line:
                    in_doc_block = False
                continue

            # Skip context manager for docs
            if 'with doc' in line:
                in_with_block = True
                continue

            # Skip 'from contextlib import nullcontext'
            if 'from contextlib import nullcontext' in line:
                continue

            cleaned_lines.append(line)

        # Remove empty lines at the start
        while cleaned_lines and cleaned_lines[0].strip() == '':
            cleaned_lines.pop(0)

        # Dedent if needed (remove one level of indentation if most lines are indented)
        indented_count = sum(1 for line in cleaned_lines if line and line.startswith('    '))
        total_lines = sum(1 for line in cleaned_lines if line.strip())

        if total_lines > 0 and indented_count / total_lines > 0.5:
            cleaned_lines = [line[4:] if line.startswith('    ') else line
                           for line in cleaned_lines]

        py_file.write_text('\n'.join(cleaned_lines))

    def _create_requirements(self, repo_dir: Path, py_file: Path):
        """Create requirements.txt for the example"""
        content = py_file.read_text()

        # Basic requirements
        requirements = ['vuer']

        # Detect common imports
        if 'trimesh' in content:
            requirements.append('trimesh')
        if 'numpy' in content or 'np.' in content:
            requirements.append('numpy')
        if 'mujoco' in content:
            requirements.append('mujoco')
        if 'PIL' in content or 'from PIL' in content:
            requirements.append('Pillow')

        req_file = repo_dir / "requirements.txt"
        req_file.write_text('\n'.join(requirements) + '\n')

    def _create_gitignore(self, repo_dir: Path):
        """Create .gitignore file"""
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
"""
        gitignore = repo_dir / ".gitignore"
        gitignore.write_text(gitignore_content)

    def _init_git_repo(self, repo_dir: Path, example_name: str):
        """Initialize git repository and make initial commit"""
        cwd = os.getcwd()
        try:
            os.chdir(repo_dir)

            # Initialize repo
            subprocess.run(['git', 'init'], check=True, capture_output=True)

            # Add all files
            subprocess.run(['git', 'add', '.'], check=True, capture_output=True)

            # Initial commit
            commit_msg = f"Initial commit for {example_name} example"
            subprocess.run(['git', 'commit', '-m', commit_msg],
                         check=True, capture_output=True)

            print(f"  ✓ Initialized git repo and committed files")

        finally:
            os.chdir(cwd)

    def add_as_submodule(self, repo_dir: Path, submodules_parent: Path):
        """Add the example repo as a git submodule"""
        if self.dry_run:
            print(f"[DRY RUN] Would add {repo_dir} as submodule")
            return

        # For now, just print instructions
        # In a real scenario, you'd push to a remote first
        rel_path = repo_dir.relative_to(submodules_parent)
        print(f"\n  To add as submodule after pushing to GitHub:")
        print(f"  git submodule add <remote-url> {rel_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert vuer examples into individual repositories'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually doing it'
    )
    parser.add_argument(
        '--example',
        type=str,
        help='Process only a specific example (e.g., "01_trimesh")'
    )
    parser.add_argument(
        '--source',
        type=Path,
        default=Path(__file__).parent.parent / 'vuer' / 'docs' / 'examples',
        help='Source directory containing examples (default: ../vuer/docs/examples)'
    )
    parser.add_argument(
        '--target',
        type=Path,
        default=Path(__file__).parent,
        help='Target directory for example repos (default: current directory)'
    )

    args = parser.parse_args()

    # Validate source directory
    if not args.source.exists():
        print(f"Error: Source directory not found: {args.source}")
        return 1

    # Create manager
    manager = ExampleRepo(args.source, args.target, args.dry_run)

    # Find examples
    examples = manager.find_examples()

    # Filter if specific example requested
    if args.example:
        examples = [(name, py, md) for name, py, md in examples
                   if name == args.example]
        if not examples:
            print(f"Error: Example '{args.example}' not found")
            return 1

    print(f"Found {len(examples)} example(s)")
    print()

    # Process each example
    created_repos = []
    for example_name, py_file, md_file in examples:
        print(f"Processing: {example_name}")
        repo_dir = manager.create_example_repo(example_name, py_file, md_file)
        created_repos.append(repo_dir)
        print()

    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Created {len(created_repos)} example repositories")
    print("\nNext steps:")
    print("1. Review each example repository")
    print("2. Create GitHub repositories for each example")
    print("3. Push each example to its remote repository")
    print("4. Add each as a git submodule:")
    print("   git submodule add <remote-url> <local-path>")

    return 0


if __name__ == '__main__':
    exit(main())