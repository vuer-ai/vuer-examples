#!/bin/bash
# Script to add example repositories as git submodules
# This assumes you've already created and pushed the individual example repos to GitHub

set -e

# Configuration
GITHUB_ORG="your-github-org"  # Change this to your GitHub organization/username
BASE_REPO_NAME="vuer-example"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to add a submodule
add_submodule() {
    local example_name=$1
    local repo_name="${BASE_REPO_NAME}-${example_name}"
    local repo_url="https://github.com/${GITHUB_ORG}/${repo_name}.git"
    local local_path="${repo_name}"

    # Check if directory already exists
    if [ -d "$local_path" ]; then
        print_warning "Directory $local_path already exists, skipping..."
        return
    fi

    # Check if it's already a submodule
    if git config --file .gitmodules --get-regexp path | grep -q "$local_path"; then
        print_warning "Submodule $local_path already registered, skipping..."
        return
    fi

    print_info "Adding submodule: $repo_name"

    # Try to add the submodule
    if git submodule add "$repo_url" "$local_path"; then
        print_info "✓ Added $repo_name"
    else
        print_error "Failed to add $repo_name"
        print_error "Make sure the repository exists at: $repo_url"
    fi
}

# Main script
main() {
    # Make sure we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_error "Not in a git repository!"
        exit 1
    fi

    print_info "GitHub Organization: $GITHUB_ORG"
    print_info "Base Repository Name: $BASE_REPO_NAME"
    echo ""

    # If specific examples are provided as arguments, process only those
    if [ $# -gt 0 ]; then
        for example in "$@"; do
            add_submodule "$example"
        done
    else
        # Otherwise, process all example directories that match the pattern
        print_info "Searching for example directories..."
        for dir in ${BASE_REPO_NAME}-*/; do
            if [ -d "$dir" ]; then
                example_name=$(basename "$dir" | sed "s/^${BASE_REPO_NAME}-//")
                add_submodule "$example_name"
            fi
        done
    fi

    echo ""
    print_info "Done! Don't forget to commit the changes:"
    echo "  git add .gitmodules"
    echo "  git commit -m 'Add example repositories as submodules'"
}

# Show help if requested
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0 [EXAMPLE_NAME...]"
    echo ""
    echo "Add vuer example repositories as git submodules."
    echo ""
    echo "If no EXAMPLE_NAME is provided, all matching directories will be processed."
    echo ""
    echo "Before running this script:"
    echo "1. Set GITHUB_ORG to your GitHub organization/username"
    echo "2. Make sure all example repositories are pushed to GitHub"
    echo ""
    echo "Examples:"
    echo "  $0                    # Add all examples as submodules"
    echo "  $0 01_trimesh         # Add only the trimesh example"
    echo "  $0 01_trimesh 02_pointcloud  # Add specific examples"
    exit 0
fi

main "$@"
