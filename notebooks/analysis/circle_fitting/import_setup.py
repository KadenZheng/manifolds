"""
Robust import setup for notebooks in the manifolds project.
This file can be imported at the beginning of any notebook to set up the path correctly.
"""

import sys
import os
from pathlib import Path

def setup_project_imports():
    """Set up the Python path to import from the project's src directory."""
    
    # Get the current file's directory
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent
    
    # Navigate up to find the project root (contains 'src' directory)
    project_root = current_dir
    while project_root.parent != project_root:  # Not at filesystem root
        if (project_root / 'src').exists() and (project_root / 'src' / '__init__.py').exists():
            break
        project_root = project_root.parent
    else:
        # Fallback: assume we're 3 levels deep in notebooks/analysis/circle_fitting/
        project_root = current_dir.parent.parent.parent
    
    # Convert to string
    project_root_str = str(project_root)
    
    # Clean up sys.modules
    modules_to_remove = [mod for mod in sys.modules.keys() if mod.startswith('src')]
    for mod in modules_to_remove:
        del sys.modules[mod]
    
    # Add project root to path if not already there
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    print(f"Project root set to: {project_root_str}")
    print(f"Python path: {sys.path[:3]}...")
    
    # Verify imports work
    try:
        import src
        print("✓ src module found and importable")
        return project_root_str
    except ImportError as e:
        print(f"✗ Failed to import src module: {e}")
        raise

# Run setup when imported
PROJECT_ROOT = setup_project_imports()