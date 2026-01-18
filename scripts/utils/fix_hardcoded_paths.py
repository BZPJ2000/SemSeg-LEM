"""
Script to fix hardcoded paths in the project.
Replaces absolute paths with relative paths using PathConfig.
"""

import os
import re
from pathlib import Path


def fix_hardcoded_paths(file_path):
    """Fix hardcoded paths in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Pattern 1: E:/A_workbench/A-lab/Unet/Unet_complit/...
        pattern1 = r'[rR]?["\']E:/A_workbench/A-lab/Unet/Unet_complit/([^"\']+)["\']'

        # Pattern 2: E:\A_workbench\A-lab\Unet\Unet_complit\...
        pattern2 = r'[rR]?["\']E:\\A_workbench\\A-lab\\Unet\\Unet_complit\\([^"\']+)["\']'

        # Replace with relative paths
        def replace_func(match):
            relative_path = match.group(1)
            # Convert to forward slashes
            relative_path = relative_path.replace('\\', '/')
            return f'paths.project_root / "{relative_path}"'

        content = re.sub(pattern1, replace_func, content)
        content = re.sub(pattern2, replace_func, content)

        # Add import if needed and content changed
        if content != original_content:
            if 'from config.paths import paths' not in content:
                # Find the last import statement
                import_pattern = r'^(import |from )'
                lines = content.split('\n')
                last_import_idx = -1

                for i, line in enumerate(lines):
                    if re.match(import_pattern, line.strip()):
                        last_import_idx = i

                if last_import_idx >= 0:
                    lines.insert(last_import_idx + 1, 'from config.paths import paths')
                    content = '\n'.join(lines)

            # Write back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return True

        return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Main function."""
    project_root = Path(__file__).parent

    # Files to fix (from grep results)
    files_to_fix = [
        'Image_computing_processing.py',
        'Salvage_all_V3.py',
