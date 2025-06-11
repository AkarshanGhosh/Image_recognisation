#!/usr/bin/env python3
"""
Script to copy your trained models from src/models to webapp/backend/models
"""

import os
import shutil
from pathlib import Path

def copy_models():
    """Copy model files from src/models to webapp/backend/models"""
    
    # Get current directory (should be webapp/backend)
    current_dir = Path(__file__).parent.absolute()
    
    # Source directory (Image_recognisation/src/models)
    project_root = current_dir.parent.parent  # Go up to Image_recognisation
    src_models_dir = project_root / "src" / "models"
    
    # Destination directory (webapp/backend/models)
    dest_models_dir = current_dir / "models"
    
    print(f"🔍 Looking for models in: {src_models_dir}")
    print(f"📁 Target directory: {dest_models_dir}")
    
    # Check if source directory exists
    if not src_models_dir.exists():
        print(f"❌ Source directory not found: {src_models_dir}")
        return False
    
    # Create destination directory if it doesn't exist
    dest_models_dir.mkdir(exist_ok=True)
    print(f"✅ Created/verified destination directory: {dest_models_dir}")
    
    # Find .pth files in source directory
    pth_files = list(src_models_dir.glob("*.pth"))
    
    if not pth_files:
        print(f"❌ No .pth files found in {src_models_dir}")
        return False
    
    print(f"\n📋 Found {len(pth_files)} model files:")
    
    # Copy each .pth file
    copied_files = []
    for pth_file in pth_files:
        dest_file = dest_models_dir / pth_file.name
        try:
            shutil.copy2(pth_file, dest_file)
            print(f"✅ Copied: {pth_file.name}")
            copied_files.append(pth_file.name)
        except Exception as e:
            print(f"❌ Failed to copy {pth_file.name}: {e}")
    
    if copied_files:
        print(f"\n🎉 Successfully copied {len(copied_files)} model files!")
        print("📁 Models are now available at:")
        for filename in copied_files:
            print(f"   - {dest_models_dir / filename}")
        return True
    else:
        print("\n❌ No files were copied successfully")
        return False

def verify_models():
    """Verify that models are in the right place"""
    current_dir = Path(__file__).parent.absolute()
    models_dir = current_dir / "models"
    
    print(f"\n🔍 Verifying models in: {models_dir}")
    
    if not models_dir.exists():
        print("❌ Models directory doesn't exist")
        return False
    
    pth_files = list(models_dir.glob("*.pth"))
    
    if not pth_files:
        print("❌ No .pth files found in models directory")
        return False
    
    print(f"✅ Found {len(pth_files)} model files:")
    for pth_file in pth_files:
        size = pth_file.stat().st_size / (1024 * 1024)  # Size in MB
        print(f"   - {pth_file.name} ({size:.1f} MB)")
    
    return True

def main():
    """Main function"""
    print("🚀 Model Copy Utility")
    print("=" * 50)
    
    # First try to copy models
    success = copy_models()
    
    if success:
        # Verify the copy was successful
        verify_models()
        
        print("\n💡 Next steps:")
        print("1. Run 'python test_models.py' to test the models")
        print("2. Start your FastAPI server with 'python main.py'")
    else:
        print("\n💡 Manual steps:")
        print("1. Manually copy your .pth files from src/models/ to webapp/backend/models/")
        print("2. Or update the model paths in model_manager.py")

if __name__ == "__main__":
    main()