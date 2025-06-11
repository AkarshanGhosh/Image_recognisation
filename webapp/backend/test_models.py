#!/usr/bin/env python3
"""
Test script to verify your models are working correctly
"""

import os
import sys
from PIL import Image
import io

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_manager():
    """Test the model manager functionality"""
    print("🧪 Testing Model Manager...")
    print("=" * 50)
    
    try:
        from model_manager import model_manager
        print("✅ Successfully imported model_manager")
    except Exception as e:
        print(f"❌ Failed to import model_manager: {e}")
        return False
    
    # Test available models
    available_models = model_manager.get_available_models()
    print(f"📋 Available models: {available_models}")
    
    if not available_models:
        print("❌ No models available for testing")
        return False
    
    # Test model info
    for model_name in available_models:
        info = model_manager.get_model_info(model_name)
        print(f"\n🔍 Model: {model_name}")
        print(f"   Classes: {info['classes']}")
        print(f"   Device: {info['device']}")
        print(f"   Path: {info['path']}")
    
    # Test prediction with a dummy image
    print("\n🖼️  Testing prediction with dummy image...")
    try:
        # Create a simple test image (64x64 RGB)
        test_image = Image.new('RGB', (64, 64), color='red')
        
        # Convert to bytes for testing
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Run prediction
        results = model_manager.predict_all(img_bytes)
        
        print("🎯 Prediction Results:")
        for model_name, result in results.items():
            if "error" in result:
                print(f"   {model_name}: ❌ {result['error']}")
            else:
                print(f"   {model_name}: {result['prediction']} ({result['confidence']:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def find_model_files():
    """Helper function to locate model files"""
    print("\n🔍 Searching for model files...")
    print("=" * 50)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    search_locations = [
        current_dir,
        os.path.join(current_dir, 'models'),
        project_root,
        os.path.join(project_root, 'models'),
        os.path.join(project_root, 'src', 'models'),
        # Additional search paths for your structure
        os.path.join(os.path.dirname(project_root), 'src', 'models'),  # Go up to Image_recognisation/src/models
        os.path.join(os.path.dirname(project_root), 'models'),         # Go up to Image_recognisation/models
    ]
    
    found_files = []
    
    for location in search_locations:
        if os.path.exists(location):
            print(f"📁 Checking: {location}")
            files = [f for f in os.listdir(location) if f.endswith('.pth')]
            if files:
                for file in files:
                    full_path = os.path.join(location, file)
                    found_files.append(full_path)
                    print(f"   ✅ Found: {file}")
            else:
                print(f"   ⚪ No .pth files found")
    
    if not found_files:
        print("\n❌ No .pth model files found!")
        print("Please ensure your trained models are saved as .pth files in one of these locations:")
        for loc in search_locations:
            print(f"   - {loc}")
    else:
        print(f"\n✅ Total model files found: {len(found_files)}")
    
    return found_files

def main():
    """Main test function"""
    print("🚀 Model Testing Suite")
    print("=" * 50)
    
    # First, find model files
    model_files = find_model_files()
    
    if not model_files:
        print("\n💡 To fix this:")
        print("1. Make sure your trained models are saved as .pth files")
        print("2. Place them in the webapp/backend/models/ directory")
        print("3. Or update the MODEL_CONFIGS in model_manager.py with correct paths")
        return
    
    # Test the model manager
    success = test_model_manager()
    
    if success:
        print("\n🎉 All tests passed! Your models are ready for the web app.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()