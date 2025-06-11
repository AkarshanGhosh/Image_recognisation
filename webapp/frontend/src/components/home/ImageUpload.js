// src/components/home/ImageUpload.js
import React, { useState, useRef } from 'react';

const ImageUpload = ({ onPrediction, setIsLoading, isLoading }) => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileSelect = (file) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleFileInputChange = (event) => {
    const file = event.target.files[0];
    handleFileSelect(file);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleSubmit = async () => {
    if (!selectedImage) return;

    setIsLoading(true);
    
    try {
      // Simulate API call with mock predictions
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const mockPrediction = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        image: previewUrl,
        models: [
          {
            name: 'ResNet-50',
            predictions: [
              { class: 'Golden Retriever', confidence: 0.89 },
              { class: 'Labrador', confidence: 0.76 },
              { class: 'Dog', confidence: 0.95 }
            ]
          },
          {
            name: 'MobileNet',
            predictions: [
              { class: 'Dog', confidence: 0.92 },
              { class: 'Animal', confidence: 0.88 },
              { class: 'Pet', confidence: 0.85 }
            ]
          }
        ]
      };

      onPrediction(mockPrediction);
    } catch (error) {
      console.error('Prediction error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const clearImage = () => {
    setSelectedImage(null);
    setPreviewUrl(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="image-upload-container">
      {!selectedImage ? (
        <div
          className={`upload-dropzone ${dragActive ? 'drag-active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <div className="upload-content">
            <div className="upload-icon">📁</div>
            <h3>Drop your image here</h3>
            <p>or click to browse files</p>
            <div className="upload-formats">
              Supports: JPG, PNG, GIF, WebP
            </div>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileInputChange}
            style={{ display: 'none' }}
          />
        </div>
      ) : (
        <div className="image-preview-container">
          <div className="image-preview">
            <img src={previewUrl} alt="Selected" />
          </div>
          <div className="image-actions">
            <button
              className="btn btn-outline btn-small"
              onClick={clearImage}
              disabled={isLoading}
            >
              🗑️ Remove
            </button>
            <button
              className="btn btn-primary"
              onClick={handleSubmit}
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <span className="loading-spinner"></span>
                  Analyzing...
                </>
              ) : (
                '🔍 Analyze Image'
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageUpload;