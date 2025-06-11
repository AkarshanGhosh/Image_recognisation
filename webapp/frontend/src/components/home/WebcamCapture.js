// src/components/home/WebcamCapture.js
import React, { useState, useRef, useCallback } from 'react';

const WebcamCapture = ({ onPrediction, setIsLoading, isLoading }) => {
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [stream, setStream] = useState(null);
  const [error, setError] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const startWebcam = useCallback(async () => {
    try {
      setError(null);
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      });
      
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        videoRef.current.play();
      }
      setIsWebcamActive(true);
    } catch (err) {
      console.error('Error accessing webcam:', err);
      setError('Unable to access webcam. Please check permissions.');
    }
  }, []);

  const stopWebcam = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    setIsWebcamActive(false);
    setCapturedImage(null);
  }, [stream]);

  const captureImage = useCallback(() => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      const ctx = canvas.getContext('2d');
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      ctx.drawImage(video, 0, 0);
      const imageDataUrl = canvas.toDataURL('image/jpeg', 0.8);
      setCapturedImage(imageDataUrl);
    }
  }, []);

  const handleAnalyze = async () => {
    if (!capturedImage) return;

    setIsLoading(true);
    
    try {
      // Simulate API call with mock predictions
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const mockPrediction = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        image: capturedImage,
        models: [
          {
            name: 'ResNet-50',
            predictions: [
              { class: 'Person', confidence: 0.94 },
              { class: 'Human Face', confidence: 0.87 },
              { class: 'Portrait', confidence: 0.82 }
            ]
          },
          {
            name: 'MobileNet',
            predictions: [
              { class: 'Person', confidence: 0.91 },
              { class: 'Human', confidence: 0.89 },
              { class: 'Face', confidence: 0.86 }
            ]
          }
        ]
      };

      onPrediction(mockPrediction);
      stopWebcam();
    } catch (error) {
      console.error('Prediction error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const retakePhoto = () => {
    setCapturedImage(null);
  };

  return (
    <div className="webcam-container">
      {error && (
        <div className="alert alert-error">
          <span>⚠️</span>
          {error}
        </div>
      )}

      {!isWebcamActive && !capturedImage && (
        <div className="webcam-placeholder">
          <div className="placeholder-content">
            <div className="placeholder-icon">📷</div>
            <h3>Use Your Camera</h3>
            <p>Capture images directly from your webcam for instant analysis</p>
            <button
              className="btn btn-primary btn-large"
              onClick={startWebcam}
              disabled={isLoading}
            >
              📹 Start Camera
            </button>
          </div>
        </div>
      )}

      {isWebcamActive && !capturedImage && (
        <div className="webcam-active">
          <div className="video-container">
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className="webcam-video"
            />
          </div>
          <div className="webcam-controls">
            <button
              className="btn btn-secondary"
              onClick={stopWebcam}
              disabled={isLoading}
            >
              ❌ Stop Camera
            </button>
            <button
              className="btn btn-primary"
              onClick={captureImage}
              disabled={isLoading}
            >
              📸 Capture Photo
            </button>
          </div>
        </div>
      )}

      {capturedImage && (
        <div className="captured-image-container">
          <div className="captured-image">
            <img src={capturedImage} alt="Captured" />
          </div>
          <div className="capture-actions">
            <button
              className="btn btn-outline"
              onClick={retakePhoto}
              disabled={isLoading}
            >
              🔄 Retake
            </button>
            <button
              className="btn btn-primary"
              onClick={handleAnalyze}
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <span className="loading-spinner"></span>
                  Analyzing...
                </>
              ) : (
                '🔍 Analyze Photo'
              )}
            </button>
          </div>
        </div>
      )}

      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  );
};

export default WebcamCapture;