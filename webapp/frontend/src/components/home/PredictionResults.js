// src/components/home/PredictionResults.js
import React from 'react';

const PredictionResults = ({ predictions, isLoading }) => {
  const latestPrediction = predictions.length > 0 ? predictions[0] : null;

  if (isLoading) {
    return (
      <div className="prediction-loading">
        <div className="loading-content">
          <div className="loading-spinner-large">
            <div className="spinner"></div>
          </div>
          <h3>🧠 AI Models Processing...</h3>
          <p>Analyzing your image with multiple neural networks</p>
          <div className="loading-steps">
            <div className="step active">📸 Image preprocessing</div>
            <div className="step active">🤖 Model inference</div>
            <div className="step">📊 Results compilation</div>
          </div>
        </div>
      </div>
    );
  }

  if (!latestPrediction) {
    return (
      <div className="prediction-empty">
        <div className="empty-content">
          <div className="empty-icon">🎯</div>
          <h3>Ready for Analysis</h3>
          <p>Upload an image or capture a photo to see AI predictions here</p>
          <div className="empty-features">
            <div className="empty-feature">
              <span className="feature-bullet">✨</span>
              Multiple AI models
            </div>
            <div className="empty-feature">
              <span className="feature-bullet">⚡</span>
              Real-time results
            </div>
            <div className="empty-feature">
              <span className="feature-bullet">🎯</span>
              High accuracy
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="prediction-results">
      {/* Image Preview */}
      <div className="result-image">
        <img src={latestPrediction.image} alt="Analyzed" />
        <div className="image-overlay">
          <span className="timestamp">
            {new Date(latestPrediction.timestamp).toLocaleTimeString()}
          </span>
        </div>
      </div>

      {/* Model Results */}
      <div className="model-results">
        {latestPrediction.models.map((model, modelIndex) => (
          <div key={modelIndex} className="model-result">
            <div className="model-header">
              <h4 className="model-name">🤖 {model.name}</h4>
              <div className="model-badge">
                {model.predictions.length} predictions
              </div>
            </div>
            
            <div className="predictions-list">
              {model.predictions.map((prediction, predIndex) => (
                <div key={predIndex} className="prediction-item">
                  <div className="prediction-info">
                    <span className="prediction-class">{prediction.class}</span>
                    <span className="prediction-confidence">
                      {(prediction.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="confidence-bar">
                    <div 
                      className="confidence-fill"
                      style={{ width: `${prediction.confidence * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Summary Stats */}
      <div className="result-summary">
        <div className="summary-stat">
          <span className="stat-label">Models Used</span>
          <span className="stat-value">{latestPrediction.models.length}</span>
        </div>
        <div className="summary-stat">
          <span className="stat-label">Top Confidence</span>
          <span className="stat-value">
            {Math.max(...latestPrediction.models.flatMap(m => 
              m.predictions.map(p => p.confidence * 100)
            )).toFixed(1)}%
          </span>
        </div>
        <div className="summary-stat">
          <span className="stat-label">Processing Time</span>
          <span className="stat-value">~2.1s</span>
        </div>
      </div>
    </div>
  );
};

export default PredictionResults;