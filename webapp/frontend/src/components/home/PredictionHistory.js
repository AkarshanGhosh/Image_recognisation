// src/components/home/PredictionHistory.js
import React, { useState } from 'react';

const PredictionHistory = ({ predictions }) => {
  const [selectedPrediction, setSelectedPrediction] = useState(null);
  const [showModal, setShowModal] = useState(false);

  const handleViewDetails = (prediction) => {
    setSelectedPrediction(prediction);
    setShowModal(true);
  };

  const closeModal = () => {
    setShowModal(false);
    setSelectedPrediction(null);
  };

  if (predictions.length === 0) {
    return (
      <div className="history-empty">
        <div className="empty-content">
          <div className="empty-icon">📊</div>
          <h3>No Predictions Yet</h3>
          <p>Your prediction history will appear here as you analyze images</p>
        </div>
      </div>
    );
  }

  return (
    <div className="prediction-history">
      <div className="history-list">
        {predictions.map((prediction, index) => (
          <div key={prediction.id} className="history-item">
            <div className="history-image">
              <img src={prediction.image} alt={`Prediction ${index + 1}`} />
            </div>
            
            <div className="history-content">
              <div className="history-header">
                <span className="history-timestamp">
                  {new Date(prediction.timestamp).toLocaleString()}
                </span>
                <div className="history-models">
                  {prediction.models.length} models used
                </div>
              </div>
              
              <div className="history-predictions">
                {prediction.models.slice(0, 2).map((model, modelIndex) => (
                  <div key={modelIndex} className="history-model">
                    <span className="model-name">{model.name}:</span>
                    <span className="top-prediction">
                      {model.predictions[0]?.class} 
                      ({(model.predictions[0]?.confidence * 100).toFixed(1)}%)
                    </span>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="history-actions">
              <button
                className="btn btn-outline btn-small"
                onClick={() => handleViewDetails(prediction)}
              >
                👁️ View Details
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Modal for detailed view */}
      {showModal && selectedPrediction && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3 className="modal-title">Prediction Details</h3>
              <button className="modal-close" onClick={closeModal}>
                ✕
              </button>
            </div>
            
            <div className="modal-body">
              <div className="modal-image">
                <img src={selectedPrediction.image} alt="Prediction detail" />
              </div>
              
              <div className="modal-timestamp">
                📅 {new Date(selectedPrediction.timestamp).toLocaleString()}
              </div>
              
              <div className="modal-results">
                {selectedPrediction.models.map((model, modelIndex) => (
                  <div key={modelIndex} className="modal-model">
                    <h4 className="modal-model-name">🤖 {model.name}</h4>
                    <div className="modal-predictions">
                      {model.predictions.map((pred, predIndex) => (
                        <div key={predIndex} className="modal-prediction">
                          <div className="prediction-row">
                            <span className="pred-class">{pred.class}</span>
                            <span className="pred-confidence">
                              {(pred.confidence * 100).toFixed(2)}%
                            </span>
                          </div>
                          <div className="pred-bar">
                            <div 
                              className="pred-fill"
                              style={{ width: `${pred.confidence * 100}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictionHistory;