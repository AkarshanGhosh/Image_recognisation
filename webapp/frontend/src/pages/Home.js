
import React, { useState } from 'react';
import ImageUpload from '../components/home/ImageUpload';
import WebcamCapture from '../components/home/WebcamCapture';
import PredictionResults from '../components/home/PredictionResults';
import PredictionHistory from '../components/home/PredictionHistory';

const Home = () => {
  const [activeTab, setActiveTab] = useState('upload');
  const [predictions, setPredictions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const handlePrediction = (newPrediction) => {
    setPredictions(prev => [newPrediction, ...prev]);
  };

  return (
    <div className="home-page">
      <div className="container">
        {/* Hero Section */}
        <section className="hero-section fade-in">
          <div className="hero-content">
            <h1 className="hero-title">
              🧠 AI Vision Platform
            </h1>
            <p className="hero-subtitle">
              Advanced multi-model image recognition powered by deep learning. 
              Upload images or use your webcam to get instant predictions from multiple AI models.
            </p>
            <div className="hero-stats">
              <div className="stat-item">
                <span className="stat-number">2+</span>
                <span className="stat-label">AI Models</span>
              </div>
              <div className="stat-item">
                <span className="stat-number">11+</span>
                <span className="stat-label">Categories</span>
              </div>
              <div className="stat-item">
                <span className="stat-number">95%</span>
                <span className="stat-label">Accuracy</span>
              </div>
            </div>
          </div>
        </section>

        {/* Main Content */}
        <section className="main-section">
          <div className="content-grid">
            {/* Left Column - Image Input */}
            <div className="input-column">
              <div className="card">
                <div className="card-header">
                  <h2 className="card-title">📸 Image Recognition</h2>
                  <p className="card-subtitle">
                    Choose your preferred input method
                  </p>
                </div>

                {/* Tab Navigation */}
                <div className="tab-navigation">
                  <button
                    className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
                    onClick={() => setActiveTab('upload')}
                  >
                    📁 Upload Image
                  </button>
                  <button
                    className={`tab-btn ${activeTab === 'webcam' ? 'active' : ''}`}
                    onClick={() => setActiveTab('webcam')}
                  >
                    📷 Use Webcam
                  </button>
                </div>

                {/* Tab Content */}
                <div className="tab-content">
                  {activeTab === 'upload' && (
                    <ImageUpload
                      onPrediction={handlePrediction}
                      setIsLoading={setIsLoading}
                      isLoading={isLoading}
                    />
                  )}
                  {activeTab === 'webcam' && (
                    <WebcamCapture
                      onPrediction={handlePrediction}
                      setIsLoading={setIsLoading}
                      isLoading={isLoading}
                    />
                  )}
                </div>
              </div>
            </div>

            {/* Right Column - Results */}
            <div className="results-column">
              <div className="card">
                <div className="card-header">
                  <h2 className="card-title">🎯 Prediction Results</h2>
                  <p className="card-subtitle">
                    Real-time analysis from multiple AI models
                  </p>
                </div>
                
                <PredictionResults
                  predictions={predictions}
                  isLoading={isLoading}
                />
              </div>
            </div>
          </div>
        </section>

        {/* History Section */}
        <section className="history-section">
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">📊 Prediction History</h2>
              <p className="card-subtitle">
                View your recent predictions and results
              </p>
            </div>
            
            <PredictionHistory predictions={predictions} />
          </div>
        </section>

        {/* Features Section */}
        <section className="features-section">
          <h2 className="section-title">✨ Platform Features</h2>
          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon">🤖</div>
              <h3 className="feature-title">Multi-Model Recognition</h3>
              <p className="feature-description">
                Simultaneous predictions from multiple specialized AI models
              </p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">📱</div>
              <h3 className="feature-title">Webcam Integration</h3>
              <p className="feature-description">
                Real-time image capture and analysis using your device camera
              </p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">🎯</div>
              <h3 className="feature-title">High Accuracy</h3>
              <p className="feature-description">
                State-of-the-art deep learning models with 95%+ accuracy
              </p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">📊</div>
              <h3 className="feature-title">Detailed Analytics</h3>
              <p className="feature-description">
                Comprehensive prediction results with confidence scores
              </p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default Home;