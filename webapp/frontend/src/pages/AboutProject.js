import React from 'react';

const AboutProject = () => {
  return (
    <div className="about-project-page">
      <div className="container">
        {/* Page Header */}
        <section className="page-header fade-in">
          <h1 className="page-title">📋 About This Project</h1>
          <p className="page-subtitle">
            Learn about the technology, architecture, and features of our AI Vision Platform
          </p>
        </section>

        {/* Project Overview */}
        <section className="content-section">
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">🎯 Project Overview</h2>
            </div>
            <div className="content-body">
              <p>
                AI Vision Platform is a modern, full-stack web application that demonstrates 
                the power of deep learning in image recognition. Built with cutting-edge 
                technologies, it provides both pre-trained models and custom model training capabilities.
              </p>
              
              <h3>🚀 Key Features</h3>
              <ul className="feature-list">
                <li><strong>Multi-Model Prediction:</strong> Simultaneous analysis using multiple AI models</li>
                <li><strong>Custom Model Training:</strong> Train personalized models with your own data</li>
                <li><strong>Webcam Integration:</strong> Real-time image capture and analysis</li>
                <li><strong>Model Download:</strong> Export trained models as standalone web applications</li>
                <li><strong>Responsive Design:</strong> Works seamlessly on desktop and mobile devices</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Technology Stack */}
        <section className="content-section">
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">⚡ Technology Stack</h2>
            </div>
            <div className="tech-grid">
              <div className="tech-category">
                <h3>Frontend</h3>
                <div className="tech-items">
                  <span className="tech-badge">React</span>
                  <span className="tech-badge">JavaScript</span>
                  <span className="tech-badge">CSS3</span>
                  <span className="tech-badge">HTML5</span>
                </div>
              </div>
              <div className="tech-category">
                <h3>Backend</h3>
                <div className="tech-items">
                  <span className="tech-badge">Python</span>
                  <span className="tech-badge">FastAPI</span>
                  <span className="tech-badge">Uvicorn</span>
                  <span className="tech-badge">Pydantic</span>
                </div>
              </div>
              <div className="tech-category">
                <h3>AI/ML</h3>
                <div className="tech-items">
                  <span className="tech-badge">PyTorch</span>
                  <span className="tech-badge">CNN</span>
                  <span className="tech-badge">Computer Vision</span>
                  <span className="tech-badge">PIL</span>
                </div>
              </div>
              <div className="tech-category">
                <h3>Database</h3>
                <div className="tech-items">
                  <span className="tech-badge">MongoDB</span>
                  <span className="tech-badge">PyMongo</span>
                  <span className="tech-badge">GridFS</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Architecture */}
        <section className="content-section">
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">🏗️ System Architecture</h2>
            </div>
            <div className="content-body">
              <div className="architecture-diagram">
                <div className="arch-layer">
                  <h4>Frontend Layer</h4>
                  <p>React-based user interface with responsive design</p>
                </div>
                <div className="arch-arrow">↓</div>
                <div className="arch-layer">
                  <h4>API Layer</h4>
                  <p>FastAPI backend with RESTful endpoints</p>
                </div>
                <div className="arch-arrow">↓</div>
                <div className="arch-layer">
                  <h4>AI Processing Layer</h4>
                  <p>PyTorch models for image classification</p>
                </div>
                <div className="arch-arrow">↓</div>
                <div className="arch-layer">
                  <h4>Data Layer</h4>
                  <p>MongoDB for metadata and model storage</p>
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default AboutProject;