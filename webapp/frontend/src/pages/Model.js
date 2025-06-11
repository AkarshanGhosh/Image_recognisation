import React, { useState } from 'react';
import TrainModel from '../components/model/TrainModel';
import MyModels from '../components/model/MyModels';

const Model = () => {
  const [activeSection, setActiveSection] = useState('train');

  return (
    <div className="model-page">
      <div className="container">
        {/* Page Header */}
        <section className="page-header fade-in">
          <h1 className="page-title">🤖 Model Management</h1>
          <p className="page-subtitle">
            Train custom AI models with your own data or download existing trained models
          </p>
        </section>

        {/* Navigation Tabs */}
        <div className="section-tabs">
          <button
            className={`section-tab ${activeSection === 'train' ? 'active' : ''}`}
            onClick={() => setActiveSection('train')}
          >
            🎓 Train New Model
          </button>
          <button
            className={`section-tab ${activeSection === 'mymodels' ? 'active' : ''}`}
            onClick={() => setActiveSection('mymodels')}
          >
            📦 My Models
          </button>
        </div>

        {/* Tab Content */}
        <div className="tab-content-area">
          {activeSection === 'train' && <TrainModel />}
          {activeSection === 'mymodels' && <MyModels />}
        </div>
      </div>
    </div>
  );
};

export default Model;