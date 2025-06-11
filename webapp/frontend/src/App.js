// src/App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/common/Layout';
import Home from './pages/Home';
import Model from './pages/Model';
import AboutProject from './pages/AboutProject';
import AboutMe from './pages/AboutMe';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Layout>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/model" element={<Model />} />
            <Route path="/about-project" element={<AboutProject />} />
            <Route path="/about-me" element={<AboutMe />} />
          </Routes>
        </Layout>
      </div>
    </Router>
  );
}

export default App;