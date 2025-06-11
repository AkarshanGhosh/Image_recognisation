import React from 'react';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="footer">
      <div className="footer-container">
        <div className="footer-content">
          {/* Brand Section */}
          <div className="footer-section">
            <div className="footer-brand">
              <span className="brand-icon">🧠</span>
              <span className="brand-text">AI Vision</span>
            </div>
            <p className="footer-description">
              Advanced AI-powered image recognition platform with custom model training capabilities.
            </p>
          </div>

          {/* Quick Links */}
          <div className="footer-section">
            <h4 className="footer-title">Quick Links</h4>
            <ul className="footer-links">
              <li><a href="/">Home</a></li>
              <li><a href="/model">Train Model</a></li>
              <li><a href="/about-project">About Project</a></li>
              <li><a href="/about-me">About Me</a></li>
            </ul>
          </div>

          {/* Features */}
          <div className="footer-section">
            <h4 className="footer-title">Features</h4>
            <ul className="footer-links">
              <li>Multi-Model Prediction</li>
              <li>Custom Model Training</li>
              <li>Webcam Integration</li>
              <li>Model Download</li>
            </ul>
          </div>

          {/* Social/Contact */}
          <div className="footer-section">
            <h4 className="footer-title">Connect</h4>
            <div className="social-links">
              <a href="#" className="social-link">
                <span>📧</span> Email
              </a>
              <a href="#" className="social-link">
                <span>💼</span> LinkedIn
              </a>
              <a href="#" className="social-link">
                <span>🐙</span> GitHub
              </a>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="footer-bottom">
          <div className="footer-copyright">
            <p>&copy; {currentYear} AI Vision Platform. Built with ❤️ using React & Python.</p>
          </div>
          <div className="footer-tech">
            <span className="tech-badge">React</span>
            <span className="tech-badge">Python</span>
            <span className="tech-badge">PyTorch</span>
            <span className="tech-badge">MongoDB</span>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;