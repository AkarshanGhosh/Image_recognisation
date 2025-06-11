import React from 'react';

const AboutMe = () => {
  return (
    <div className="about-me-page">
      <div className="container">
        {/* Page Header */}
        <section className="page-header fade-in">
          <h1 className="page-title">👨‍💻 About Me</h1>
          <p className="page-subtitle">
            Meet the developer behind this AI Vision Platform
          </p>
        </section>

        {/* Profile Section */}
        <section className="content-section">
          <div className="profile-card">
            <div className="profile-header">
              <div className="profile-avatar">
                <span className="avatar-emoji">👨‍💻</span>
              </div>
              <div className="profile-info">
                <h2 className="profile-name">AI Developer</h2>
                <p className="profile-title">Full-Stack AI Engineer</p>
                <div className="profile-links">
                  <a href="#" className="profile-link">
                    <span>📧</span> Email
                  </a>
                  <a href="#" className="profile-link">
                    <span>💼</span> LinkedIn
                  </a>
                  <a href="#" className="profile-link">
                    <span>🐙</span> GitHub
                  </a>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* About Section */}
        <section className="content-section">
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">🚀 About Me</h2>
            </div>
            <div className="content-body">
              <p>
                I'm a passionate full-stack developer specializing in AI and machine learning 
                applications. With expertise in both frontend and backend technologies, 
                I enjoy creating intelligent solutions that bridge the gap between 
                cutting-edge AI research and practical real-world applications.
              </p>
              
              <h3>💡 What I Do</h3>
              <ul className="skills-list">
                <li>🤖 AI/ML Model Development and Deployment</li>
                <li>🌐 Full-Stack Web Application Development</li>
                <li>📊 Data Science and Analytics</li>
                <li>☁️ Cloud Computing and DevOps</li>
                <li>📱 Mobile-First Responsive Design</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Skills Section */}
        <section className="content-section">
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">🛠️ Technical Skills</h2>
            </div>
            <div className="skills-grid">
              <div className="skill-category">
                <h3>Programming Languages</h3>
                <div className="skill-items">
                  <span className="skill-badge advanced">Python</span>
                  <span className="skill-badge advanced">JavaScript</span>
                  <span className="skill-badge intermediate">TypeScript</span>
                  <span className="skill-badge intermediate">SQL</span>
                </div>
              </div>
              <div className="skill-category">
                <h3>AI/ML Frameworks</h3>
                <div className="skill-items">
                  <span className="skill-badge advanced">PyTorch</span>
                  <span className="skill-badge intermediate">TensorFlow</span>
                  <span className="skill-badge intermediate">Scikit-learn</span>
                  <span className="skill-badge intermediate">OpenCV</span>
                </div>
              </div>
              <div className="skill-category">
                <h3>Web Technologies</h3>
                <div className="skill-items">
                  <span className="skill-badge advanced">React</span>
                  <span className="skill-badge advanced">FastAPI</span>
                  <span className="skill-badge intermediate">Node.js</span>
                  <span className="skill-badge intermediate">Express</span>
                </div>
              </div>
              <div className="skill-category">
                <h3>Tools & Platforms</h3>
                <div className="skill-items">
                  <span className="skill-badge advanced">Git</span>
                  <span className="skill-badge intermediate">Docker</span>
                  <span className="skill-badge intermediate">AWS</span>
                  <span className="skill-badge intermediate">MongoDB</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Contact Section */}
        <section className="content-section">
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">📬 Get In Touch</h2>
            </div>
            <div className="content-body">
              <p>
                I'm always interested in discussing new projects, creative ideas, 
                or opportunities to be part of your vision. Feel free to reach out!
              </p>
              
              <div className="contact-grid">
                <div className="contact-item">
                  <span className="contact-icon">📧</span>
                  <div>
                    <h4>Email</h4>
                    <p>your.email@example.com</p>
                  </div>
                </div>
                <div className="contact-item">
                  <span className="contact-icon">💼</span>
                  <div>
                    <h4>LinkedIn</h4>
                    <p>linkedin.com/in/yourprofile</p>
                  </div>
                </div>
                <div className="contact-item">
                  <span className="contact-icon">🐙</span>
                  <div>
                    <h4>GitHub</h4>
                    <p>github.com/yourusername</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default AboutMe;