import React from 'react'

function Footer() {
  return (
    <footer className="footer">
      <div className="container">
        <p>
          Built with ❤️ using React, FastAPI, and Transformers
        </p>
        <p className="tech-stack">
          <span className="badge">React</span>
          <span className="badge">FastAPI</span>
          <span className="badge">Transformers</span>
          <span className="badge">GPT-2</span>
        </p>
      </div>
    </footer>
  )
}

export default Footer
