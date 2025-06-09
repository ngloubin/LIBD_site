import React from 'react';
import { Rocket, Info } from 'lucide-react';

const Header = () => {
  return (
    <header className="header">
      <div className="header-content">
        <a href="#" className="logo">
          🚀 LIBRAS.AI
        </a>
        <nav className="nav-links">
          <a href="#about" className="nav-link">
            <Info size={16} />
            Sobre
          </a>
          <a href="#help" className="nav-link">
            <Rocket size={16} />
            Ajuda
          </a>
        </nav>
      </div>
    </header>
  );
};

export default Header;

