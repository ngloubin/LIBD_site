import React from 'react';
import { Heart, Rocket } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="footer">
      <div className="footer-content">
        <div className="team-credit">
          <Rocket size={20} style={{ display: 'inline', marginRight: '8px' }} />
          Desenvolvido pela equipe FOGUETÃO
        </div>
        <p>
          Promovendo acessibilidade e inclusão digital para pessoas surdas e ouvintes.
        </p>
        <p style={{ marginTop: '1rem', fontSize: '0.9rem' }}>
          © 2025 Reconhecimento de Sinais em LIBRAS • Feito com{' '}
          <Heart size={14} style={{ display: 'inline', color: '#ff4757' }} /> pela equipe FOGUETÃO
        </p>
      </div>
    </footer>
  );
};

export default Footer;

