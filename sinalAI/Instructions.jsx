import React from 'react';
import { Zap, Camera, Hand, Eye } from 'lucide-react';

const Instructions = () => {
  return (
    <div className="instructions fade-in-up">
      <h3 className="instructions-title">
        <Zap size={20} />
        Como utilizar:
      </h3>
      <ol className="instructions-list">
        <li>
          <Camera size={16} style={{ display: 'inline', marginRight: '8px' }} />
          Permita o acesso à sua câmera quando solicitado
        </li>
        <li>
          <Hand size={16} style={{ display: 'inline', marginRight: '8px' }} />
          Posicione sua mão no centro da tela, dentro da área demarcada
        </li>
        <li>
          <Hand size={16} style={{ display: 'inline', marginRight: '8px' }} />
          Faça os sinais do alfabeto LIBRAS de forma clara
        </li>
        <li>
          <Eye size={16} style={{ display: 'inline', marginRight: '8px' }} />
          Observe a tradução em tempo real na área à direita
        </li>
      </ol>
    </div>
  );
};

export default Instructions;

