import React from 'react';

const AlphabetReference = () => {
  // Dicionário de sinais conforme o código fornecido
  const SINAIS = {
    'A': 'Letra A em LIBRAS',
    'O': 'Letra O em LIBRAS',
    'H': 'Letra H em LIBRAS',
    'C': 'Letra C em LIBRAS',
    'I': 'Letra I em LIBRAS',
    'T': 'Letra T em LIBRAS',
    'E': 'Letra E em LIBRAS',
    'U': 'Letra U em LIBRAS'
  };

  return (
    <div>
      <h3 className="text-sm font-medium text-muted-foreground mb-2">Alfabeto LIBRAS Reconhecido:</h3>
      <div className="alphabet-reference">
        {Object.entries(SINAIS).map(([letter, description]) => (
          <div key={letter} className="alphabet-item">
            <span className="alphabet-letter">{letter}</span>
            <span className="text-xs text-muted-foreground">{description}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AlphabetReference;

