import React from 'react';

const TranslationDisplay = ({ currentSign, signHistory }) => {
  return (
    <div className="translation-container">
      <h2 className="text-xl font-bold mb-4">Tradução em Tempo Real</h2>
      
      {/* Sinal atual */}
      <div className="current-sign">
        {currentSign ? (
          <span className="highlight">{currentSign}</span>
        ) : (
          <span className="text-muted-foreground">Aguardando...</span>
        )}
      </div>
      
      {/* Histórico de sinais */}
      <div>
        <h3 className="text-sm font-medium text-muted-foreground mb-2">Histórico de Sinais:</h3>
        <div className="sign-history">
          {signHistory && signHistory.length > 0 ? (
            signHistory.map((sign, index) => (
              <span 
                key={index} 
                className={`sign-item ${index === signHistory.length - 1 ? 'active' : ''}`}
              >
                {sign}
              </span>
            ))
          ) : (
            <span className="text-sm text-muted-foreground">Nenhum sinal detectado ainda</span>
          )}
        </div>
      </div>
      
      {/* Palavra formada */}
      {signHistory && signHistory.length > 0 && (
        <div className="mt-6">
          <h3 className="text-sm font-medium text-muted-foreground mb-2">Palavra Formada:</h3>
          <div className="p-3 bg-secondary/10 text-secondary rounded-lg text-center font-bold">
            {signHistory.join('')}
          </div>
        </div>
      )}
    </div>
  );
};

export default TranslationDisplay;

