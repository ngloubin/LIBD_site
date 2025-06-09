import { useState, useEffect } from 'react';
import './App.css';

// Componentes de layout
import Header from './components/layout/Header';
import Footer from './components/layout/Footer';
import Instructions from './components/layout/Instructions';

// Componentes da câmera
import CameraFeed from './components/camera/CameraFeed';

// Componentes de tradução
import TranslationDisplay from './components/translation/TranslationDisplay';
import AlphabetReference from './components/translation/AlphabetReference';
import Controls from './components/translation/Controls';

// Serviço de reconhecimento
import recognitionService from './services/recognitionService';

function App() {
  // Estados para gerenciar os sinais detectados
  const [currentSign, setCurrentSign] = useState(null);
  const [signHistory, setSignHistory] = useState([]);
  const [isModelLoaded, setIsModelLoaded] = useState(false);

  // Verificar se o modelo está carregado
  useEffect(() => {
    const checkModelStatus = () => {
      setIsModelLoaded(recognitionService.isReady());
    };
    
    // Verificar inicialmente
    checkModelStatus();
    
    // Verificar periodicamente
    const interval = setInterval(checkModelStatus, 1000);
    
    return () => clearInterval(interval);
  }, []);

  // Função para processar detecções da câmera
  const handleDetection = (sign) => {
    if (sign && sign !== currentSign) {
      setCurrentSign(sign);
      setSignHistory(prev => [...prev, sign]);
    }
  };

  // Limpar histórico de sinais
  const handleClearHistory = () => {
    setSignHistory([]);
    setCurrentSign(null);
  };

  // Reiniciar detecção
  const handleResetDetection = () => {
    setCurrentSign(null);
  };

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      
      <main className="flex-grow container py-8">
        <h1 className="text-3xl font-bold text-center mb-8">
          Reconhecimento de Sinais em LIBRAS
        </h1>
        
        <Instructions />
        
        {!isModelLoaded && (
          <div className="text-center p-4 mb-6 bg-muted rounded-lg">
            <p className="text-muted-foreground">Carregando modelo de reconhecimento...</p>
            <div className="w-full bg-muted-foreground/20 h-2 mt-2 rounded-full overflow-hidden">
              <div className="bg-primary h-full rounded-full pulse" style={{ width: '60%' }}></div>
            </div>
          </div>
        )}
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <CameraFeed onDetection={handleDetection} />
          </div>
          
          <div className="space-y-6">
            <TranslationDisplay 
              currentSign={currentSign} 
              signHistory={signHistory} 
            />
            
            <Controls 
              onClearHistory={handleClearHistory}
              onResetDetection={handleResetDetection}
            />
            
            <AlphabetReference />
          </div>
        </div>
      </main>
      
      <Footer />
    </div>
  );
}

export default App;

