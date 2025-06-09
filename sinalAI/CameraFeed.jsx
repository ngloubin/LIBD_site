import React, { useRef, useEffect, useState } from 'react';
import { Camera, Play, Square, Smartphone, Monitor } from 'lucide-react';
import recognitionService from '../../services/recognitionService';

const CameraFeed = ({ 
  isActive, 
  onSignDetected, 
  isDemoMode, 
  onToggleDemo,
  isMobileView,
  onToggleView 
}) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [modelStatus, setModelStatus] = useState('Carregando modelo...');
  const [error, setError] = useState(null);

  // Inicializar o modelo de reconhecimento
  useEffect(() => {
    const initializeModel = async () => {
      try {
        setModelStatus('Carregando modelo de IA...');
        await recognitionService.initialize();
        setModelStatus('Sistema pronto para uso');
      } catch (error) {
        console.error('Erro ao inicializar modelo:', error);
        setModelStatus('Modo demonstração ativo');
        setError('Modelo não disponível - usando simulação');
      }
    };

    initializeModel();
  }, []);

  // Gerenciar stream da câmera
  useEffect(() => {
    if (isActive && !isDemoMode) {
      startCamera();
    } else {
      stopCamera();
    }

    return () => stopCamera();
  }, [isActive, isDemoMode]);

  // Processar frames quando ativo
  useEffect(() => {
    let intervalId;
    
    if (isActive) {
      if (isDemoMode) {
        // Modo demonstração - simular detecções
        intervalId = setInterval(() => {
          simulateDetection();
        }, 2000);
      } else if (stream && recognitionService.isReady()) {
        // Modo real - processar frames da câmera
        intervalId = setInterval(() => {
          processFrame();
        }, 100); // 10 FPS
      }
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isActive, isDemoMode, stream]);

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: 640, 
          height: 480,
          facingMode: 'user'
        }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        setStream(mediaStream);
        setError(null);
      }
    } catch (error) {
      console.error('Erro ao acessar câmera:', error);
      setError('Não foi possível acessar a câmera. Usando modo demonstração.');
      onToggleDemo();
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const processFrame = async () => {
    if (!videoRef.current || !canvasRef.current || isProcessing) return;

    try {
      setIsProcessing(true);
      
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      
      // Configurar canvas
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      // Desenhar frame atual
      ctx.drawImage(video, 0, 0);
      
      // Obter dados da imagem
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      
      // Processar com o modelo
      const result = await recognitionService.detectSign(imageData);
      
      if (result && onSignDetected) {
        onSignDetected(result.sign, result.confidence);
      }
    } catch (error) {
      console.error('Erro ao processar frame:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const simulateDetection = () => {
    const signs = ['A', 'O', 'H', 'C', 'I', 'T', 'E', 'U'];
    const randomSign = signs[Math.floor(Math.random() * signs.length)];
    const confidence = 0.7 + Math.random() * 0.3; // 70-100%
    
    if (onSignDetected) {
      onSignDetected(randomSign, confidence);
    }
  };

  return (
    <div className="card fade-in-up">
      <div className="camera-container">
        {!isDemoMode && isActive ? (
          <>
            <video
              ref={videoRef}
              className="camera-feed"
              autoPlay
              playsInline
              muted
            />
            <canvas
              ref={canvasRef}
              style={{ display: 'none' }}
            />
          </>
        ) : (
          <div className="camera-placeholder">
            <div style={{ textAlign: 'center' }}>
              <Camera size={48} style={{ marginBottom: '1rem', opacity: 0.5 }} />
              <p>{isDemoMode ? 'Modo de Demonstração' : 'Câmera Inativa'}</p>
              <p style={{ fontSize: '0.9rem', opacity: 0.7 }}>
                {isDemoMode ? 'Simulando detecção de sinais' : modelStatus}
              </p>
              {error && (
                <p style={{ color: 'var(--accent-orange)', fontSize: '0.8rem', marginTop: '0.5rem' }}>
                  {error}
                </p>
              )}
            </div>
          </div>
        )}
      </div>

      <div className="controls">
        <button
          onClick={() => onToggleDemo()}
          className={`btn ${isActive && !isDemoMode ? 'btn-primary' : 'btn-secondary'}`}
        >
          <Camera size={16} />
          {isActive && !isDemoMode ? 'Câmera Ativa' : 'Ativar Câmera'}
        </button>

        <button
          onClick={() => onToggleDemo(true)}
          className={`btn ${isDemoMode ? 'btn-accent' : 'btn-secondary'}`}
        >
          <Play size={16} />
          {isDemoMode ? 'Demo Ativo' : 'Modo Demonstração'}
        </button>

        <button
          onClick={onToggleView}
          className="btn btn-secondary"
          title={isMobileView ? 'Alternar para versão Desktop' : 'Alternar para versão Mobile'}
        >
          {isMobileView ? <Monitor size={16} /> : <Smartphone size={16} />}
          {isMobileView ? 'Desktop' : 'Mobile'}
        </button>
      </div>

      {isProcessing && (
        <div style={{ 
          textAlign: 'center', 
          marginTop: '1rem', 
          color: 'var(--spacex-light-blue)',
          fontSize: '0.9rem'
        }}>
          <div className="loading" style={{ 
            width: '20px', 
            height: '20px', 
            border: '2px solid var(--spacex-light-blue)',
            borderTop: '2px solid transparent',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite',
            display: 'inline-block',
            marginRight: '8px'
          }} />
          Processando...
        </div>
      )}
    </div>
  );
};

export default CameraFeed;

