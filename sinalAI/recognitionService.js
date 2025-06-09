import * as ort from 'onnxruntime-web';
import * as tf from '@tensorflow/tfjs';

// Dicionário de sinais conforme o código fornecido
const SINAIS = {
  0: "A",
  1: "O",
  2: "H",
  3: "C",
  4: "I",
  5: "T",
  6: "E",
  7: "U"
};

// Configurações
const MODEL_PATH = '/models/best.onnx';
const CONF_THRESHOLD = 0.6;

class RecognitionService {
  constructor() {
    this.model = null;
    this.isModelLoaded = false;
    this.isLoading = false;
    this.lastDetectedSign = null;
    this.detectionBuffer = [];
    this.bufferSize = 3; // Buffer para suavização
  }

  // Inicializar o serviço e carregar o modelo
  async initialize() {
    if (this.isLoading || this.isModelLoaded) return;
    
    try {
      this.isLoading = true;
      
      // Configurar opções do ONNX Runtime
      const options = {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      };
      
      // Carregar o modelo
      this.model = await ort.InferenceSession.create(MODEL_PATH, options);
      
      this.isModelLoaded = true;
      console.log('Modelo ONNX carregado com sucesso');
    } catch (error) {
      console.error('Erro ao carregar o modelo ONNX:', error);
      throw error;
    } finally {
      this.isLoading = false;
    }
  }

  // Pré-processar a imagem para o formato esperado pelo modelo
  async preprocessImage(imageData) {
    try {
      // Converter ImageData para tensor
      const tensor = tf.browser.fromPixels(imageData);
      
      // Redimensionar para o tamanho esperado pelo modelo (640x640 é comum para YOLO)
      const resized = tf.image.resizeBilinear(tensor, [640, 640]);
      
      // Normalizar os valores de pixel para [0, 1]
      const normalized = resized.div(255.0);
      
      // Expandir dimensões para incluir o batch (1, 640, 640, 3)
      const batched = normalized.expandDims(0);
      
      // Converter para o formato esperado pelo ONNX (NCHW: batch, channels, height, width)
      const transposed = tf.transpose(batched, [0, 3, 1, 2]);
      
      // Obter os dados como Float32Array
      const data = await transposed.data();
      
      // Limpar tensores para liberar memória
      tensor.dispose();
      resized.dispose();
      normalized.dispose();
      batched.dispose();
      transposed.dispose();
      
      return new Float32Array(data);
    } catch (error) {
      console.error('Erro ao pré-processar a imagem:', error);
      throw error;
    }
  }

  // Executar a detecção em uma imagem
  async detectSign(imageData) {
    if (!this.isModelLoaded) {
      throw new Error('Modelo não carregado. Chame initialize() primeiro.');
    }
    
    try {
      // Pré-processar a imagem
      const inputData = await this.preprocessImage(imageData);
      
      // Criar tensor de entrada para o modelo
      const inputTensor = new ort.Tensor('float32', inputData, [1, 3, 640, 640]);
      const feeds = { images: inputTensor };
      
      // Executar inferência
      const results = await this.model.run(feeds);
      
      // Processar resultados (formato específico do YOLO)
      const detections = this.processDetections(results);
      
      // Aplicar suavização com buffer
      if (detections.length > 0) {
        // Adicionar ao buffer
        this.detectionBuffer.push(detections);
        if (this.detectionBuffer.length > this.bufferSize) {
          this.detectionBuffer.shift(); // Remover o mais antigo
        }
        
        // Contar ocorrências de cada classe no buffer
        const allDetections = this.detectionBuffer.flat();
        const classCounts = {};
        
        allDetections.forEach(det => {
          const classId = det.class;
          classCounts[classId] = (classCounts[classId] || 0) + 1;
        });
        
        // Encontrar a classe mais frequente
        let maxCount = 0;
        let mostFrequentClass = null;
        
        Object.entries(classCounts).forEach(([classId, count]) => {
          if (count > maxCount) {
            maxCount = count;
            mostFrequentClass = parseInt(classId);
          }
        });
        
        // Verificar se a classe mais frequente é diferente da última detectada
        if (mostFrequentClass !== null && SINAIS[mostFrequentClass] !== this.lastDetectedSign) {
          this.lastDetectedSign = SINAIS[mostFrequentClass];
          return {
            sign: this.lastDetectedSign,
            confidence: detections[0].confidence // Usar a confiança da detecção atual
          };
        }
      }
      
      // Nenhuma nova detecção
      return null;
      
    } catch (error) {
      console.error('Erro ao detectar sinal:', error);
      throw error;
    }
  }

  // Processar as detecções do modelo YOLO
  processDetections(results) {
    // Obter o tensor de saída (formato específico do YOLO)
    const outputTensor = results.output0 || results.outputs || Object.values(results)[0];
    const output = outputTensor.data;
    
    // Formato típico do YOLO: [batch, num_detections, 85]
    // onde 85 = 4 (bbox) + 1 (confidence) + 80 (classes)
    // Neste caso, temos 8 classes para os sinais LIBRAS
    
    const detections = [];
    const numDetections = outputTensor.dims[1];
    const numValues = outputTensor.dims[2];
    
    for (let i = 0; i < numDetections; i++) {
      const baseIdx = i * numValues;
      
      // Extrair confiança
      const confidence = output[baseIdx + 4];
      
      // Filtrar por confiança
      if (confidence < CONF_THRESHOLD) continue;
      
      // Encontrar a classe com maior probabilidade
      let maxClassProb = 0;
      let classId = -1;
      
      for (let c = 0; c < 8; c++) {
        const classProb = output[baseIdx + 5 + c];
        if (classProb > maxClassProb) {
          maxClassProb = classProb;
          classId = c;
        }
      }
      
      // Verificar se temos uma classe válida
      if (classId >= 0 && SINAIS[classId]) {
        // Extrair coordenadas da bounding box
        const x = output[baseIdx];
        const y = output[baseIdx + 1];
        const w = output[baseIdx + 2];
        const h = output[baseIdx + 3];
        
        detections.push({
          class: classId,
          label: SINAIS[classId],
          confidence: confidence,
          bbox: { x, y, width: w, height: h }
        });
      }
    }
    
    // Ordenar por confiança (maior primeiro)
    return detections.sort((a, b) => b.confidence - a.confidence);
  }

  // Verificar se o modelo está carregado
  isReady() {
    return this.isModelLoaded;
  }

  // Limpar recursos
  dispose() {
    this.model = null;
    this.isModelLoaded = false;
    this.detectionBuffer = [];
    this.lastDetectedSign = null;
  }
}

// Exportar uma instância única do serviço
const recognitionService = new RecognitionService();
export default recognitionService;

