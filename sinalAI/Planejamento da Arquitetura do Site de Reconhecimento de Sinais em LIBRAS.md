# Planejamento da Arquitetura do Site de Reconhecimento de Sinais em LIBRAS

## 1. Visão Geral da Arquitetura

O site será desenvolvido como uma aplicação web de página única (SPA) utilizando React para garantir uma experiência de usuário fluida e responsiva. A arquitetura será composta pelos seguintes componentes principais:

### 1.1 Componentes Principais
- **Interface do Usuário (Frontend)**: Desenvolvida em React com componentes responsivos
- **Módulo de Acesso à Câmera**: Utilizando a API MediaDevices do navegador
- **Módulo de Processamento de Imagem**: Integração com o modelo ONNX para detecção de sinais
- **Módulo de Feedback Visual**: Exibição em tempo real dos sinais detectados e sua tradução

### 1.2 Tecnologias Utilizadas
- **React**: Framework para desenvolvimento da interface
- **ONNX Runtime Web**: Para execução do modelo de IA no navegador
- **TensorFlow.js**: Para pré-processamento de imagens (se necessário)
- **Tailwind CSS**: Para estilização responsiva
- **MediaDevices API**: Para acesso à câmera do dispositivo

## 2. Fluxo de Funcionamento

1. O usuário acessa o site e concede permissão para uso da câmera
2. A câmera é inicializada e começa a capturar frames em tempo real
3. Cada frame é processado pelo modelo ONNX para detecção de sinais
4. Os sinais detectados são traduzidos para o alfabeto latino
5. O resultado da tradução é exibido na interface com feedback visual
6. O histórico de sinais detectados é mantido para formar palavras/frases

## 3. Estrutura de Diretórios

```
libras-recognition/
├── public/
│   ├── index.html
│   ├── models/
│   │   └── best.onnx
│   └── assets/
│       └── (imagens e outros recursos estáticos)
├── src/
│   ├── components/
│   │   ├── Camera.jsx
│   │   ├── SignRecognition.jsx
│   │   ├── TranslationDisplay.jsx
│   │   ├── Header.jsx
│   │   └── Footer.jsx
│   ├── services/
│   │   ├── cameraService.js
│   │   └── recognitionService.js
│   ├── utils/
│   │   └── helpers.js
│   ├── App.jsx
│   └── index.js
└── package.json
```

## 4. Requisitos Técnicos

### 4.1 Requisitos do Navegador
- Suporte à API MediaDevices para acesso à câmera
- Suporte a WebGL para processamento do modelo ONNX
- Navegadores modernos (Chrome, Firefox, Safari, Edge)

### 4.2 Requisitos de Performance
- Processamento em tempo real (mínimo 15 FPS)
- Baixa latência na detecção e exibição dos resultados
- Otimização para dispositivos móveis

### 4.3 Requisitos de Acessibilidade
- Design responsivo para diferentes tamanhos de tela
- Contraste adequado para melhor visualização
- Feedback visual e textual para os sinais detectados

