# Documentação do Projeto - Reconhecimento de Sinais em LIBRAS

## Visão Geral

Este projeto é um site web responsivo que integra um sistema de reconhecimento de sinais do alfabeto da Língua Brasileira de Sinais (LIBRAS) utilizando inteligência artificial. O sistema permite que os usuários façam sinais em tempo real através da câmera e recebam feedback visual instantâneo sobre a tradução dos sinais para o alfabeto latino.

## Tecnologias Utilizadas

### Frontend
- **React 19.1.0**: Framework principal para desenvolvimento da interface
- **Tailwind CSS**: Framework de estilização responsiva
- **Lucide React**: Biblioteca de ícones
- **Shadcn/UI**: Componentes de interface pré-construídos

### Inteligência Artificial
- **ONNX Runtime Web 1.22.0**: Para execução do modelo de IA no navegador
- **TensorFlow.js 4.22.0**: Para pré-processamento de imagens

### Ferramentas de Desenvolvimento
- **Vite**: Bundler e servidor de desenvolvimento
- **ESLint**: Linting de código
- **PostCSS**: Processamento de CSS

## Estrutura do Projeto

```
libras-recognition/
├── public/
│   ├── models/
│   │   └── best.onnx (modelo de reconhecimento)
│   └── index.html
├── src/
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Header.jsx
│   │   │   ├── Footer.jsx
│   │   │   └── Instructions.jsx
│   │   ├── camera/
│   │   │   └── CameraFeed.jsx
│   │   ├── translation/
│   │   │   ├── TranslationDisplay.jsx
│   │   │   ├── AlphabetReference.jsx
│   │   │   └── Controls.jsx
│   │   └── ui/ (componentes shadcn/ui)
│   ├── services/
│   │   └── recognitionService.js
│   ├── App.jsx
│   ├── App.css
│   └── main.jsx
├── package.json
└── vite.config.js
```

## Funcionalidades Implementadas

### 1. Interface Responsiva
- Layout adaptável para desktop, tablet e mobile
- Design moderno com cores acessíveis
- Navegação intuitiva

### 2. Sistema de Câmera
- Acesso à câmera do dispositivo
- Modo de demonstração para testes sem câmera
- Controles para ativar/desativar câmera

### 3. Reconhecimento de Sinais
- Integração com modelo ONNX para detecção de sinais
- Suporte aos sinais: A, O, H, C, I, T, E, U
- Processamento em tempo real
- Sistema de buffer para suavização de detecções

### 4. Feedback Visual
- Exibição do sinal atual detectado
- Histórico de sinais detectados
- Formação de palavras a partir dos sinais
- Referência visual do alfabeto LIBRAS

### 5. Controles de Usuário
- Botão para limpar histórico
- Botão para reiniciar detecção
- Instruções de uso

## Como Usar

### Requisitos do Sistema
- Navegador moderno com suporte a:
  - MediaDevices API (para acesso à câmera)
  - WebGL (para processamento do modelo ONNX)
  - ES6+ (para JavaScript moderno)

### Instalação e Execução

1. **Instalar dependências:**
   ```bash
   pnpm install
   ```

2. **Executar em modo de desenvolvimento:**
   ```bash
   pnpm run dev
   ```

3. **Construir para produção:**
   ```bash
   pnpm run build
   ```

### Uso da Aplicação

1. **Acesso inicial**: Abra o site no navegador
2. **Ativação**: Clique em "Ativar Câmera" ou "Modo Demonstração"
3. **Posicionamento**: Posicione a mão no centro da área da câmera
4. **Sinais**: Faça os sinais do alfabeto LIBRAS
5. **Visualização**: Observe a tradução em tempo real na área lateral

## Modelo de Reconhecimento

### Sinais Suportados
O sistema reconhece 8 sinais do alfabeto LIBRAS:
- A, O, H, C, I, T, E, U

### Configurações do Modelo
- **Formato**: ONNX
- **Entrada**: Imagens 640x640 pixels
- **Confiança mínima**: 60%
- **Buffer de suavização**: 3 frames

### Modo de Demonstração
Para fins de demonstração e testes, o sistema inclui um modo de simulação que:
- Funciona sem câmera real
- Gera detecções aleatórias dos sinais suportados
- Permite testar toda a funcionalidade da interface

## Acessibilidade

### Recursos de Acessibilidade
- Contraste adequado de cores
- Textos descritivos para elementos visuais
- Layout responsivo para diferentes dispositivos
- Instruções claras de uso

### Compatibilidade
- **Navegadores**: Chrome, Firefox, Safari, Edge (versões recentes)
- **Dispositivos**: Desktop, tablet, smartphone
- **Sistemas**: Windows, macOS, Linux, iOS, Android

## Próximos Passos

### Para Implementação com Modelo Real
1. Substituir o arquivo `best.onnx` pelo modelo treinado real
2. Ajustar as configurações de pré-processamento conforme necessário
3. Calibrar os parâmetros de confiança e buffer
4. Testar com dados reais de sinais LIBRAS

### Melhorias Futuras
- Suporte a mais sinais do alfabeto LIBRAS
- Reconhecimento de palavras e frases completas
- Integração com API de tradução para formação de palavras
- Modo de treinamento para usuários
- Histórico persistente de sessões
- Exportação de resultados

## Considerações Técnicas

### Performance
- Processamento otimizado para 10 FPS
- Uso eficiente de memória com limpeza de tensores
- Buffer de detecções para reduzir falsos positivos

### Segurança
- Processamento local no navegador (sem envio de dados)
- Acesso à câmera apenas com permissão do usuário
- Não armazenamento de imagens ou vídeos

### Escalabilidade
- Arquitetura modular para fácil extensão
- Serviços separados para diferentes funcionalidades
- Configurações centralizadas

## Suporte e Manutenção

### Logs e Debugging
- Console do navegador para logs de desenvolvimento
- Tratamento de erros com mensagens amigáveis
- Fallbacks para funcionalidades não suportadas

### Atualizações
- Estrutura preparada para atualizações do modelo
- Versionamento de componentes
- Compatibilidade com futuras versões das dependências

