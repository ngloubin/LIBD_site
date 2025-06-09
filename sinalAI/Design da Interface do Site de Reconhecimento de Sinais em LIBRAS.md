# Design da Interface do Site de Reconhecimento de Sinais em LIBRAS

## 1. Princípios de Design

O design da interface seguirá os seguintes princípios:

- **Acessibilidade**: Interface intuitiva e acessível para todos os usuários
- **Simplicidade**: Layout limpo e direto, focado na funcionalidade principal
- **Feedback Visual**: Indicações claras sobre o estado do sistema e ações do usuário
- **Responsividade**: Adaptação a diferentes tamanhos de tela e dispositivos

## 2. Paleta de Cores

A paleta de cores será baseada em tons que proporcionam bom contraste e legibilidade:

- **Cor Primária**: #3B82F6 (Azul)
- **Cor Secundária**: #10B981 (Verde)
- **Cor de Destaque**: #F59E0B (Âmbar)
- **Cor de Fundo**: #F9FAFB (Cinza claro)
- **Cor de Texto**: #1F2937 (Cinza escuro)
- **Cor de Alerta**: #EF4444 (Vermelho)

## 3. Tipografia

- **Fonte Principal**: Inter (sans-serif)
- **Tamanhos**:
  - Títulos: 24px - 32px
  - Subtítulos: 18px - 24px
  - Corpo: 16px
  - Pequeno: 14px

## 4. Layout das Telas

### 4.1 Tela Principal

A tela principal será dividida em seções claramente definidas:

```
+-----------------------------------------------+
|                   CABEÇALHO                   |
+-----------------------------------------------+
|                                               |
|                INSTRUÇÕES                     |
|                                               |
+-----------------------------------------------+
|                                |              |
|                                |              |
|                                |              |
|        VISUALIZAÇÃO            |  TRADUÇÃO    |
|          CÂMERA               |   E SINAIS   |
|                                |  DETECTADOS  |
|                                |              |
|                                |              |
+-----------------------------------------------+
|                                               |
|              CONTROLES E OPÇÕES               |
|                                               |
+-----------------------------------------------+
|                  RODAPÉ                       |
+-----------------------------------------------+
```

### 4.2 Componentes da Interface

#### Cabeçalho
- Logo do projeto
- Título: "Reconhecimento de Sinais em LIBRAS"
- Menu de navegação (se necessário)

#### Seção de Instruções
- Breve explicação sobre como utilizar o sistema
- Indicação para posicionar as mãos corretamente
- Status da câmera e do sistema de reconhecimento

#### Visualização da Câmera
- Feed da câmera em tempo real
- Overlay com guias de posicionamento
- Indicadores visuais de detecção

#### Tradução e Sinais Detectados
- Exibição do sinal atual detectado em destaque
- Histórico dos últimos sinais detectados
- Palavras formadas (quando aplicável)
- Alfabeto em LIBRAS para referência

#### Controles e Opções
- Botão para iniciar/pausar o reconhecimento
- Botão para limpar o histórico
- Opções de configuração (sensibilidade, etc.)

#### Rodapé
- Informações sobre o projeto
- Links úteis
- Créditos

## 5. Elementos de Interface

### 5.1 Botões
- Estilo arredondado com preenchimento de cor
- Hover state com leve escurecimento
- Ícones acompanhados de texto para melhor compreensão

### 5.2 Cards
- Cantos levemente arredondados
- Sombra suave para elevação
- Padding interno consistente

### 5.3 Feedback Visual
- Animações sutis para transições
- Destacar sinais reconhecidos com borda colorida
- Indicadores de progresso para carregamento do modelo

## 6. Responsividade

### 6.1 Desktop (>1024px)
- Layout conforme descrito acima
- Visualização da câmera e tradução lado a lado

### 6.2 Tablet (768px - 1024px)
- Visualização da câmera em tamanho reduzido
- Tradução abaixo da visualização da câmera

### 6.3 Mobile (<768px)
- Layout em coluna única
- Visualização da câmera ocupando largura total
- Seções empilhadas verticalmente
- Menu compacto (hamburger)

## 7. Microinterações

- Feedback visual ao detectar um sinal (highlight)
- Animação suave ao adicionar um novo sinal ao histórico
- Indicador pulsante durante o processamento
- Transição suave entre estados da interface

