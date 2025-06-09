import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Trash2, RefreshCw } from 'lucide-react';

const Controls = ({ onClearHistory, onResetDetection }) => {
  return (
    <div className="controls">
      <Button 
        variant="outline" 
        onClick={onClearHistory}
        className="flex items-center gap-2"
      >
        <Trash2 className="h-4 w-4" />
        Limpar Histórico
      </Button>
      
      <Button 
        variant="outline" 
        onClick={onResetDetection}
        className="flex items-center gap-2"
      >
        <RefreshCw className="h-4 w-4" />
        Reiniciar Detecção
      </Button>
    </div>
  );
};

export default Controls;

