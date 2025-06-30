// script.js
document.addEventListener('DOMContentLoaded', () => {
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const formVideoBtn = document.getElementById('formVideoBtn');
    const wordDisplay = document.getElementById('word');
    const errorDisplay = document.getElementById('error');
    let stream = null;
    let mediaRecorder = null;
    let recordedBlobs = [];

    // Usar sessionStorage pra isolar temp_path por aba
    const getTempPath = () => sessionStorage.getItem('temp_path');
    const setTempPath = (path) => sessionStorage.setItem('temp_path', path);
    const clearTempPath = () => sessionStorage.removeItem('temp_path');

    startBtn.addEventListener('click', async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            const videoElement = document.getElementById('video');
            videoElement.srcObject = stream;
            videoElement.play();

            recordedBlobs = [];
            const options = { mimeType: 'video/webm;codecs=vp9' };
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                errorDisplay.textContent = 'Formato video/webm;codecs=vp9 não suportado. Tente outro navegador. Se a palavra não for exibida, pode ser um erro no servidor. Por favor, espere 15 segundos e tente novamente.';
                stream.getTracks().forEach(track => track.stop());
                setTimeout(() => { errorDisplay.textContent = ''; }, 15000);
                return;
            }
            mediaRecorder = new MediaRecorder(stream, options);
            mediaRecorder.ondataavailable = event => {
                if (event.data && event.data.size > 0) {
                    recordedBlobs.push(event.data);
                }
            };
            mediaRecorder.onstop = () => {
                if (recordedBlobs.length === 0) {
                    errorDisplay.textContent = 'Erro: Nenhum dado gravado. Se a palavra não for exibida, pode ser um erro no servidor. Por favor, espere 15 segundos e tente novamente.';
                    stream.getTracks().forEach(track => track.stop());
                    setTimeout(() => { errorDisplay.textContent = ''; }, 15000);
                    return;
                }
                const blob = new Blob(recordedBlobs, { type: 'video/webm' });
                const formData = new FormData();
                formData.append('video', blob, 'recorded_video.webm');

                const tryFetch = async (retries = 2, delay = 1000) => {
                    for (let i = 0; i < retries; i++) {
                        try {
                            const response = await fetch('/upload_video', {
                                method: 'POST',
                                body: formData,
                                signal: AbortSignal.timeout(30000)
                            });
                            if (!response.ok) {
                                throw new Error(`Erro no servidor: ${response.status}`);
                            }
                            const data = await response.json();
                            if (data.error) {
                                errorDisplay.textContent = `${data.error} ${data.wait_message || ''}`;
                                wordDisplay.textContent = '';
                                setTimeout(() => { errorDisplay.textContent = ''; }, 15000);
                            } else {
                                setTempPath(data.temp_path);
                                errorDisplay.textContent = 'Vídeo enviado com sucesso! Clique em "Formar Palavras" para processar.';
                                wordDisplay.textContent = '';
                            }
                            return;
                        } catch (err) {
                            if (i < retries - 1) {
                                await new Promise(resolve => setTimeout(resolve, delay));
                                continue;
                            }
                            errorDisplay.textContent = `Erro ao enviar vídeo: ${err.message}. Se a palavra não for exibida, pode ser um erro no servidor. Por favor, espere 15 segundos e tente novamente.`;
                            wordDisplay.textContent = '';
                            setTimeout(() => { errorDisplay.textContent = ''; }, 15000);
                        }
                    }
                };
                tryFetch();
            };
            mediaRecorder.start(1000);
            errorDisplay.textContent = 'Gravação iniciada!';
            setTimeout(() => {
                if (mediaRecorder && mediaRecorder.state === 'recording') {
                    mediaRecorder.stop();
                    stream.getTracks().forEach(track => track.stop());
                    errorDisplay.textContent = 'Gravação parada automaticamente (10s). Enviando vídeo...';
                }
            }, 10000);
        } catch (err) {
            errorDisplay.textContent = `Erro ao acessar a webcam: ${err.message}. Tente usar HTTPS ou permitir a webcam. Se a palavra não for exibida, pode ser um erro no servidor. Por favor, espere 15 segundos e tente novamente.`;
            setTimeout(() => { errorDisplay.textContent = ''; }, 15000);
        }
    });

    stopBtn.addEventListener('click', () => {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
            stream.getTracks().forEach(track => track.stop());
            errorDisplay.textContent = 'Gravação parada. Enviando vídeo...';
        } else {
            errorDisplay.textContent = 'Nenhuma gravação em andamento.';
        }
    });

    formVideoBtn.addEventListener('click', () => {
        const tempPath = getTempPath();
        if (!tempPath) {
            errorDisplay.textContent = 'Nenhum vídeo enviado. Grave um vídeo primeiro. Se a palavra não for exibida, pode ser um erro no servidor. Por favor, espere 15 segundos e tente novamente.';
            setTimeout(() => { errorDisplay.textContent = ''; }, 15000);
            return;
        }
        errorDisplay.textContent = 'Processando letras do vídeo...';
        const tryFetch = async (retries = 2, delay = 1000) => {
            for (let i = 0; i < retries; i++) {
                try {
                    const response = await fetch('/formar_palavras_video', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ temp_path: tempPath }),
                        signal: AbortSignal.timeout(120000)
                    });
                    if (!response.ok) {
                        if (response.status === 429) {
                            errorDisplay.textContent = 'Servidor ocupado, tente novamente em alguns segundos. Se a palavra não for exibida, pode ser um erro no servidor. Por favor, espere 15 segundos e tente novamente.';
                            wordDisplay.textContent = '';
                            setTimeout(() => { errorDisplay.textContent = ''; }, 15000);
                            return;
                        }
                        throw new Error(`Erro no servidor: ${response.status}`);
                    }
                    const data = await response.json();
                    if (data.error) {
                        errorDisplay.textContent = `${data.word} ${data.wait_message || ''}`;
                        wordDisplay.textContent = '';
                        setTimeout(() => { errorDisplay.textContent = ''; }, 15000);
                    } else {
                        wordDisplay.textContent = data.word;
                        errorDisplay.textContent = '';
                    }
                    clearTempPath();
                    return;
                } catch (error) {
                    if (i < retries - 1) {
                        await new Promise(resolve => setTimeout(resolve, delay));
                        continue;
                    }
                    errorDisplay.textContent = `Erro ao formar palavras: ${error.message}. Se a palavra não for exibida, pode ser um erro no servidor. Por favor, espere 15 segundos e tente novamente.`;
                    wordDisplay.textContent = '';
                    setTimeout(() => { errorDisplay.textContent = ''; }, 15000);
                }
            }
        };
        tryFetch();
    });
});