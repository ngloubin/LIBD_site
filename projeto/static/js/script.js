document.getElementById('start').addEventListener('click', function() {
    fetch('/start_recording', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.status.includes('Erro')) {
                document.getElementById('error').textContent = data.status;
                document.getElementById('error').style.display = 'block';
            } else {
                document.getElementById('video').style.display = 'block';
                document.getElementById('video').src = '/video_feed?' + new Date().getTime();
                document.getElementById('start').disabled = true;
                document.getElementById('stop').disabled = false;
                document.getElementById('formar_tempo_real').disabled = false;
                document.getElementById('formar_video').disabled = true;
            }
        });
});

document.getElementById('stop').addEventListener('click', function() {
    fetch('/stop_recording', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            document.getElementById('video').style.display = 'none';
            document.getElementById('video').src = '';
            document.getElementById('start').disabled = false;
            document.getElementById('stop').disabled = true;
            document.getElementById('formar_tempo_real').disabled = true;
            document.getElementById('formar_video').disabled = false;
        });
});

document.getElementById('formar_tempo_real').addEventListener('click', function() {
    document.getElementById('loader').style.display = 'block';
    fetch('/formar_palavras_tempo_real', { method: 'POST' })
        .then(response => {
            if (!response.ok) throw new Error('Erro ' + response.status);
            return response.json();
        })
        .then(data => {
            document.getElementById('words').textContent = data.word || 'Nenhuma palavra formada';
            document.getElementById('loader').style.display = 'none';
        })
        .catch(error => {
            document.getElementById('error').textContent = 'Erro ao formar palavras: ' + error;
            document.getElementById('error').style.display = 'block';
            document.getElementById('loader').style.display = 'none';
        });
});

document.getElementById('formar_video').addEventListener('click', function() {
    document.getElementById('loader').style.display = 'block';
    fetch('/formar_palavras_video', { method: 'POST' })
        .then(response => {
            if (!response.ok) throw new Error('Erro ' + response.status);
            return response.json();
        })
        .then(data => {
            document.getElementById('words').textContent = data.word || 'Nenhuma palavra formada';
            document.getElementById('loader').style.display = 'none';
        })
        .catch(error => {
            document.getElementById('error').textContent = 'Erro ao formar palavras: ' + error;
            document.getElementById('error').style.display = 'block';
            document.getElementById('loader').style.display = 'none';
        });
});