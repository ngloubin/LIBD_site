from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
import cv2
import numpy as np
from groq import Groq
import os
import threading
import logging
import time
import re
from threading import Lock

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações
MODEL = YOLO("yolo11s.pt")  # Modelo pré-treinado YOLO11s
MODEL_PATH = "best77.onnx"    # Dados treinados no formato ONNX
CONF_THRESHOLD = 0.3
SINAIS = {0: "A", 1: "O", 2: "H", 3: "C", 4: "I", 5: "T", 6: "E", 7: "U"}
GROQ_API_KEY = "sua chave API aqui"
MAX_CONCURRENT = 2
current_processing = 0
processing_lock = Lock()

# Verificar GROQ_API_KEY
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY não configurada. Insira uma chave válida no código.")
    raise ValueError("GROQ_API_KEY não configurada. Insira uma chave válida no código.")
logger.info(f"GROQ_API_KEY configurada: {GROQ_API_KEY[:4]}...{GROQ_API_KEY[-4:]}")

# Carregar modelo YOLO11s com dados treinados no formato ONNX
try:
    logger.info(f"Carregando modelo YOLO11s com dados treinados de {MODEL_PATH}")
    model = YOLO(MODEL_PATH, task='detect')  # Carrega diretamente o modelo ONNX
    logger.info("Modelo YOLO11s carregado com sucesso")
except Exception as e:
    logger.error(f"Erro ao carregar o modelo YOLO: {e}")
    raise

# Cliente Groq
try:
    client = Groq(api_key=GROQ_API_KEY)
    logger.info("Cliente Groq configurado com sucesso")
except Exception as e:
    logger.error(f"Erro ao configurar o cliente Groq: {e}")
    raise

app = Flask(__name__)

class VideoRecorder:
    def __init__(self):
        self.recording = False
        self.cap = None
        self.out = None
        self.detected_letters_video = []
        self.lock = threading.Lock()

    def start_recording(self):
        if not self.recording:
            self.detected_letters_video.clear()
            self.recording = True
            for index in [0, 1, 2]:
                self.cap = cv2.VideoCapture(index)
                if self.cap.isOpened():
                    logger.info(f"Webcam encontrada no índice {index}")
                    break
            if not self.cap.isOpened():
                logger.warning("Webcam não encontrada")
                self.recording = False
                return False
            fourcc = cv2.VideoWriter_fourcc(*'vp09')
            self.out = cv2.VideoWriter('temp.webm', fourcc, 20.0, (640, 480))
            threading.Thread(target=self.record_loop, daemon=True).start()
            logger.info("Gravação de vídeo iniciada")
            return True
        return True

    def preprocess_frame(self, frame):
        frame = cv2.convertScaleAbs(frame, alpha=2.0, beta=25)  # Ajuste de contraste
        frame = cv2.resize(frame, (640, 640))  # Redimensionar para 640x640
        return frame

    def record_loop(self):
        while self.recording:
            if self.cap is None or not self.cap.isOpened():
                logger.error("Webcam não disponível")
                break
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Erro ao capturar frame")
                break
            if self.out:
                self.out.write(frame)
            try:
                frame = self.preprocess_frame(frame)
                results = model(frame, conf=CONF_THRESHOLD)
                current_signs = []
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        if cls in SINAIS:
                            current_signs.append(SINAIS[cls])
                            logger.info(f"Detecção: {SINAIS[cls]} (conf={conf:.2f})")
                if current_signs:
                    with self.lock:
                        self.detected_letters_video.extend(current_signs)
                    logger.info(f"Letras detectadas no vídeo: {current_signs}")
                else:
                    logger.info("Nenhuma letra detectada neste frame")
            except Exception as e:
                logger.error(f"Erro na detecção no vídeo: {e}")
            time.sleep(0.05)

    def stop_recording(self):
        if self.recording:
            self.recording = False
            if self.cap:
                self.cap.release()
            if self.out:
                self.out.release()
            logger.info("Gravação de vídeo parada")
            return list(dict.fromkeys(self.detected_letters_video))
        return []

    def process_video(self, video_data):
        global current_processing
        with processing_lock:
            if current_processing >= MAX_CONCURRENT:
                logger.warning("Servidor ocupado, limite de processamento atingido")
                return None
            current_processing += 1
        try:
            temp_path = f"temp_{int(time.time() * 1000)}.webm"
            with open(temp_path, 'wb') as f:
                f.write(video_data)
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                logger.error("Não foi possível abrir o vídeo")
                return []
            detected_letters = []
            frame_count = 0
            start_time = time.time()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                try:
                    frame = self.preprocess_frame(frame)
                    results = model(frame, conf=CONF_THRESHOLD)
                    for r in results:
                        for box in r.boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            if cls in SINAIS:
                                detected_letters.append(SINAIS[cls])
                                logger.info(f"Frame {frame_count}: Detecção {SINAIS[cls]} (conf={conf:.2f})")
                    if not results[0].boxes:
                        logger.info(f"Frame {frame_count}: Nenhuma letra detectada")
                except Exception as e:
                    logger.error(f"Erro na detecção no vídeo: {e}")
            cap.release()
            os.remove(temp_path)
            elapsed_time = time.time() - start_time
            logger.info(f"Total de frames processados: {frame_count}, Tempo: {elapsed_time:.2f}s")
            return list(dict.fromkeys(detected_letters))
        finally:
            with processing_lock:
                current_processing -= 1

@app.route('/')
def index():
    logger.info("Rota /index acessada")
    return render_template('index.html')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    logger.info("Iniciando gravação via /start_recording")
    if recorder.start_recording():
        return jsonify({'status': 'Gravação de vídeo iniciada (servidor)'})
    return jsonify({'status': 'Webcam não disponível no servidor. Use a gravação pelo navegador.'}), 200

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    logger.info("Parando gravação via /stop_recording")
    letters = recorder.stop_recording()
    return jsonify({'status': 'Gravação de vídeo parada', 'letters': letters})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    logger.info("Recebendo vídeo via /upload_video")
    if 'video' not in request.files:
        logger.error("Nenhum vídeo enviado")
        return jsonify({'error': 'Nenhum vídeo enviado', 'wait_message': 'Se a palavra não for exibida, pode ser um erro no servidor. Por favor, espere 15 segundos e tente novamente.'}), 400
    video_file = request.files['video']
    try:
        video_data = video_file.read()
        temp_path = f"temp_{int(time.time() * 1000)}.webm"
        with open(temp_path, 'wb') as f:
            f.write(video_data)
        logger.info(f"Vídeo salvo temporariamente como {temp_path}")
        return jsonify({'status': 'Vídeo salvo com sucesso', 'temp_path': temp_path})
    except Exception as e:
        logger.error(f"Erro ao salvar vídeo: {e}")
        return jsonify({'error': f'Erro ao salvar vídeo: {str(e)}', 'wait_message': 'Se a palavra não for exibida, pode ser um erro no servidor. Por favor, espere 15 segundos e tente novamente.'}), 500

@app.route('/formar_palavras_video', methods=['POST'])
def formar_palavras_video():
    logger.info("Processando vídeo via /formar_palavras_video")
    start_time = time.time()
    try:
        temp_path = request.json.get('temp_path')
        if not temp_path:
            logger.error("Caminho do vídeo não fornecido")
            return jsonify({'word': 'Caminho do vídeo não fornecido', 'error': True, 'wait_message': 'Se a palavra não for exibida, pode ser um erro no servidor. Por favor, espere 15 segundos e tente novamente.'}), 400
        if not os.path.exists(temp_path):
            logger.error(f"Vídeo {temp_path} não encontrado")
            return jsonify({'word': 'Vídeo não encontrado', 'error': True, 'wait_message': 'Se a palavra não for exibida, pode ser um erro no servidor. Por favor, espere 15 segundos e tente novamente.'}), 404
        with open(temp_path, 'rb') as f:
            video_data = f.read()
        letters = recorder.process_video(video_data)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if letters is None:
            logger.warning("Servidor ocupado, limite de processamento atingido")
            return jsonify({'word': 'Servidor ocupado, tente novamente em alguns segundos', 'error': True, 'wait_message': 'Se a palavra não for exibida, pode ser um erro no servidor. Por favor, espere 15 segundos e tente novamente.'}), 429
        logger.info(f"Letras para vídeo: {letters}")
        if not letters:
            logger.warning("Nenhuma letra detectada no vídeo")
            return jsonify({'word': 'Nenhuma letra detectada', 'error': True, 'wait_message': 'Se a palavra não for exibida, pode ser um erro no servidor. Por favor, espere 15 segundos e tente novamente.'}), 400
        response = formar_palavras(letters)
        logger.info(f"Tempo total de processamento: {time.time() - start_time:.2f} segundos")
        return response
    except Exception as e:
        logger.error(f"Erro ao processar vídeo: {e}")
        return jsonify({'word': f'Erro ao processar vídeo: {str(e)}', 'error': True, 'wait_message': 'Se a palavra não for exibida, pode ser um erro no servidor. Por favor, espere 15 segundos e tente novamente.'}), 500

def formar_palavras(letters):
    letras_str = ", ".join(letters)
    prompt = (
        "FORMATO EXIGIDO: 'PALAVRAS: [<palavra>], SOBRAS: [<letras_nao_usadas>]'\n"
        "Letras disponíveis APENAS ESTAS: [A, O, H, C, I, T, E, U] (MAIÚSCULAS)\n\n"
        "Regras ESTRITAS:\n"
        "1. Forme APENAS UM CUMPRIMENTO válido ('OI', 'TCHAU', 'ATE') se TODAS as letras necessárias estiverem presentes.\n"
        "2. Letras necessárias para cada cumprimento:\n"
        "   - 'OI': O, I\n"
        "   - 'TCHAU': T, C, H, A, U\n"
        "   - 'ATE': A, T, E\n"
        "3. NÃO forme palavras adicionais (ex.: 'CO', 'TIO', 'CATE') com letras restantes ou de qualquer outra forma.\n"
        "4. Todas as letras fornecidas que NÃO forem usadas no cumprimento DEVEM ser listadas como SOBRAS.\n"
        "5. Se nenhum cumprimento for possível, retorne PALAVRAS: [] e liste TODAS as letras como SOBRAS.\n\n"
        "Exemplos CORRETOS:\n"
        "- Letras: O, I → PALAVRAS: ['OI'], SOBRAS: []\n"
        "- Letras: T, C, H, A, U → PALAVRAS: ['TCHAU'], SOBRAS: []\n"
        "- Letras: A, T, E → PALAVRAS: ['ATE'], SOBRAS: []\n"
        "- Letras: A, T, E, O, C → PALAVRAS: ['ATE'], SOBRAS: [O, C]\n"
        "- Letras: O, H, C, I → PALAVRAS: ['OI'], SOBRAS: [H, C]\n"
        "- Letras: H, C, U → PALAVRAS: [], SOBRAS: [H, C, U]\n\n"
        f"LETRAS DETECTADAS: {letras_str}\n"
        "RESPOSTA OBRIGATÓRIA NO FORMATO: 'PALAVRAS: [<palavra>], SOBRAS: [<letras>]'"
    )
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="deepseek-r1-distill-llama-70b",
            temperature=0.0,
            max_tokens=2048,
            top_p=0.95,
            timeout=60
        )
        response = chat_completion.choices[0].message.content
        logger.info(f"Resposta do Groq: {response}")
        if "PALAVRAS: " in response and ", SOBRAS: " in response:
            palavras = response.split("PALAVRAS: ")[1].split(", SOBRAS: ")[0].strip("[]").replace("'", "")
            sobras = response.split(", SOBRAS: ")[1].strip("[]").replace("'", "").split(", ")
            sobras = [s.strip() for s in sobras if s.strip()]
            return jsonify({'word': palavras if palavras else 'Nenhuma palavra formada', 'sobras': sobras, 'error': False})
        logger.warning("Formato de resposta inválido do Groq, tentando extrair manualmente")
        match = re.search(r"PALAVRAS: \['(OI|TCHAU|ATE)'\]", response)
        if match:
            palavra = match.group(1)
            used_letters = set(palavra)
            sobras = [l for l in letters if l not in used_letters]
            return jsonify({'word': palavra, 'sobras': sobras, 'error': False})
        logger.error("Não foi possível extrair palavra válida")
        return jsonify({'word': 'Erro ao formar a palavra: formato inválido', 'error': True, 'wait_message': 'Se a palavra não for exibida, pode ser um erro no servidor. Por favor, espere 15 segundos e tente novamente.'}), 500
    except Exception as e:
        logger.error(f"Erro ao chamar Groq: {e}")
        return jsonify({'word': f'Erro ao formar a palavra: {str(e)}', 'error': True, 'wait_message': 'Se a palavra não for exibida, pode ser um erro no servidor. Por favor, espere 15 segundos e tente novamente.'}), 500

@app.route('/video_feed')
def video_feed():
    logger.info("Acessando feed de vídeo")
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    while recorder.recording:
        if recorder.cap is None or not recorder.cap.isOpened():
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Webcam indisponivel", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', placeholder)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            break
        ret, frame = recorder.cap.read()
        if not ret:
            break
        try:
            frame = recorder.preprocess_frame(frame)
            results = model(frame, conf=CONF_THRESHOLD)
            annotated_frame = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            logger.error(f"Erro ao gerar frame: {e}")
        time.sleep(0.1)

recorder = VideoRecorder()

if __name__ == '__main__':
    logger.info("Iniciando na porta 5000")
    app.run(host='0.0.0.0', port=5000, debug=True)