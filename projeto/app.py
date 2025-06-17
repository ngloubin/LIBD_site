# from flask import Flask, render_template, Response, jsonify, request
# from ultralytics import YOLO
# import cv2
# import numpy as np
# from groq import Groq
# import os
# import threading
# from collections import deque
# import time
# import logging

# # Configuração de logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Configurações
# MODEL_PATH = "best.onnx"  # Substitua por "yolo11s.pt" se preferir usar o YOLO11s
# CONF_THRESHOLD = 0.3  # Ajustado para capturar mais detecções
# SINAIS = {0: "A", 1: "O", 2: "H", 3: "C", 4: "I", 5: "T", 6: "E", 7: "U"}  # Ajuste conforme o modelo
# VIDEO_PATH = os.path.join(os.getcwd(), "temp_video.mp4")

# # Carregar modelo YOLO
# try:
#     model = YOLO(MODEL_PATH, task='detect')
#     logger.info("Modelo YOLO carregado com sucesso")
# except Exception as e:
#     logger.error(f"Erro ao carregar o modelo YOLO: {e}")
#     raise

# # Cliente Groq
# client = Groq(api_key=os.getenv("GROQ_API_KEY", "sua-chave-aqui"))  # Coloque sua chave do Groq

# app = Flask(__name__)

# class VideoRecorder:
#     def __init__(self):
#         self.recording = False
#         self.cap = None
#         self.out = None
#         self.detected_letters = deque(maxlen=50)
#         self.lock = threading.Lock()

#     def start_recording(self):
#         if not self.recording:
#             self.recording = True
#             for index in [0, 1, 2]:
#                 self.cap = cv2.VideoCapture(index)
#                 if self.cap.isOpened():
#                     logger.info(f"Webcam encontrada no índice {index}")
#                     break
#             if not self.cap.isOpened():
#                 logger.error("Erro ao abrir a webcam")
#                 self.recording = False
#                 return False
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             self.out = cv2.VideoWriter(VIDEO_PATH, fourcc, 20.0, (640, 480))
#             threading.Thread(target=self.record_loop, daemon=True).start()
#             logger.info("Gravação iniciada")
#             return True
#         return True

#     def record_loop(self):
#         while self.recording:
#             if self.cap is None or not self.cap.isOpened():
#                 logger.error("Webcam não disponível")
#                 break
#             ret, frame = self.cap.read()
#             if not ret:
#                 logger.error("Erro ao capturar frame")
#                 break
#             self.out.write(frame)
#             time.sleep(0.05)

#     def stop_recording(self):
#         if self.recording:
#             self.recording = False
#             if self.out:
#                 self.out.release()
#             if self.cap:
#                 self.cap.release()
#             logger.info("Gravação parada")

#     def process_video(self):
#         self.detected_letters.clear()
#         if not os.path.exists(VIDEO_PATH):
#             logger.error("Vídeo não encontrado")
#             return []
#         cap = cv2.VideoCapture(VIDEO_PATH)
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             try:
#                 results = model(frame, conf=CONF_THRESHOLD)
#                 current_signs = [SINAIS.get(int(box.cls[0]), "") for r in results for box in r.boxes if box.cls[0] in SINAIS]
#                 if current_signs:
#                     with self.lock:
#                         self.detected_letters.extend(current_signs)
#             except Exception as e:
#                 logger.error(f"Erro na detecção: {e}")
#             time.sleep(0.01)
#         cap.release()
#         # Remove duplicatas mantendo ordem
#         return list(dict.fromkeys(self.detected_letters))

# recorder = VideoRecorder()

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/start_recording', methods=['POST'])
# def start_recording():
#     if recorder.start_recording():
#         return jsonify({'status': 'Gravação iniciada'})
#     return jsonify({'status': 'Erro ao iniciar gravação'}), 500

# @app.route('/stop_recording', methods=['POST'])
# def stop_recording():
#     recorder.stop_recording()
#     return jsonify({'status': 'Gravação parada'})

# @app.route('/formar_palavras', methods=['POST'])
# def formar_palavras():
#     letters = recorder.process_video()
#     if not letters:
#         return jsonify({'word': 'Nenhuma letra detectada'}), 400
#     letras_str = ", ".join(letters)
#     prompt = (
#         "FORMATO EXIGIDO: 'PALAVRAS: <lista>, SOBRAS: <letras_nao_usadas>'\n"
#         "Letras disponíveis APENAS ESTAS: [A, O, H, C, I, T, E, U] (MAIÚSCULAS)\n\n"
#         "Regras:\n"
#         "1. Use SOMENTE as letras fornecidas\n"
#         "2. Priorize cumprimentos: 'OI', 'TCHAU', 'ATE'\n"
#         "3. Depois, palavras comuns em português\n"
#         f"LETRAS DETECTADAS: {letras_str}\n"
#         "RESPOSTA NO FORMATO: 'PALAVRAS: [...], SOBRAS: [...]'"
#     )
#     try:
#         chat_completion = client.chat.completions.create(
#             messages=[{"role": "user", "content": prompt}],
#             model="llama3-70b-8192",
#             timeout=10
#         )
#         response = chat_completion.choices[0].message.content
#         palavras = response.split("PALAVRAS: ")[1].split(", SOBRAS: ")[0].strip()
#         return jsonify({'word': palavras})
#     except Exception as e:
#         logger.error(f"Erro ao chamar Groq: {e}")
#         return jsonify({'word': 'Erro ao formar a palavra'}), 500

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# def gen_frames():
#     while recorder.recording:
#         if recorder.cap is None or not recorder.cap.isOpened():
#             placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
#             cv2.putText(placeholder, "Webcam indisponivel", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#             ret, buffer = cv2.imencode('.jpg', placeholder)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#             break
#         ret, frame = recorder.cap.read()
#         if not ret:
#             break
#         try:
#             results = model(frame, conf=CONF_THRESHOLD)
#             annotated_frame = results[0].plot()
#             ret, buffer = cv2.imencode('.jpg', annotated_frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#         except Exception as e:
#             logger.error(f"Erro ao gerar frame: {e}")
#         time.sleep(0.1)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
import cv2
import numpy as np
from groq import Groq
import os
import threading
from collections import deque
import time
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações
MODEL_PATH = "best.onnx"  # Substitua pelo seu modelo
CONF_THRESHOLD = 0.3
SINAIS = {0: "A", 1: "O", 2: "H", 3: "C", 4: "I", 5: "T", 6: "E", 7: "U"}
VIDEO_PATH = os.path.join(os.getcwd(), "temp_video.mp4")

# Carregar modelo YOLO
try:
    model = YOLO(MODEL_PATH, task='detect')
    logger.info("Modelo YOLO carregado com sucesso")
except Exception as e:
    logger.error(f"Erro ao carregar o modelo YOLO: {e}")
    raise

# Cliente Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY", "sua-chave-aqui"))  # Substitua pela sua chave

app = Flask(__name__)

class VideoRecorder:
    def __init__(self):
        self.recording = False
        self.cap = None
        self.out = None
        self.detected_letters_real_time = deque(maxlen=50)
        self.detected_letters_video = deque(maxlen=50)
        self.lock = threading.Lock()

    def start_recording(self):
        if not self.recording:
            self.recording = True
            for index in [0, 1, 2]:
                self.cap = cv2.VideoCapture(index)
                if self.cap.isOpened():
                    logger.info(f"Webcam encontrada no índice {index}")
                    break
            if not self.cap.isOpened():
                logger.error("Erro ao abrir a webcam")
                self.recording = False
                return False
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(VIDEO_PATH, fourcc, 20.0, (640, 480))
            threading.Thread(target=self.record_loop, daemon=True).start()
            logger.info("Gravação iniciada")
            return True
        return True

    def record_loop(self):
        while self.recording:
            if self.cap is None or not self.cap.isOpened():
                logger.error("Webcam não disponível")
                break
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Erro ao capturar frame")
                break
            self.out.write(frame)
            try:
                results = model(frame, conf=CONF_THRESHOLD)
                current_signs = [SINAIS.get(int(box.cls[0]), "") for r in results for box in r.boxes if box.cls[0] in SINAIS]
                if current_signs:
                    with self.lock:
                        self.detected_letters_real_time.extend(current_signs)
                    logger.info(f"Letras detectadas em tempo real: {current_signs}")
            except Exception as e:
                logger.error(f"Erro na detecção em tempo real: {e}")
            time.sleep(0.05)

    def stop_recording(self):
        if self.recording:
            self.recording = False
            if self.out:
                self.out.release()
            if self.cap:
                self.cap.release()
            logger.info("Gravação parada")

    def process_video(self):
        self.detected_letters_video.clear()
        if not os.path.exists(VIDEO_PATH):
            logger.error("Vídeo não encontrado")
            return []
        cap = cv2.VideoCapture(VIDEO_PATH)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            try:
                results = model(frame, conf=CONF_THRESHOLD)
                current_signs = [SINAIS.get(int(box.cls[0]), "") for r in results for box in r.boxes if box.cls[0] in SINAIS]
                if current_signs:
                    with self.lock:
                        self.detected_letters_video.extend(current_signs)
                    logger.info(f"Letras detectadas no vídeo: {current_signs}")
            except Exception as e:
                logger.error(f"Erro na detecção no vídeo: {e}")
            time.sleep(0.01)
        cap.release()
        return list(dict.fromkeys(self.detected_letters_video))

    def get_real_time_letters(self):
        with self.lock:
            return list(dict.fromkeys(self.detected_letters_real_time))

recorder = VideoRecorder()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    if recorder.start_recording():
        return jsonify({'status': 'Gravação iniciada'})
    return jsonify({'status': 'Erro ao iniciar gravação'}), 500

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    recorder.stop_recording()
    return jsonify({'status': 'Gravação parada'})

@app.route('/formar_palavras_tempo_real', methods=['POST'])
def formar_palavras_tempo_real():
    letters = recorder.get_real_time_letters()
    logger.info(f"Letras para tempo real: {letters}")
    if not letters:
        logger.warning("Nenhuma letra detectada em tempo real")
        return jsonify({'word': 'Nenhuma letra detectada'}), 400
    return formar_palavras(letters)

@app.route('/formar_palavras_video', methods=['POST'])
def formar_palavras_video():
    letters = recorder.process_video()
    logger.info(f"Letras para vídeo: {letters}")
    if not letters:
        logger.warning("Nenhuma letra detectada no vídeo")
        return jsonify({'word': 'Nenhuma letra detectada'}), 400
    return formar_palavras(letters)

def formar_palavras(letters):
    letras_str = ", ".join(letters)
    prompt = (
        "FORMATO EXIGIDO: 'PALAVRAS: <lista>, SOBRAS: <letras_nao_usadas>'\n"
        "Letras disponíveis APENAS ESTAS: [A, O, H, C, I, T, E, U] (MAIÚSCULAS)\n\n"
        "Regras:\n"
        "1. Use SOMENTE as letras fornecidas\n"
        "2. Priorize cumprimentos: 'OI', 'TCHAU', 'ATE'\n"
        "3. Depois, palavras comuns em português\n"
        f"LETRAS DETECTADAS: {letras_str}\n"
        "RESPOSTA NO FORMATO: 'PALAVRAS: [...], SOBRAS: [...]'"
    )
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            timeout=10
        )
        response = chat_completion.choices[0].message.content
        logger.info(f"Resposta do Groq: {response}")
        if "PALAVRAS: " in response:
            palavras = response.split("PALAVRAS: ")[1].split(", SOBRAS: ")[0].strip("[]")
            return jsonify({'word': palavras if palavras else 'Nenhuma palavra formada'})
        logger.warning("Formato de resposta inválido do Groq")
        return jsonify({'word': 'Erro ao formar a palavra'}), 500
    except Exception as e:
        logger.error(f"Erro ao chamar Groq: {e}")
        return jsonify({'word': 'Erro ao formar a palavra'}), 500

@app.route('/video_feed')
def video_feed():
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
            results = model(frame, conf=CONF_THRESHOLD)
            annotated_frame = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            logger.error(f"Erro ao gerar frame: {e}")
        time.sleep(0.1)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)