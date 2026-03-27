                                               ，，from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session, flash, jsonify
import os
import subprocess
from detect_logic import setup_folders, handle_detection, get_image_counts, convert_video_format

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.secret_key = 'your_secret_key'
setup_folders()

USER_CREDENTIALS = {
    "admin": "123456",
    "user": "password"
}
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            flash('账号或密码错误')
            return render_template('first.html')
    return render_template('first.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        confirm = request.form['confirm'].strip()

        if not username or not password or not confirm:
            flash('请填写所有字段')
            return render_template('register.html')

        if username in USER_CREDENTIALS:
            flash('用户名已存在')
            return render_template('register.html')

        if password != confirm:
            flash('两次密码输入不一致')
            return render_template('register.html')

        USER_CREDENTIALS[username] = password
        flash('注册成功，请登录')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/detect_page')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html',
                           chart_labels=[],
                           chart_values=[],
                           pie_labels=[],
                           pie_values=[],
                           counts_data={},
                           lstm_forecast=[])

@app.route('/answer')
def answer():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('answer.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/detect', methods=['POST'])
def detect():
    if 'username' not in session:
        return redirect(url_for('login'))

    if 'file' not in request.files or request.files['file'].filename == '':
        return redirect(url_for('index'))

    file = request.files['file']
    result = handle_detection(file)

    ext = result['ext']
    filename = result['filename']
    chart_labels = result.get('chart_labels', [])
    chart_values = result.get('chart_values', [])
    pie_labels = result.get('pie_labels', [])
    pie_values = result.get('pie_values', [])
    counts_data = result.get('counts_data', {})
    lstm_forecast = result.get('lstm_forecast', [])
    output_folder = result['fixed_output_folder']

    if ext in ['jpg', 'jpeg', 'png']:
        output_img_path = os.path.join(output_folder, filename).replace("\\", "/")
        output_img_url = "/" + output_img_path
        counts = get_image_counts(filename)

        return render_template('index.html',
                               output_img=output_img_url,
                               counts=counts,
                               counts_data=counts_data,
                               chart_labels=chart_labels,
                               chart_values=chart_values,
                               pie_labels=pie_labels,
                               pie_values=pie_values,
                               lstm_forecast=[])

    elif ext in ['mp4', 'avi', 'mov']:
        detected_video = None
        for f in os.listdir(output_folder):
            if f.lower().endswith(('.mp4', '.avi', '.mov')):
                detected_video = os.path.join(output_folder, f)
                break

        if detected_video:
            converted_video = convert_video_format(detected_video)
            detected_video_url = '/' + converted_video.replace("\\", "/")

            return render_template('index.html',
                                   output_video=detected_video_url,
                                   counts_data=counts_data,
                                   chart_labels=chart_labels,
                                   chart_values=chart_values,
                                   pie_labels=pie_labels,
                                   pie_values=pie_values,
                                   lstm_forecast=lstm_forecast)
        else:
            return "未找到检测后的视频文件", 500
    else:
        return "不支持的文件格式", 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('static/uploads', filename)

# === Ollama llama3 聊天功能 新增部分 ===

def call_ollama(message):
    try:
        proc = subprocess.Popen(
            ['ollama', 'run', 'llama3', message],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        out, err = proc.communicate(timeout=30)
        if err:
            print(f"Ollama 错误输出: {err.strip()}")
        return out.strip(), err.strip()
    except subprocess.TimeoutExpired:
        proc.kill()
        return None, "请求超时"
    except Exception as e:
        return None, str(e)



@app.route('/chat', methods=['POST'])
def chat():
    if 'username' not in session:
        return jsonify({'error': '请先登录'}), 401

    data = request.get_json()
    message = data.get('message', '').strip()
    if not message:
        return jsonify({'error': '消息不能为空'}), 400

    reply, error = call_ollama(message)
    if error:
        return jsonify({'error': error}), 500
    if not reply:
        reply = "模型未返回结果。"
    return jsonify({'reply': reply})


if __name__ == '__main__':
    app.run(debug=True)
