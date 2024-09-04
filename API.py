from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Flaskアプリケーションのインスタンスを作成
app = Flask(__name__)

# トレーニング済みモデルとトークナイザーのロード
model_path = './bert-sakura-model'  # トレーニング済みモデルのパス
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# 推論モードに設定
model.eval()

# APIエンドポイントを作成
@app.route('/predict', methods=['POST'])
def predict():
    # リクエストからテキストデータを取得
    data = request.json
    text = data['text']

    # テキストをトークナイズ
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # モデルを使用して予測を行う
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    # 予測結果を返す
    result = {
        'text': text,
        'prediction': 'Sakura Review' if prediction == 1 else 'Genuine Review'
    }
    return jsonify(result)

# アプリケーションを起動
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
