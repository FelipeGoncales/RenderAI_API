from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from flask import Flask, jsonify, request, send_file
import os
from dotenv import load_dotenv
from flask_cors import CORS

# Obtém o token do .env
load_dotenv()

api_token = os.getenv("API_KEY")
app = Flask(__name__)

CORS(app)

# Rota para gerar imagem
@app.route('/gen_image', methods=["POST"])
def gen_image():
    prompt = request.form.get("prompt")

    if not prompt:
        return jsonify({
            "error": "Prompt não encontrado."
        }), 404

    file = request.files.get('image')

    if not file:
        return jsonify({"error": "Imagem não encontrada."}), 404

    image = Image.open(file)

    client = genai.Client(api_key=api_token)

    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[
            prompt,
            image
        ],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"]
        )
    )

    # Salva direto — o .data já são bytes PNG puros
    try:
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                img = Image.open(BytesIO(part.inline_data.data))

                img_io = BytesIO()

                img.save(img_io, format="PNG")

                img_io.seek(0)

                return send_file(
                    img_io,
                    mimetype="image/png",
                )

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)