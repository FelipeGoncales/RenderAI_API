from google import genai
from google.genai import types, errors as genai_errors
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

@app.route('/gen_image', methods=["POST"])
def gen_image():
    prompt = request.form.get("prompt")

    file = request.files.get('image')

    if not file:
        return jsonify({"error": "Imagem não encontrada."}), 404

    image = Image.open(file)

    client = genai.Client(api_key=api_token)

    # ── 1. Enhance do prompt ───────────────────────────────────────────
    try:
        # ── 1. Enhance do prompt com análise da imagem ─────────────────────
        enhance_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                image,
                f"""
                You are an expert AI specialized in architectural image-to-image transformation.
                
                The user provided:
                1. A reference interior design sketch/render
                2. A description in Portuguese of desired modifications
                
                User request:
                "{prompt}"
                
                TASK:
                Transform the reference image into a highly photorealistic interior photograph while preserving the original architecture and layout.
                
                ANALYZE THE REFERENCE IMAGE:
                - Identify the room type
                - Identify all furniture and objects
                - Identify materials, walls, floor, lighting and structure
                - Identify camera angle and perspective
                
                STRICT PRESERVATION RULES:
                - Preserve the EXACT room layout
                - Preserve furniture positions
                - Preserve camera angle and composition
                - Preserve room proportions
                - Preserve architectural structure
                - Do NOT add new furniture unless explicitly requested
                - Do NOT remove existing furniture unless explicitly requested
                
                MODIFICATION RULES:
                - Apply ALL user-requested changes clearly and visibly
                - Material changes must be obvious
                - Color changes must be dominant and easy to notice
                - Requested objects and finishes must appear realistically integrated
                - Translate all Portuguese color/material descriptions into realistic professional interior design materials
                
                PHOTOREALISM RULES:
                - The final image must look like a real interior photo
                - Realistic lighting and shadows
                - Natural reflections
                - Real-world material textures
                - No CGI look
                - No sketch appearance
                - No cartoon effect
                - No artificial outlines
                
                STYLE:
                - Luxury interior photography
                - Modern architectural photography
                - Realistic exposure and contrast
                - Physically accurate materials
                
                IMPORTANT:
                The generated image must clearly reflect the user modifications while preserving the original environment structure.
                """
            ]
        )

        enhanced_prompt = enhance_response.text.strip()

    except genai_errors.ClientError as e:
        if e.code == 429:
            return jsonify({
                "error": "limite_atingido",
                "message": "Limite de gerações atingido. Tente novamente em alguns minutos."
            }), 429

        return jsonify({
            "error": "erro_api",
            "message": f"Erro ao processar o prompt: {str(e)}"
        }), 400

    except genai_errors.ServerError as e:
        if e.code == 503:
            return jsonify({
                "error": "servico_indisponivel",
                "message": "O serviço está com alta demanda no momento. Tente novamente em instantes."
            }), 503

        return jsonify({
            "error": "erro_servidor",
            "message": f"Erro interno do servidor de IA: {str(e)}"
        }), 500

    # ── 2. Gera a imagem ───────────────────────────────────────────────
    try:
        file.seek(0)
        image = Image.open(file)

        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[enhanced_prompt, image],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"]
            )
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
        
                img_io = BytesIO(part.inline_data.data)
        
                img_io.seek(0)
        
                return send_file(
                    img_io,
                    mimetype="image/png"
                )

        return jsonify({
            "error": "sem_imagem",
            "message": "Nenhuma imagem foi gerada. Tente novamente."
        }), 400

    except genai_errors.ClientError as e:
        if e.code == 429:
            return jsonify({
                "error": "limite_atingido",
                "message": "Limite de gerações atingido. Tente novamente em alguns minutos."
            }), 429

        return jsonify({
            "error": "erro_api",
            "message": f"Erro ao gerar imagem: {str(e)}"
        }), 400

    except genai_errors.ServerError as e:
        if e.code == 503:
            return jsonify({
                "error": "servico_indisponivel",
                "message": "O serviço está com alta demanda no momento. Tente novamente em instantes."
            }), 503

        return jsonify({
            "error": "erro_servidor",
            "message": f"Erro interno do servidor de IA: {str(e)}"
        }), 500

    except Exception as e:
        return jsonify({
            "error": "erro_desconhecido",
            "message": f"Erro inesperado: {str(e)}"
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
