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
                f"""You are an expert prompt engineer for photorealistic interior rendering AI models.

                Your ONLY job is to receive a 3D sketch image and a user description, and output a single image generation prompt that produces a PHOTOREALISTIC result — like a real interior photograph, not a 3D render.

                User description in portuguese: "{prompt}"

                ---

                ANALYZE THE IMAGE FIRST:
                Look carefully at the sketch and identify every detail before writing the prompt.

                ---

                NOW WRITE THE PROMPT following this exact structure:

                Photorealistic interior photograph of a [room type]. This is a real photo taken with a Sony A7R IV camera, 20mm wide angle lens, f/4.0 aperture, ISO 800. The room is real, the materials are real, the lighting is real.

                EXACT LAYOUT (do not change anything):
                Describe each piece of furniture exactly where it appears in the sketch. Same positions, same proportions, same camera angle. The viewer should feel they are standing in the exact same spot as the 3D sketch camera.

                MATERIALS:
                [Translate every color/material the user mentioned into realistic material descriptions. Example: "nox" = deep matte charcoal wood with subtle grain, "petale" = warm matte beige-cream with soft rosé undertone]
                - Walls: [from user or default: warm off-white with micro-texture plaster]
                - Floor: [from user or default: large format porcelain tile, low-gloss with subtle reflections]
                - Each furniture piece: [material, finish, texture]

                USER ADDITIONS:
                [Include every object, decoration or modification the user requested]

                LIGHTING:
                Soft warm interior lighting at 2700K. Natural light entering through the window creating soft directional shadows on the floor. Light bouncing off the floor onto furniture undersides. No harsh shadows. No blown highlights. Realistic ambient occlusion in room corners and under furniture.

                PHOTOGRAPHIC REALISM — CRITICAL:
                - ZERO sketch lines or outlines visible
                - ZERO CGI glow or artificial sheen
                - Real fabric wrinkles on bedding and soft surfaces
                - Real wood grain texture on wooden surfaces
                - Real dust particles visible in window light beam
                - Grout lines visible between floor tiles
                - Slight fingerprints or natural wear on matte surfaces
                - The image must be completely indistinguishable from a real interior photograph
                - Style: Architectural Digest magazine, professional interior photography

                OUTPUT ONLY THE PROMPT. NO EXPLANATIONS. NO PREAMBLE. NO MARKDOWN HEADERS."""
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
