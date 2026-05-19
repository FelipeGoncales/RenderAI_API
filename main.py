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
                f"""You are a world-class architectural visualization expert and photorealistic CGI artist working for a custom furniture company.

                The user sent this description in portuguese: "{prompt}"

                STEP 1 — Analyze the reference 3D sketch and extract:
                - Type of environment (bedroom, living room, kitchen, office, etc.)
                - All furniture pieces present and their exact positions
                - Room dimensions (small/medium/large based on proportions)
                - Camera angle, height and perspective
                - Window/door positions and sizes
                - Any structural elements (niches, beams, steps, etc.)

                STEP 2 — Generate a photorealistic image generation prompt in English that:
                1. Uses your analysis to accurately describe the scene (correct furniture, correct layout)
                2. Incorporates EVERYTHING the user described (colors, materials, style, objects, additions)
                3. Produces output IDENTICAL to a real interior photograph — not a render, not CGI

                STRICT RULES:
                - NEVER invent or add furniture not present in the sketch
                - NEVER remove furniture that exists in the sketch
                - NEVER change furniture positions
                - Preserve the EXACT camera angle from the reference
                - Translate all color names the user mentioned to professional material descriptions in English
                - If the user mentions brand color names (ex: "nox", "petale", "off-white"), describe them accurately as material finishes

                PHOTOREALISM RULES:
                - Output must be indistinguishable from a DSLR interior photograph
                - No visible sketch lines, no cartoon outlines, no CGI glow
                - Surfaces must have real-world imperfections: subtle grain, micro-scratches, fabric wrinkles
                - Lighting must cast physically accurate shadows
                - Materials must have realistic specularity

                OUTPUT FORMAT — return ONLY the prompt, no explanations, no preamble:

                Transform this 3D sketch into a photograph-quality interior image. Preserve the EXACT camera angle, room layout, furniture placement and proportions from the reference. Do not move, add or remove any furniture.

                **Environment identified:** [type of room and brief description from your analysis]

                **Furniture & Layout:** [list each furniture piece with its position, exactly as found in sketch]

                **Materials & Colors:** [translate and expand every material/color the user mentioned with professional finish descriptions]

                **User Requests:** [all additional elements, objects or changes the user described]

                **Lighting:**
                - Analyze the sketch and describe the most realistic lighting for this specific environment
                - Include natural light sources (windows, skylights) based on what exists in the sketch
                - Add appropriate artificial lighting for the room type
                - Physically accurate shadows, light bounce and ambient occlusion

                **Photography Settings:**
                - Camera: Sony A7R IV, wide angle lens appropriate for room size
                - Aperture f/4.0, ISO 800, natural grain
                - Perspective matching the original sketch angle exactly

                **Photorealism:**
                - Path-traced global illumination
                - PBR materials with roughness and metallic maps
                - 8K texture resolution
                - Subsurface scattering on fabric and organic materials
                - Realistic micro-surface detail on all materials
                - No CGI artifacts, no render glow, no outline strokes
                - Must pass as a real photograph taken on location

                **Quality:** Architectural Digest / Dezeen magazine interior photography quality"""
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
                img = Image.open(BytesIO(part.inline_data.data))
                img_io = BytesIO()
                img.save(img_io, format="PNG")
                img_io.seek(0)
                return send_file(img_io, mimetype="image/png")

        return jsonify({
            "error": "sem_imagem",
            "message": "Nenhuma imagem foi gerada. Tente novamente."
        }), 400

    except genai_errors.ClientError as e:
        if e.status_code == 429:
            return jsonify({
                "error": "limite_atingido",
                "message": "Limite de gerações atingido. Tente novamente em alguns minutos."
            }), 429

        return jsonify({
            "error": "erro_api",
            "message": f"Erro ao gerar imagem: {str(e)}"
        }), 400

    except genai_errors.ServerError as e:
        if e.status_code == 503:
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