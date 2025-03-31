from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import pytesseract
import openai
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from io import BytesIO
import os
api_key = os.getenv("OPENAI_API_KEY")


# Set OpenAI API key


# Path to the font file
font_path = "Noto_Sans_Devanagari 2/static/NotoSansDevanagari-Regular.ttf"

app = FastAPI()

class ImageRequest(BaseModel):
    image_url: str

def load_image_from_url(image_url):
    """Loads an image from a URL."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(image_url, headers=headers, stream=True)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    return None

def translate_text_with_openai(text):
    """Translates English text to Hindi using OpenAI API."""
    if not text.strip():
        return ""
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful translator."},
            {"role": "user", "content": f'Translate the following English text into Hindi: "{text}"'}
        ]
    )
    return response["choices"][0]["message"]["content"].strip() if response else ""

def extract_text_with_boxes(image):
    """Extracts text and bounding boxes from an image."""
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    data = pytesseract.image_to_data(gray, lang="eng", output_type=pytesseract.Output.DICT)
    return [{"text": data["text"][i], "bbox": (data["left"][i], data["top"][i], data["width"][i], data["height"][i])}
            for i in range(len(data["text"])) if data["text"][i].strip()]

def remove_text_from_image(pil_img, extracted_words):
    """Removes text from the image using inpainting."""
    img = np.array(pil_img)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for word_data in extracted_words:
        x, y, w, h = word_data["bbox"]
        mask[y:y+h, x:x+w] = 255  
    return cv2.inpaint(img, mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)

def overlay_translated_text(image, extracted_words, translated_words, font_path):
    """Overlays translated text onto the image."""
    draw = ImageDraw.Draw(image)
    for word_data, translated_word in zip(extracted_words, translated_words):
        if translated_word.strip():
            x, y, w, h = word_data["bbox"]
            font = ImageFont.truetype(font_path, 20)
            draw.text((x, y), translated_word, fill="black", font=font)
    return image

@app.post("/process-image")
async def process_image(request: ImageRequest):
    """Processes an image from a given URL."""
    image = load_image_from_url(request.image_url)
    if image is None:
        raise HTTPException(status_code=400, detail="Failed to load image.")
    extracted_words = extract_text_with_boxes(image)
    if not extracted_words:
        raise HTTPException(status_code=400, detail="No text detected.")
    english_text = " ".join([word["text"] for word in extracted_words])
    translated_text = translate_text_with_openai(english_text).split()
    clean_image = Image.fromarray(remove_text_from_image(image, extracted_words))
    final_image = overlay_translated_text(clean_image, extracted_words, translated_text, font_path)
    output_path = "translated_output.png"
    final_image.save(output_path)
    return {"message": "Image processed successfully", "output_image": output_path}
