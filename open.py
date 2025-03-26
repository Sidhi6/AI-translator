from flask import Flask, request, jsonify
import requests
import pytesseract
import openai
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from io import BytesIO
from dotenv import load_dotenv
import os

app = Flask(__name__)
# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Path to the font file
font_path = "NotoSansDevanagari-Regular.ttf"

def load_image_from_url(image_url):
    """Loads an image from a URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }
    response = requests.get(image_url, headers=headers, stream=True)

    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        print(f"⚠️ Error downloading image: {response.status_code} - {response.reason}")
        return None

def translate_text_with_openai(text):
    """Translates English text to Hindi using OpenAI API."""
    if not text.strip():
        return ""

    prompt = f"""
    Translate the following English text into correct Hindi while maintaining original spacing and word order where possible:
    "{text}"
    """

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": "You are a helpful translator."},
                  {"role": "user", "content": prompt}]
    )

    if response and response["choices"]:
        return response["choices"][0]["message"]["content"].strip()
    
    print("⚠️ Translation Failed!")
    return ""

def extract_text_with_boxes(image):
    """Extracts text and bounding boxes from an image."""
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    data = pytesseract.image_to_data(gray, lang="eng", output_type=pytesseract.Output.DICT)
    extracted_words = []

    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if not word:
            continue  

        bbox = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
        extracted_words.append({"text": word, "bbox": bbox})

    return extracted_words

def remove_text_from_image(pil_img, extracted_words):
    """Removes text from the image using inpainting."""
    img = np.array(pil_img)

    if img.shape[-1] == 4:  
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    for word_data in extracted_words:
        x, y, w, h = word_data["bbox"]
        pad = 10  
        mask[max(0, y-pad):min(img.shape[0], y+h+pad), max(0, x-pad):min(img.shape[1], x+w+pad)] = 255  

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    inpainted = cv2.inpaint(img, mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
    blurred = cv2.GaussianBlur(inpainted, (5, 5), 0)
    final_output = cv2.addWeighted(inpainted, 0.8, blurred, 0.2, 0)

    return final_output

def get_dynamic_font_size(text, box_width, font_path):
    """Determines the best font size to fit the text within a given width."""
    max_font_size = 20
    min_font_size = 10

    for font_size in range(max_font_size, min_font_size, -1):
        font = ImageFont.truetype(font_path, font_size)
        text_width = font.getbbox(text)[2] - font.getbbox(text)[0]

        if text_width <= box_width:
            return font  
    
    return ImageFont.truetype(font_path, min_font_size)

def overlay_translated_text(image, extracted_words, translated_words, font_path):
    """Overlays translated text onto the image."""
    draw = ImageDraw.Draw(image)

    if len(translated_words) != len(extracted_words):
        print("⚠️ Translation mismatch! Adjusting...")
        translated_words += [""] * (len(extracted_words) - len(translated_words))

    for word_data, translated_word in zip(extracted_words, translated_words):
        x, y, w, h = word_data["bbox"]
        
        if not translated_word.strip():
            continue  

        font = get_dynamic_font_size(translated_word, w, font_path)
        bbox = font.getbbox(translated_word) 
        text_width = bbox[2] - bbox[0] 
        text_height = bbox[3] - bbox[1] 

        adjusted_x = x + (w - text_width) // 2  
        adjusted_y = y + (h - text_height) // 2  

        draw.text((adjusted_x, adjusted_y), translated_word, fill="black", font=font)
    
    return image

import os
import requests
import pytesseract
import openai
from flask import Flask, request, jsonify
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from io import BytesIO

# Load OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Path to the font file (store it in the same folder as the script)
FONT_PATH = "NotoSansDevanagari-Regular.ttf"

app = Flask(__name__)

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

    prompt = f'Translate the following English text into Hindi: "{text}"'
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": "You are a helpful translator."},
                  {"role": "user", "content": prompt}]
    )

    return response["choices"][0]["message"]["content"].strip() if response else ""

@app.route("/translate", methods=["POST"])
def process_image():
    """API endpoint to process an image from a URL."""
    data = request.json
    image_url = data.get("image_url")

    if not image_url:
        return jsonify({"error": "Image URL is required"}), 400

    image = load_image_from_url(image_url)
    if not image:
        return jsonify({"error": "Failed to load image"}), 500

    extracted_text = pytesseract.image_to_string(image, lang="eng")
    translated_text = translate_text_with_openai(extracted_text)

    return jsonify({"original_text": extracted_text, "translated_text": translated_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)






# def process_image(image_url):
#     """Main function to process an image from a URL."""
#     image = load_image_from_url(image_url)
#     if image is None:
#         print("Error: Failed to load image from URL.")
#         return

#     extracted_words = extract_text_with_boxes(image)
#     if not extracted_words:
#         print("Error: No text was extracted from the image.")
#         return
    
#     # Extract all words in order
#     english_text = " ".join([word["text"] for word in extracted_words])
    
#     # Caching the translation result to avoid multiple API calls
#     translated_text = translate_text_with_openai(english_text)
#     translated_words = translated_text.split()

#     # Remove original text from image
#     clean_image = remove_text_from_image(image, extracted_words)
#     clean_pil_image = Image.fromarray(clean_image)

#     # Overlay translated text
#     final_image = overlay_translated_text(clean_pil_image, extracted_words, translated_words, font_path)

#     # Show and save final translated image
#     final_image.show()
#     final_image.save("translated_output.png")

# # Example Usage
# image_url = "https://bhumikaprakashan.com/uploads/ebook/ebook-175/14.jpg"
# process_image(image_url)
