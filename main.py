from PIL import Image, ImageDraw, ImageFont
import easyocr
import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm

# SAHI
from sahi.slicing import slice_image

# Phi-4
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Font path settings
FONT_PATH = "fonts/NotoSansCJKtc-Regular.otf"
#
#FONT_PATH = "arialuni.ttf"          # Arial Unicode MS (Windows)
# or
#FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Linux fallback
# or macOS
#FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
USE_GPU = True

# Phi-4-mini-instruct
PHI_MODEL_NAME = "microsoft/Phi-4-mini-instruct"

print(f"Loading Phi-4 model: {PHI_MODEL_NAME} ... (Can be slow for the first time)")
tokenizer = AutoTokenizer.from_pretrained(PHI_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    PHI_MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="cuda" if USE_GPU and torch.cuda.is_available() else "cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
model.eval()


def translate_with_phi(text: str, max_new_tokens=256) -> str:
    """Improved prompt: handle technical, possible cut-off, duplicates, OCR noise"""
    if not text.strip():
        return text

    prompt = f"""You are a professional technical translator specializing in English to Traditional Chinese (Taiwan usage).

Rules you must strictly follow:
- Translate accurately, naturally, and professionally.
- Preserve all technical terms, measurements, abbreviations, model names, part numbers — do NOT translate them unless they are common words.
- If the text appears cut-off, incomplete, or has minor OCR errors (e.g. "flow rate" → "fiow rate"), correct them intelligently but do NOT invent new content.
- If there are duplicated or redundant phrases, remove duplicates and produce clean output.
- Keep original structure: line breaks, lists, short phrases — do not turn into long paragraphs unless necessary.
- Output ONLY the translated text. No explanations, no English, no markdown.
- Remove duplicated text
- OCR may not be correct. Some words are merged so you should split and translate the text properly. 
For example : "Style Mame" is "Style Name"

Input text (may be fragmented or slightly noisy):
{text}

Traditional Chinese translation:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.08,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract translated part
    if "Traditional Chinese translation:" in generated:
        translated = generated.split("Traditional Chinese translation:")[-1].strip()
    else:
        translated = generated[len(prompt):].strip()

    # Cleanup
    translated = translated.split("\n\n")[0].strip()
    return translated if translated else text


def is_design_pack(bbox, image_shape):
    """
    Check if is in design pack area.
    (Assume design pack is always at the right part of image)

    Arguments: 
        - bbox: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    height, width = image_shape[:2]

    if bbox[0][0] > width * 0.65:
        return True

    return False


def translate2chinese(input_path, output_path,
                            SLICE_SIZE=1024,
                            OVERLAP_RATIO=0.25,
                            MIN_AREA_RATIO=0.05
                            ):
    img_cv = cv2.imread(input_path)
    if img_cv is None:
        raise ValueError(f"Cannot read image: {input_path}")

    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # EasyOCR only extract English - available in this situation
    reader = easyocr.Reader(['en'], gpu=USE_GPU)

    # ==================== SAHI + OCR ====================
    image_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    slice_result = slice_image(
        image=image_rgb,
        slice_height=SLICE_SIZE,
        slice_width=SLICE_SIZE,
        overlap_height_ratio=OVERLAP_RATIO,
        overlap_width_ratio=OVERLAP_RATIO,
        auto_slice_resolution=False,
        min_area_ratio=MIN_AREA_RATIO,
        verbose=False
    )

    all_detections = []
    translated_pairs = []
    
    for slice_obj in tqdm(slice_result, desc="OCR on SAHI slices", unit="slice"):
        tile_image = slice_obj['image']
        tile_results = reader.readtext(
            tile_image,
            paragraph=False,
            detail=1,
            canvas_size=2560,
            mag_ratio=1.5,
            text_threshold=0.65,
        )

        offset_x, offset_y = slice_obj['starting_pixel']

        for bbox_tile, text, prob in tile_results:
            global_bbox = [
                [int(pt[0] + offset_x), int(pt[1] + offset_y)]
                for pt in bbox_tile
            ]
            all_detections.append((global_bbox, text.strip(), prob))

    print(f"Detected {len(all_detections)} text boxes")

    # ==================== Process 1 by 1 ====================
    for bbox, text, prob in tqdm(all_detections, desc="Translating & overlaying (Phi-4)", unit="box"):
        if prob < 0.2 or is_design_pack(bbox, img_cv.shape):
            continue

        try:
            translated = translate_with_phi(text)
            translated_pairs.append((text, translated))
            print(f"Original: {text[:60]}... → translated: {translated[:60]}...")
        except Exception as e:
            print(f"Phi-4 failed to translate: {e}")
            translated = text

        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

        erase_box = [x1, y1, x2, y2]
        draw.rectangle(erase_box, fill=(255, 255, 255))

        box_w = x2 - x1
        box_h = y2 - y1
        font_size = max(10, int(box_h * 0.78))

        try:
            font = ImageFont.truetype(FONT_PATH, font_size)
        except:
            font = ImageFont.load_default()

        text_bbox = draw.textbbox((0, 0), translated, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        text_x = x1 + (box_w - text_w) // 2
        text_y = y1 + (box_h - text_h) // 2

        draw.text((text_x, text_y), translated, font=font, fill=(0, 0, 0))

    final_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, final_img)
    print(f"已儲存：{output_path}")

    # Draw translation
    txt_path = "translated_techpack.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Tech Pack ===\n\n")
        for orig, trans in translated_pairs:
            f.write(f"Original: {orig}\nTranslated: {trans}\n{'-'*70}\n")
    print(f"Translated text saved to: {txt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tech Pack Translator")
    parser.add_argument("--input", required=True, help="input image path")
    parser.add_argument("--output", default="output.jpg", help="output image path")
    args = parser.parse_args()

    translate2chinese(args.input, args.output)