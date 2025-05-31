# DementiaCheckBot
# A Telegram bot for analyzing MRI images and generating reports. 

import logging, io
from datetime import datetime

import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError

from telegram import Update, InputFile
from telegram.error import TelegramError
from telegram.ext import (Updater, CommandHandler, MessageHandler,
                          Filters, ConversationHandler, CallbackContext)

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader


# model & constants 
model = tf.keras.models.load_model(
    "EN_B0v9999.keras",
    custom_objects={"GaussianNoise": tf.keras.layers.GaussianNoise},
)
class_names = [
    "MildDemented", "ModerateDemented",
    "NonDemented",  "VeryMildDemented"
]

doctor_notes = {
    "NonDemented":      "No signs of dementia detected. Maintain regular check‑ups.",
    "VeryMildDemented": "Very mild cognitive symptoms observed. Recommend monitoring.",
    "MildDemented":     "Mild dementia detected. Clinical evaluation advised.",
    "ModerateDemented": "Moderate dementia identified. Consult a neurologist promptly.",
}

NAME, AGE, GENDER, SYMPTOMS, REASON, IMAGE = range(6)
user_data: dict[str, any] = {}

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)


# ─────────────────────────  conversation flow  ──────────────────────────
def start(update: Update, context: CallbackContext) -> int:
    update.message.reply_text("👋 Welcome to DementiaCheckBot!\nWhat is the patient's name?")
    return NAME


def get_name(update: Update, context: CallbackContext) -> int:
    user_data["name"] = update.message.text.title().strip()
    update.message.reply_text("📆 Patient’s age?")
    return AGE


def get_age(update: Update, context: CallbackContext) -> int:
    user_data["age"] = update.message.text.strip()
    update.message.reply_text("⚧️ Gender? (Male / Female / Prefer not to say)")
    return GENDER


def get_gender(update: Update, context: CallbackContext) -> int:
    user_data["gender"] = update.message.text.capitalize().strip()
    update.message.reply_text("🧠 List symptoms (comma‑separated) \n e.g Memory loss, Confusion, Headaches, Dizziness")
    return SYMPTOMS


def get_symptoms(update: Update, context: CallbackContext) -> int:
    user_data["symptoms"] = [s.strip().capitalize() for s in update.message.text.split(",")]
    update.message.reply_text("📄 Reason for scan? (comma‑separated) \n e.g Routine check, Family history, Head trauma")
    return REASON


def get_reason(update: Update, context: CallbackContext) -> int:
    user_data["reason"] = [r.strip().capitalize() for r in update.message.text.split(",")]
    update.message.reply_text("📸 Upload the MRI image (as a photo, not as a file).")
    return IMAGE


# ─────────────────────────  image handler  ──────────────────────────────
def get_image(update: Update, context: CallbackContext) -> int:
    logging.info("🔔 get_image called")

    # 1) Validate we actually got a photo
    if not update.message.photo:
        update.message.reply_text("⚠️ Please send the MRI as an in‑chat photo, not as a file.")
        return IMAGE

    # 2) Download highest‑resolution version
    try:
        photo_file = update.message.photo[-1].get_file()
        logging.info("Photo file_id: %s", photo_file.file_id)
        stream = io.BytesIO()
        photo_file.download(out=stream)
        stream.seek(0)
    except TelegramError as e:
        logging.exception("Telegram download error")
        update.message.reply_text("⚠️ Failed to download the image from Telegram.")
        return ConversationHandler.END

    # 3) Load with PIL
    try:
        image = Image.open(stream).convert("RGB")
        logging.info("Image opened OK  size=%s  mode=%s", image.size, image.mode)
    except UnidentifiedImageError:
        logging.exception("PIL could not identify image")
        update.message.reply_text("⚠️ Unsupported image format. Please try JPG or PNG.")
        return ConversationHandler.END

    user_data["image"] = image  # keep for PDF

    # 4) ML prediction
    logging.info("Running model.predict()")
    pre = np.expand_dims(
        tf.keras.applications.efficientnet.preprocess_input(
            np.array(image.resize((224, 224)), dtype="float32")
        ),
        0,
    )
    pred_vec = model.predict(pre)
    label = class_names[int(np.argmax(pred_vec))]
    user_data["diagnosis"] = label

    update.message.reply_text(f"✅ Prediction: *{label}*", parse_mode="Markdown")
    update.message.reply_text("📄 Generating PDF report…")

    # 5) PDF creation + send
    pdf_buf = generate_pdf(user_data)
    pdf_buf.seek(0)

    try:
        update.message.reply_document(
            InputFile(pdf_buf, filename="BrainCheck_Report.pdf"))
    except TelegramError as e:
        logging.exception("Telegram failed to send PDF")
        update.message.reply_text("⚠️ Couldn’t send PDF (see bot logs).")

    return ConversationHandler.END


def cancel(update: Update, context: CallbackContext) -> int:
    update.message.reply_text("❌ Conversation cancelled.")
    return ConversationHandler.END

def help_command(update: Update, context: CallbackContext) -> None:
    update.message.reply_text(
        "🆘 *DementiaCheckBot Help*\n\n"
        "/start  –  begin a new MRI analysis\n"
        "/cancel –  abort the current conversation\n\n"
        "Flow:\n"
        "1️⃣ Name  2️⃣ Age  3️⃣ Gender  4️⃣ Symptoms  5️⃣ Reason  6️⃣ MRI photo\n\n"
        "• Send the MRI as a *photo* (JPEG/PNG ≤ 20 MB).\n"
        "• You’ll get a prediction *and* a PDF report.\n"
        "• This bot is for demo purposes and doesn’t replace medical advice.",
        parse_mode="Markdown"
    )

# ─────────────────────────  pdf generator  ──────────────────────────────
def generate_pdf(d: dict) -> io.BytesIO:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)

    # header
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(300, 750, "MRI Report")
    c.setFont("Helvetica", 10)
    c.drawCentredString(300, 735, datetime.now().strftime("%B %d, %Y  %I:%M %p"))

    y, lh = 700, 18

    # demographics
    c.setFont("Helvetica-Bold", 12); c.drawString(50, y, "Patient Demographics")
    y -= lh; c.setFont("Helvetica", 11)
    c.drawString(70, y, f"Name: {d['name']}");      y -= lh
    c.drawString(70, y, f"Age: {d['age']}");        y -= lh
    c.drawString(70, y, f"Gender: {d['gender']}");  y -= lh + 10

    # findings
    c.setFont("Helvetica-Bold", 12); c.drawString(50, y, "Medical Examination Findings")
    y -= lh; c.setFont("Helvetica", 11)
    c.drawString(70, y, "Symptoms:");  y -= lh
    for s in d.get("symptoms") or ["None reported"]:
        c.drawString(90, y, f"• {s}"); y -= lh
    y -= 5
    c.drawString(70, y, "Reason for Scan:"); y -= lh
    for r in d.get("reason") or ["Not specified"]:
        c.drawString(90, y, f"• {r}"); y -= lh

    # diagnosis + image
    y -= 15
    c.setFont("Helvetica-Bold", 12); c.drawString(50, y, "Diagnostic Results")
    y -= lh; c.setFont("Helvetica", 11)
    c.drawString(70, y, f"Predicted Diagnosis: {d['diagnosis']}")

    try:
        t = io.BytesIO(); d["image"].save(t, "PNG"); t.seek(0)
        c.drawImage(ImageReader(t), 370, y - 20, width=200, height=200)
    except Exception as e:
        logging.warning("Image embed failed: %s", e)
        c.drawString(370, y - 20, "[image error]")

    # recommendation
    y -= 120
    c.setFont("Helvetica-Bold", 12); c.drawString(50, y, "Doctor’s Recommendation")
    y -= lh; c.setFont("Helvetica", 11)
    c.drawString(70, y, doctor_notes.get(d["diagnosis"], "Further evaluation advised."))

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 40, "Note: Report generated by DementiaCheckBot – not a clinical diagnosis.")
    c.showPage(); c.save(); buf.seek(0)
    return buf


# ─────────────────────────  global error logger  ────────────────────────
def log_error(update: Update, context: CallbackContext):
    logging.exception("Unhandled exception:", exc_info=context.error)


# ─────────────────────────  main loop  ───────────────────────────────────
def main() -> None:
    timeouts = dict(connect_timeout=15, read_timeout=15)
    updater = Updater("YOUR_TBOT_TOKEN", request_kwargs=timeouts, use_context=True)
    dp = updater.dispatcher

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            NAME:     [MessageHandler(Filters.text & ~Filters.command, get_name)],
            AGE:      [MessageHandler(Filters.text & ~Filters.command, get_age)],
            GENDER:   [MessageHandler(Filters.text & ~Filters.command, get_gender)],
            SYMPTOMS: [MessageHandler(Filters.text & ~Filters.command, get_symptoms)],
            REASON:   [MessageHandler(Filters.text & ~Filters.command, get_reason)],
            IMAGE:    [MessageHandler(Filters.photo, get_image)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    dp.add_handler(conv)
    dp.add_error_handler(log_error)
    dp.add_handler(CommandHandler("help", help_command))

    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
