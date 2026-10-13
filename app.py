import os
import secrets
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as rl_canvas

app = Flask(__name__)
# Secrets come from the environment so they never live in the repository.
# Set SECRET_KEY to keep sessions valid across restarts; otherwise a random key is used.
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or secrets.token_hex(32)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///pneumoinsight.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(50), nullable=False)


class PatientRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(50), unique=True, nullable=False)
    patient_name = db.Column(db.String(150), nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(20))
    filename = db.Column(db.String(100))
    result_filename = db.Column(db.String(100))
    prediction = db.Column(db.String(50))
    confidence = db.Column(db.String(20))
    created_by = db.Column(db.String(100))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    # NEW: stores the plain heatmap (no overlay) for quadrant analysis
    heatmap_raw_filename = db.Column(db.String(100))
    # NEW: stores pre-computed explainability text
    explanation = db.Column(db.Text)
    # NEW: stores confidence tier label
    confidence_tier = db.Column(db.String(30))


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ─────────────────────────────────────────────────────────────────────────────
# AI MODEL SETUP
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = 'best_densenet121_pneumonia.keras'
model = tf.keras.models.load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIDENCE TIER
# ─────────────────────────────────────────────────────────────────────────────

def get_confidence_tier(conf_float):

    pct = conf_float * 100
    if pct >= 90:
        return {
            'label': 'High Confidence',
            'badge_class': 'bg-success',
            'advice': 'This prototype achieved high model certainty on this input. Note: PneumoInsight is an educational research prototype developed as a final-year capstone project and is not approved for clinical use.'
        }
    elif pct >= 70:
        return {
            'label': 'Moderate Confidence',
            'badge_class': 'bg-warning text-dark',
            'advice': 'This prototype achieved moderate model certainty on this input. Results at this confidence level should be interpreted with caution. PneumoInsight is an educational research prototype and is not intended for clinical decision-making.'
        }
    else:
        return {
            'label': 'Low Confidence',
            'badge_class': 'bg-danger',
            'advice': 'This prototype achieved low model certainty on this input. This result is unreliable and should be disregarded for any practical purpose. PneumoInsight is an educational research prototype developed for academic demonstration only.'
        }


# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT 2 — GRAD-CAM WITH QUADRANT DETECTION
# Returns (overlay_img, raw_heatmap, explanation_text).
# ─────────────────────────────────────────────────────────────────────────────

def detect_active_quadrants(heatmap_norm, threshold=0.5):
    """
    Splits the 224×224 normalised heatmap into four quadrants:
      Top-Left     → Upper-left lung field
      Top-Right    → Upper-right lung field
      Bottom-Left  → Lower-left lung field (includes left base and costophrenic angle)
      Bottom-Right → Lower-right lung field (includes right base and costophrenic angle)

    Returns a list of activated region names (mean activation > threshold).
    Uses radiological convention: the image is a posterior-anterior (PA) view,
    so the patient's right lung appears on the LEFT side of the image.
    """
    h, w = heatmap_norm.shape
    mid_h, mid_w = h // 2, w // 2

    quadrants = {
        'right upper lung field':  heatmap_norm[:mid_h, :mid_w],    # image top-left  = patient's right
        'left upper lung field':   heatmap_norm[:mid_h, mid_w:],    # image top-right = patient's left
        'right lower lung field':  heatmap_norm[mid_h:, :mid_w],    # image bot-left  = patient's right
        'left lower lung field':   heatmap_norm[mid_h:, mid_w:],    # image bot-right = patient's left
    }

    activated = [name for name, region in quadrants.items() if region.mean() > threshold]
    return activated


def build_explanation(prediction, conf_float, activated_regions):
    """
    Builds explanation,from the prediction label, confidence value, and activated quadrants.
    """
    pct = conf_float * 100
    tier = get_confidence_tier(conf_float)

    region_friendly = {
        'right upper lung field': 'upper-right area of the lung (right upper lung field)',
        'left upper lung field':  'upper-left area of the lung (left upper lung field)',
        'right lower lung field': 'lower-right area of the lung (right lower lung field)',
        'left lower lung field':  'lower-left area of the lung (left lower lung field)',
    }

    if prediction == 'PNEUMONIA':
        if activated_regions:

            friendly_regions = [region_friendly.get(r, r) for r in activated_regions]
            region_str = ' and '.join(friendly_regions)
            count_word = 'these areas' if len(activated_regions) > 1 else 'this area'

            location_note = (
                f"The DenseNet121 heatmap highlighted the {region_str} as the most "
                f"important area for this decision. This means the model "
                f"detected signs of possible infection or fluid build-up "
                f"(pulmonary consolidation or infiltration) in {count_word}. "
                f"This pattern is commonly seen in bacterial or viral lung "
                f"infection (pneumonia)."
            )
        else:
            location_note = (
                "The DenseNet121 heatmap shows activity spread across both lungs "
                "without one single standout area. This kind of spread-out "
                "pattern can be seen in infections that affect both lungs at "
                "the same time (bilateral pneumonia) or in infections that "
                "spread through the lung tissue itself (interstitial pneumonia)."
            )

        explanation = (
            f"The DenseNet121 model is {pct:.2f}% confident that this chest X-ray "
            f"shows signs of PNEUMONIA (lung infection). "
            f"Confidence level: {tier['label']}. "
            f"{location_note} "
            f"Note: This result was produced by a DenseNet121 deep learning "
            f"model as part of an academic research prototype. The coloured "
            f"heatmap (Grad-CAM) shows which parts of the X-ray the DenseNet121 model focused "
            f"on — warmer colours (red, orange) mean higher AI attention."
        )

    else:  # NORMAL
        if activated_regions:
            friendly_regions = [region_friendly.get(r, r) for r in activated_regions]
            region_str = ' and '.join(friendly_regions)

            location_note = (
                f"The heatmap shows some colour in the {region_str}, but "
                f"these are faint and do not reach the level that would "
                f"suggest infection or fluid (consolidation). They most likely "
                f"represent normal structures visible on an X-ray, such as "
                f"the point where the airways enter the lung (hilum), the "
                f"heart outline (cardiac border), or the dome-shaped muscle "
                f"below the lungs (diaphragm)."
            )
        else:
            location_note = (
                "The heatmap shows only faint, spread-out colour with no "
                "single bright spot. This is what the DenseNet121 model typically produces "
                "when it does not detect any significant signs of infection "
                "or disease (pulmonary pathology) in the X-ray."
            )

        explanation = (
            f"The DenseNet121 model is {pct:.2f}% confident that this chest X-ray "
            f"appears NORMAL — no clear signs of pneumonia (lung infection) "
            f"were detected. "
            f"Confidence level: {tier['label']}. "
            f"{location_note} "
            f"Note: This result was produced by a DenseNet121 deep learning "
            f"model as part of an academic research prototype. The coloured "
            f"heatmap (Grad-CAM) shows which parts of the X-ray the DenseNet121 model focused "
            f"on when reaching this conclusion."
        )

    return explanation


def generate_gradcam(img_path, target_path, raw_heatmap_path=None):
    """
    Generates the Grad-CAM overlay and optionally saves the raw normalised
    heatmap as a separate greyscale PNG for downstream quadrant analysis.

    Returns (activated_regions list, explanation string, conf_float).
    Falls back if model is None.
    """
    if model is None:
        return [], "Model not loaded.", 0.0

    # ── Load and preprocess ──────────────────────────────────────────────────
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # ── Guided Grad-CAM ──────────────────────────────────────────────────────
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer("relu").output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch)
        loss = predictions[:, 0]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    guided_grads = (
        tf.cast(output > 0, "float32") *
        tf.cast(grads > 0, "float32") *
        grads
    )
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam = np.dot(output, weights)
    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)


    conf_float = float(predictions[0][0])
    pred_label = "PNEUMONIA" if conf_float > 0.5 else "NORMAL"
    # For NORMAL, confidence is inverted
    display_conf = conf_float if conf_float > 0.5 else 1.0 - conf_float

    activated_regions = detect_active_quadrants(heatmap_norm, threshold=0.5)
    explanation = build_explanation(pred_label, display_conf, activated_regions)

    # Saving raw heatmap as greyscale PNG
    if raw_heatmap_path:
        raw_grey = np.uint8(255 * heatmap_norm)
        cv2.imwrite(raw_heatmap_path, raw_grey)

    # ── Render and save overlay ──────────────────────────────────────────────
    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (224, 224))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_norm), cv2.COLORMAP_JET)
    output_img = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)
    cv2.imwrite(target_path, output_img)

    return activated_regions, explanation, display_conf


# ─────────────────────────────────────────────────────────────────────────────
#PDF REPORT
# ─────────────────────────────────────────────────────────────────────────────

def build_pdf_report(rec):

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=50, leftMargin=50,
        topMargin=50, bottomMargin=50
    )

    styles = getSampleStyleSheet()
    story = []

    # ── Colours ─────────────────────────────────────────────────────────────
    BRAND_BLUE   = colors.HexColor('#0052D4')
    LIGHT_GREY   = colors.HexColor('#F4F7F6')
    DARK_TEXT    = colors.HexColor('#1A1A2E')
    MUTED_TEXT   = colors.HexColor('#6C757D')
    SUCCESS_GRN  = colors.HexColor('#198754')
    WARNING_AMB  = colors.HexColor('#CC8800')
    DANGER_RED   = colors.HexColor('#DC3545')

    def badge_color(tier_label):
        if 'High' in tier_label:
            return SUCCESS_GRN
        elif 'Moderate' in tier_label:
            return WARNING_AMB
        else:
            return DANGER_RED

    # ── Custom paragraph styles ──────────────────────────────────────────────
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Title'],
        fontSize=20,
        textColor=BRAND_BLUE,
        spaceAfter=4,
        fontName='Helvetica-Bold'
    )
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=MUTED_TEXT,
        spaceAfter=12,
        fontName='Helvetica'
    )
    section_style = ParagraphStyle(
        'Section',
        parent=styles['Heading2'],
        fontSize=11,
        textColor=BRAND_BLUE,
        spaceBefore=14,
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=10,
        textColor=DARK_TEXT,
        leading=15,
        fontName='Helvetica'
    )
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=MUTED_TEXT,
        leading=12,
        fontName='Helvetica-Oblique',
        borderPad=6
    )

    # ── HEADER ───────────────────────────────────────────────────────────────
    story.append(Paragraph("PneumoInsight AI", title_style))
    story.append(Paragraph("Automated Pneumonia Diagnostic Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=BRAND_BLUE, spaceAfter=10))

    # ── PATIENT INFORMATION TABLE ────────────────────────────────────────────
    story.append(Paragraph("Patient Information", section_style))

    patient_data = [
        ['Patient Name', rec.patient_name,  'Patient ID',  rec.patient_id],
        ['Age',          str(rec.age or '—'), 'Gender',      rec.gender or '—'],
        ['Date & Time',  rec.timestamp.strftime('%Y-%m-%d %H:%M'), 'Assessed By', rec.created_by],
    ]

    patient_table = Table(patient_data, colWidths=[1.3*inch, 2.2*inch, 1.3*inch, 2.2*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND',  (0, 0), (0, -1), LIGHT_GREY),
        ('BACKGROUND',  (2, 0), (2, -1), LIGHT_GREY),
        ('FONTNAME',    (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME',    (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE',    (0, 0), (-1, -1), 9),
        ('TEXTCOLOR',   (0, 0), (-1, -1), DARK_TEXT),
        ('GRID',        (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, LIGHT_GREY]),
        ('PADDING',     (0, 0), (-1, -1), 6),
        ('VALIGN',      (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(patient_table)

    # ── DIAGNOSIS RESULT ────────────────────────────────────────────────────
    story.append(Paragraph("Diagnostic Result", section_style))

    tier_info = get_confidence_tier(
        float(rec.confidence.strip('%')) / 100 if rec.confidence else 0.0
    )
    b_col = badge_color(tier_info['label'])

    diag_color = DANGER_RED if rec.prediction == 'PNEUMONIA' else SUCCESS_GRN

    result_data = [
        ['Prediction',  Paragraph(f'<font color="{diag_color.hexval() if hasattr(diag_color,"hexval") else "#DC3545"}"><b>{rec.prediction}</b></font>', body_style),
         'Confidence',  rec.confidence or '—'],
        ['Confidence Tier',
         Paragraph(f'<b>{tier_info["label"]}</b>', body_style),
         'Clinical Advice',
         Paragraph(tier_info['advice'], disclaimer_style)],
    ]
    result_table = Table(result_data, colWidths=[1.3*inch, 2.2*inch, 1.3*inch, 2.2*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND',  (0, 0), (0, -1), LIGHT_GREY),
        ('BACKGROUND',  (2, 0), (2, -1), LIGHT_GREY),
        ('FONTNAME',    (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME',    (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE',    (0, 0), (-1, -1), 9),
        ('TEXTCOLOR',   (0, 0), (-1, -1), DARK_TEXT),
        ('GRID',        (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
        ('PADDING',     (0, 0), (-1, -1), 6),
        ('VALIGN',      (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(result_table)

    # ── IMAGES: Original + Grad-CAM side by side ────────────────────────────
    story.append(Paragraph("Chest X-Ray and Grad-CAM Heatmap Analysis", section_style))

    orig_path = os.path.join(app.config['UPLOAD_FOLDER'], rec.filename)
    heat_path = os.path.join(app.config['UPLOAD_FOLDER'], rec.result_filename)

    img_row = []
    img_labels = []

    if os.path.exists(orig_path):
        img_row.append(RLImage(orig_path, width=2.6*inch, height=2.6*inch))
        img_labels.append(Paragraph('<b>Original X-Ray</b>', body_style))
    else:
        img_row.append(Paragraph('[Image unavailable]', disclaimer_style))
        img_labels.append(Paragraph('', body_style))

    if os.path.exists(heat_path):
        img_row.append(RLImage(heat_path, width=2.6*inch, height=2.6*inch))
        img_labels.append(Paragraph(
            '<b>Grad-CAM Heatmap Overlay</b><br/>'
            '<font size="8" color="#6C757D">Red/orange regions indicate highest AI activation</font>',
            body_style
        ))
    else:
        img_row.append(Paragraph('[Heatmap unavailable]', disclaimer_style))
        img_labels.append(Paragraph('', body_style))

    img_table = Table(
        [img_row, img_labels],
        colWidths=[3.3*inch, 3.3*inch]
    )
    img_table.setStyle(TableStyle([
        ('ALIGN',   (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',  (0, 0), (-1, -1), 'TOP'),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.HexColor('#DEE2E6')),
    ]))
    story.append(img_table)

    # ── AI EXPLANATION ───────────────────────────────────────────────────────
    story.append(Paragraph("AI Explainability Analysis", section_style))

    explanation_text = rec.explanation or (
        "Explainability data not available for this record. "
        "Please re-run the analysis to generate an explanation."
    )
    story.append(Paragraph(explanation_text, body_style))

    # ── DISCLAIMER ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor('#DEE2E6'), spaceAfter=8))
    disclaimer_text = (
        "<b>RESEARCH PROTOTYPE NOTICE:</b> This report was generated by PneumoInsight, an "
        "educational AI decision-support system developed as a capstone research project "
        "at the University of Zimbabwe, Department of Analytics and Informatics (2026). "
        "This system is a proof-of-concept prototype and has not been clinically validated "
        "or approved for medical use. It is intended solely for academic demonstration of "
        "explainable deep learning techniques in medical imaging. No clinical decisions "
        "should be made on the basis of this output. &copy; 2026 B. Chikumbirike, University of Zimbabwe."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and check_password_hash(user.password, request.form.get('password')):
            login_user(user)
            return redirect(url_for('admin_panel') if user.role == 'Admin' else url_for('dashboard'))
        flash('Invalid login credentials.')
    return render_template('login.html')


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    if current_user.role == 'Admin':
        return redirect(url_for('admin_panel'))
    if request.method == 'POST':
        if current_user.role == 'Nurse':
            flash("Action Restricted: Contact IT Support.")
            return redirect(url_for('dashboard'))

        file = request.files.get('file')
        if file and model:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # ── Inference ────────────────────────────────────────────────────
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            preds = model.predict(np.expand_dims(img_array, axis=0))
            conf_raw = float(preds[0][0])
            label = "PNEUMONIA" if conf_raw > 0.5 else "NORMAL"
            display_conf = conf_raw if conf_raw > 0.5 else 1.0 - conf_raw
            conf_str = f"{display_conf:.2%}"

            # ── Confidence tier ───────────────────────────────────────────────
            tier_info = get_confidence_tier(display_conf)

            # ── Grad-CAM + quadrant detection ────────────────────────────────
            res_filename = "res_" + filename
            raw_filename = "raw_" + filename + ".png"
            _, explanation, _ = generate_gradcam(
                filepath,
                os.path.join(app.config['UPLOAD_FOLDER'], res_filename),
                raw_heatmap_path=os.path.join(app.config['UPLOAD_FOLDER'], raw_filename)
            )

            # ── Auto-generate Patient ID ──────────────────────────────────────
            pid = f"PID-{datetime.now().year}-{PatientRecord.query.count() + 1:03d}"

            # ── Save to DB ────────────────────────────────────────────────────
            db.session.add(PatientRecord(
                patient_id=pid,
                patient_name=request.form.get('patient_name'),
                age=request.form.get('age'),
                gender=request.form.get('gender'),
                filename=filename,
                result_filename=res_filename,
                heatmap_raw_filename=raw_filename,
                prediction=label,
                confidence=conf_str,
                explanation=explanation,
                confidence_tier=tier_info['label'],
                created_by=current_user.username
            ))
            db.session.commit()
            return redirect(url_for('history'))

    return render_template('dashboard.html')


@app.route('/history')
@login_required
def history():
    records = PatientRecord.query.order_by(PatientRecord.timestamp.desc()).all()
    return render_template('history.html', records=records)


@app.route('/download_report/<int:id>')
@login_required
def download_report(id):
    rec = PatientRecord.query.get_or_404(id)
    buffer = build_pdf_report(rec)
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"PneumoInsight_Report_{rec.patient_id}.pdf",
        mimetype='application/pdf'
    )


@app.route('/export_analytics')
@login_required
def export_analytics():
    records = PatientRecord.query.all()
    data = [
        {
            'Patient ID': r.patient_id,
            'Name': r.patient_name,
            'Age': r.age,
            'Gender': r.gender,
            'Prediction': r.prediction,
            'Confidence': r.confidence,
            'Confidence Tier': r.confidence_tier or '—',
            'Assessed By': r.created_by,
            'Timestamp': r.timestamp.strftime('%Y-%m-%d %H:%M') if r.timestamp else '—'
        }
        for r in records
    ]
    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Diagnostics')
        worksheet = writer.sheets['Diagnostics']
        for i, col in enumerate(df.columns):
            worksheet.set_column(i, i, max(len(col) + 2, 14))
    output.seek(0)
    return send_file(output, as_attachment=True, download_name="PneumoInsight_Analytics.xlsx")


@app.route('/admin', methods=['GET', 'POST'])
@login_required
def admin_panel():
    if current_user.role != 'Admin':
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        u = request.form.get('username')
        p = request.form.get('password')
        r = request.form.get('role')
        if not User.query.filter_by(username=u).first():
            db.session.add(User(
                username=u,
                password=generate_password_hash(p, method='pbkdf2:sha256'),
                role=r
            ))
            db.session.commit()
    return render_template('admin.html', users=User.query.all())


@app.route('/reset_password/<int:id>', methods=['POST'])
@login_required
def reset_password(id):
    if current_user.role == 'Admin':
        user = User.query.get(id)
        if user:
            user.password = generate_password_hash(
                request.form.get('new_password'), method='pbkdf2:sha256'
            )
            db.session.commit()
    return redirect(url_for('admin_panel'))


@app.route('/delete_user/<int:id>', methods=['POST'])
@login_required
def delete_user(id):
    if current_user.role == 'Admin':
        user = User.query.get(id)
        if user and user.username != 'admin':
            db.session.delete(user)
            db.session.commit()
    return redirect(url_for('admin_panel'))


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username='admin').first():
            # Set ADMIN_PASSWORD before the first run, or a random one is generated and printed once.
            admin_password = os.environ.get('ADMIN_PASSWORD')
            if not admin_password:
                admin_password = secrets.token_urlsafe(12)
                print(f"\nFirst run: created user 'admin' with password: {admin_password}")
                print("Change it from the admin panel. This message is shown only once.\n")
            db.session.add(User(
                username='admin',
                role='Admin',
                password=generate_password_hash(admin_password, method='pbkdf2:sha256')
            ))
            db.session.commit()
    # Debug mode exposes an interactive debugger, so it is opt-in: set FLASK_DEBUG=1 locally.
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1')
