import os
import io
import warnings
import uuid
import bcrypt
from datetime import datetime, timedelta
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import google.generativeai as genai
import cv2
from deepface import DeepFace
from PIL import Image
import time
import sqlite3
from typing import Optional, Tuple
import base64

warnings.filterwarnings("ignore")

# ===================== Page Config & Enhanced Styles =====================
st.set_page_config(
    page_title="DataPro: Secure Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
body { background-color: #f0f2f6; font-family: 'Inter', sans-serif; }
.stApp { max-width: 1400px; margin: 0 auto; }
.auth-container {
    background: linear-gradient(135deg, #2b2d42 0%, #4f5b93 100%);
    padding: 2.5rem; border-radius: 20px; color: white;
    text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.2); margin: 2rem 0;
}
.auth-header { font-size: 2rem; font-weight: 700; margin-bottom: 1.5rem; }
.auth-button { background: #38ef7d; color: #1a1a1a; font-weight: 600; padding: 0.8rem 1.5rem; border-radius: 12px; transition: transform 0.2s; }
.auth-button:hover { transform: scale(1.05); }
.success-auth { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 1.5rem; border-radius: 15px; color: white; text-align: center; margin: 1rem 0; animation: fadeIn 0.5s; }
.failed-auth { background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%); padding: 1.5rem; border-radius: 15px; color: white; text-align: center; margin: 1rem 0; }
.sidebar .sidebar-content { background: #ffffff; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); padding: 1rem; }
.tab-container { background: #ffffff; border-radius: 15px; padding: 1.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
</style>
""", unsafe_allow_html=True)

# ===================== Database Setup =====================
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY, username TEXT UNIQUE, password_hash TEXT,
        face_embedding BLOB, created_at TIMESTAMP
    )''')
    conn.commit()
    conn.close()

init_db()

# ===================== Session State Initialization =====================
if "initialized" not in st.session_state:
    st.session_state.update({
        "processed_data": None, "original_data": None, "last_uploaded_file": None,
        "gemini_chat": None, "gemini_key": None, "data_brief": "",
        "authenticated": False, "current_user": None, "auth_attempts": 0,
        "max_auth_attempts": 3, "auth_mode": "credentials", "registration_complete": False,
        "just_registered_user": None, "session_expiry": None
    })
    st.session_state.initialized = True

# ===================== Face Auth Helpers =====================
class DeepFaceTransformer(VideoTransformerBase):
    def __init__(self, model_name="ArcFace"):
        self.face_embedding = None
        self.error = None
        self.model_name = model_name
        self.face_detected = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr")
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            result = DeepFace.extract_faces(rgb_frame, detector_backend="retinaface", enforce_detection=False)
            if not result or len(result) == 0 or ("face" not in result[0]) or result[0]["face"] is None:
                self.error = "No face detected in the webcam frame. Please ensure your face is visible and well-lit."
                self.face_detected = False
                return img
            if len(result) > 1:
                self.error = "Multiple faces detected. Please show only one face."
                self.face_detected = False
                return img
            face = result[0]["face"]
            face_uint8 = (face * 255).astype(np.uint8) if face.max() <= 1.0 else face.astype(np.uint8)
            embedding = DeepFace.represent(face_uint8, model_name=self.model_name, detector_backend="skip")[0]["embedding"]
            self.face_embedding = np.array(embedding)
            self.error = None
            self.face_detected = True
            return img
        except Exception as e:
            self.error = f"Error processing webcam frame: {str(e)}. Try adjusting lighting or camera position."
            self.face_detected = False
            return img

def encode_face_from_image(image: Image.Image, model_name: str = "ArcFace") -> Tuple[Optional[np.ndarray], Optional[str]]:
    try:
        img_array = np.array(image)
        if img_array.ndim == 3 and img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
        if img_array.ndim == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        rgb_image = img_array if img_array.shape[-1] == 3 else cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        result = DeepFace.extract_faces(rgb_image, detector_backend="retinaface", enforce_detection=False)
        if not result or len(result) == 0:
            return None, "No face detected in the image. Please upload a clear, well-lit photo with one face."
        if len(result) > 1:
            return None, "Multiple faces detected. Please upload an image with only one face."
        face = result[0]["face"]
        if face is None or face.size == 0:
            return None, "Invalid face array. Please upload a valid image."
        face_uint8 = (face * 255).astype(np.uint8) if face.max() <= 1.0 else face.astype(np.uint8)
        embedding = DeepFace.represent(face_uint8, model_name=model_name, detector_backend="skip")[0]["embedding"]
        return np.array(embedding), None
    except Exception as e:
        return None, f"Error processing image: {str(e)}. Ensure the image is clear and contains one face."

def authenticate_face(test_embedding: np.ndarray, threshold: float = 8.0) -> Tuple[bool, Optional[str]]:
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT user_id, face_embedding FROM users WHERE face_embedding IS NOT NULL")
    users = c.fetchall()
    conn.close()
    if not users:
        return False, None
    for user_id, embedding_blob in users:
        try:
            ref_embedding = np.frombuffer(embedding_blob, dtype=np.float64)
            distance = np.linalg.norm(test_embedding - ref_embedding)
            if distance < threshold:
                return True, user_id
        except Exception as e:
            st.error(f"Verification error for user ID {user_id}: {str(e)}")
    return False, None

def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def register_user(username: str, password: str, face_image: Optional[Image.Image] = None) -> Tuple[bool, str]:
    user_id = str(uuid.uuid4())
    password_hash = hash_password(password) if password else None
    face_embedding = None
    if face_image:
        embedding, error = encode_face_from_image(face_image)
        if error:
            return False, error
        face_embedding = embedding.tobytes()
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (user_id, username, password_hash, face_embedding, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, username, password_hash, face_embedding, datetime.now())
        )
        conn.commit()
        return True, user_id
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    except Exception as e:
        return False, f"Database error: {str(e)}"
    finally:
        conn.close()

def authenticate_credentials(username: str, password: str) -> Tuple[bool, Optional[str]]:
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT user_id, password_hash FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()
        if user and user[1]:
            user_id, password_hash = user
            if verify_password(password, password_hash):
                return True, user_id
            else:
                return False, "Invalid password."
        return False, "Username not found."
    except Exception as e:
        return False, f"Authentication error: {str(e)}"

# ===================== Authentication UI =====================
def login_page():
    st.markdown('<div class="auth-container"><h2 class="auth-header">🔐 DataPro Login</h2></div>', unsafe_allow_html=True)
    auth_type = st.radio("Login Method", ["Credentials", "Face Authentication"], index=0, horizontal=True)
    st.session_state.auth_mode = "credentials" if auth_type == "Credentials" else "face"
    if st.session_state.auth_mode == "face":
        face_login()
    else:
        credentials_login()

def credentials_login():
    with st.form("credentials_form"):
        st.subheader("🔑 Login with Credentials")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login", type="primary")
        if submit:
            if not username or not password:
                st.markdown('<div class="failed-auth">❌ Please enter both username and password.</div>', unsafe_allow_html=True)
            else:
                success, result = authenticate_credentials(username, password)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.current_user = username
                    st.session_state.auth_attempts = 0
                    st.session_state.session_expiry = datetime.now() + timedelta(minutes=30)
                    st.markdown(f'<div class="success-auth">✅ Welcome back, {username}!</div>', unsafe_allow_html=True)
                    st.balloons()
                    time.sleep(0.6)
                    st.rerun()
                else:
                    st.session_state.auth_attempts += 1
                    remaining = st.session_state.max_auth_attempts - st.session_state.auth_attempts
                    st.markdown(f'<div class="failed-auth">❌ {result}. {remaining} attempts remaining.</div>', unsafe_allow_html=True)
                    st.info("💡 Ensure username and password are correct. Register if you haven't.")

def face_login():
    st.markdown('<div class="auth-container"><h3>🔐 Face Authentication</h3></div>', unsafe_allow_html=True)
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT username FROM users WHERE face_embedding IS NOT NULL")
    registered_users = [row[0] for row in c.fetchall()]
    conn.close()
    if not registered_users:
        st.warning("⚠️ No users registered with face authentication. Please register with a face image first.")
        return
    st.info(f"👥 Registered Users with Face Data: {', '.join(registered_users)}")
    model_name = st.selectbox("Face Recognition Model", ["ArcFace", "Facenet", "VGG-Face", "DeepFace"], key="auth_model")
    threshold = st.slider("Face Match Threshold (lower = stricter)", 5.0, 15.0, 8.0, 0.5)
    
    st.subheader("📷 Upload Photo to Authenticate")
    auth_image = st.file_uploader("Upload a clear photo with one face", type=['jpg', 'jpeg', 'png'], key="auth_upload")
    if auth_image:
        image = Image.open(auth_image).convert("RGB")
        st.image(image, caption="Authentication Photo", width=300)
        if st.button("Authenticate with Photo", key="auth_btn"):
            with st.spinner("Verifying identity..."):
                embedding, error = encode_face_from_image(image, model_name)
                if embedding is not None:
                    is_match, user_id = authenticate_face(embedding, threshold)
                    if is_match:
                        conn = sqlite3.connect("users.db")
                        c = conn.cursor()
                        c.execute("SELECT username FROM users WHERE user_id = ?", (user_id,))
                        username = c.fetchone()[0]
                        conn.close()
                        st.session_state.authenticated = True
                        st.session_state.current_user = username
                        st.session_state.auth_attempts = 0
                        st.session_state.session_expiry = datetime.now() + timedelta(minutes=30)
                        st.markdown(f'<div class="success-auth">✅ Welcome back, {username}!</div>', unsafe_allow_html=True)
                        st.balloons()
                        time.sleep(0.6)
                        st.rerun()
                    else:
                        st.session_state.auth_attempts += 1
                        remaining = st.session_state.max_auth_attempts - st.session_state.auth_attempts
                        st.markdown(f'<div class="failed-auth">❌ Face not recognized. {remaining} attempts remaining.</div>', unsafe_allow_html=True)
                        st.info("💡 Ensure the photo matches a registered user's face.")
                else:
                    st.session_state.auth_attempts += 1
                    remaining = st.session_state.max_auth_attempts - st.session_state.auth_attempts
                    st.markdown(f'<div class="failed-auth">❌ {error} {remaining} attempts remaining.</div>', unsafe_allow_html=True)
                    st.info("💡 Upload a clear, well-lit photo with one face.")

    st.subheader("📹 Webcam Authentication")
    ctx = webrtc_streamer(
        key="auth_webcam",
        video_transformer_factory=lambda: DeepFaceTransformer(model_name),
        media_stream_constraints={"video": True, "audio": False}
    )
    if ctx and ctx.video_transformer:
        transformer = ctx.video_transformer
        if transformer.error:
            st.markdown(f'<div class="failed-auth">❌ {transformer.error}</div>', unsafe_allow_html=True)
        elif transformer.face_detected:
            st.success("✅ Face detected in webcam. Click below to authenticate.")
        else:
            st.warning("⚠️ No face detected yet. Ensure your face is visible and well-lit.")
        
        if st.button("Authenticate with Webcam", key="webcam_auth_btn", disabled=not transformer.face_detected):
            with st.spinner("Verifying identity..."):
                if transformer.face_embedding is not None:
                    is_match, user_id = authenticate_face(transformer.face_embedding, threshold)
                    if is_match:
                        conn = sqlite3.connect("users.db")
                        c = conn.cursor()
                        c.execute("SELECT username FROM users WHERE user_id = ?", (user_id,))
                        username = c.fetchone()[0]
                        conn.close()
                        st.session_state.authenticated = True
                        st.session_state.current_user = username
                        st.session_state.auth_attempts = 0
                        st.session_state.session_expiry = datetime.now() + timedelta(minutes=30)
                        st.markdown(f'<div class="success-auth">✅ Welcome back, {username}!</div>', unsafe_allow_html=True)
                        st.balloons()
                        time.sleep(0.6)
                        st.rerun()
                    else:
                        st.session_state.auth_attempts += 1
                        remaining = st.session_state.max_auth_attempts - st.session_state.auth_attempts
                        st.markdown(f'<div class="failed-auth">❌ Face not recognized. {remaining} attempts remaining.</div>', unsafe_allow_html=True)
                        st.info("💡 Ensure your face matches a registered user's face.")
                else:
                    st.session_state.auth_attempts += 1
                    remaining = st.session_state.max_auth_attempts - st.session_state.auth_attempts
                    st.markdown(f'<div class="failed-auth">❌ No face detected in webcam frame. {remaining} attempts remaining.</div>', unsafe_allow_html=True)
                    st.info("💡 Adjust your position or lighting to ensure one face is visible.")

def register_user_face():
    st.markdown('<div class="auth-container"><h2 class="auth-header">👤 Register New User</h2></div>', unsafe_allow_html=True)
    if st.session_state.get("registration_complete", False):
        username = st.session_state.get("just_registered_user", "")
        st.markdown(f'<div class="success-auth">🎉 User "{username}" registered successfully!</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➡️ Continue to App", key="reg_go_to_app"):
                st.session_state.authenticated = True
                st.session_state.current_user = username
                st.session_state.session_expiry = datetime.now() + timedelta(minutes=30)
                st.session_state.registration_complete = False
                st.rerun()
        with col2:
            if st.button("↩️ Register Another User", key="reg_another_user"):
                st.session_state.registration_complete = False
                st.session_state.just_registered_user = None
                st.rerun()
        st.stop()
    with st.form("register_form"):
        st.subheader("User Details")
        username = st.text_input("Username (unique)", key="register_username")
        password = st.text_input("Password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm_password")
        st.subheader("📷 Optional: Add Face Authentication")
        face_image = st.file_uploader("Upload a clear photo with one face", type=['jpg', 'jpeg', 'png'], key="register_upload")
        submit = st.form_submit_button("Register", type="primary")
        if submit:
            if not username or not password or password != confirm_password:
                st.markdown('<div class="failed-auth">❌ Please provide a username and matching passwords.</div>', unsafe_allow_html=True)
            else:
                success, result = register_user(username, password, face_image)
                if success:
                    st.session_state.registration_complete = True
                    st.session_state.just_registered_user = username
                    st.rerun()
                else:
                    st.markdown(f'<div class="failed-auth">❌ Registration failed: {result}</div>', unsafe_allow_html=True)

# ===================== Data Helpers =====================
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file), None
        else:
            return pd.read_excel(file), None
    except Exception as e:
        return None, str(e)

@st.cache_data
def detect_outliers_counts(df, numeric_cols):
    out_counts = {}
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            out_counts[col] = {"IQR": 0, "Z": 0, "ModZ": 0}
            continue
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR = Q3 - Q1 if pd.notnull(Q3) and pd.notnull(Q1) else 0
        iqr_ct = int(((s < Q1 - 1.5*IQR) | (s > Q3 + 1.5*IQR)).sum()) if IQR != 0 else 0
        z = np.abs(stats.zscore(s, nan_policy="omit"))
        z_ct = int((z > 3).sum()) if isinstance(z, np.ndarray) else 0
        med = np.median(s)
        mad = np.median(np.abs(s - med)) if len(s) else 0
        if mad == 0:
            modz_ct = 0
        else:
            modz = 0.6745 * (s - med) / mad
            modz_ct = int((np.abs(modz) > 3.5).sum())
        out_counts[col] = {"IQR": iqr_ct, "Z": z_ct, "ModZ": modz_ct}
    return out_counts

def safe_mode_fill(series):
    mode_values = series.mode()
    if len(mode_values) > 0:
        return mode_values.iloc[0]
    return series.median() if pd.api.types.is_numeric_dtype(series) else "Unknown"

def build_data_brief(df: pd.DataFrame, max_cats=12) -> str:
    lines = []
    lines.append(f"ROWS={len(df)}, COLS={len(df.columns)}")
    dtypes = {c: str(t) for c, t in df.dtypes.items()}
    lines.append("DTYPES=" + "; ".join([f"{c}:{t}" for c, t in dtypes.items()]))
    miss = df.isnull().sum()
    miss_pct = (miss/len(df)*100).round(2)
    if len(miss[miss>0]) > 0:
        lines.append("MISSING=" + "; ".join([f"{c}:{int(miss[c])} ({miss_pct[c]}%)" for c in miss.index if miss[c]>0]))
    else:
        lines.append("MISSING=None")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()[:8]
    if num_cols:
        lines.append("NUMERIC_SUMMARY=")
        for c in num_cols:
            s = df[c].dropna()
            if s.empty:
                continue
            lines.append(f"  {c}: min={s.min():.3g}, q1={s.quantile(.25):.3g}, med={s.median():.3g}, q3={s.quantile(.75):.3g}, max={s.max():.3g}, mean={s.mean():.3g}, std={s.std():.3g}, skew={s.skew():.3g}")
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()[:8]
    if cat_cols:
        lines.append("CATEGORICAL_TOPS=")
        for c in cat_cols:
            vc = df[c].astype(str).value_counts(dropna=True).head(5)
            joined = ", ".join([f"{i}:{int(v)}" for i, v in vc.items()])
            lines.append(f"  {c}: {joined}")
    if num_cols:
        outs = detect_outliers_counts(df, num_cols)
        lines.append("OUTLIERS_SNAPSHOT=" + "; ".join([f"{c}(IQR={o['IQR']},Z={o['Z']},MZ={o['ModZ']})" for c, o in outs.items()]))
    return "\n".join(lines)

def ensure_gemini(model_name="gemini-1.5-flash"):
    key = st.session_state.gemini_key or os.getenv("GEMINI_API_KEY")
    if not key:
        st.warning("⚠️ Gemini API key is missing. Add it in the sidebar or set GEMINI_API_KEY env var.")
        return None
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel(model_name)
        if st.session_state.gemini_chat is None:
            st.session_state.gemini_chat = model.start_chat(history=[
                {"role": "user", "parts": ["You are a data analyst assistant. Answer concisely with clear bullet points and, when helpful, short formulas. If unsure, ask a scoped follow-up."]},
                {"role": "model", "parts": ["Understood. I'll analyze the provided dataset context."]},
            ])
        return st.session_state.gemini_chat
    except Exception as e:
        st.error(f"Gemini init error: {e}")
        return None

def df_to_latex_table(df: pd.DataFrame, max_rows=100) -> str:
    df = df.head(max_rows).copy()
    df = df.fillna('NaN')
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace('_', '\\_').str.replace('&', '\\&').str.replace('%', '\\%').str.replace('$', '\\$').str.replace('#', '\\#')
    num_cols = len(df.columns)
    header = ' & '.join([f'\\textbf{{{col}}}' for col in df.columns]) + ' \\\\ \\hline\n'
    rows = []
    for _, row in df.iterrows():
        row_str = ' & '.join(row.astype(str)) + ' \\\\'
        rows.append(row_str)
    latex_table = f"""
\\begin{{tabular}}{{{'l' * num_cols}}}
\\hline
{header}
{''.join(rows)}
\\hline
\\end{{tabular}}
"""
    return latex_table

def create_latex_document(df: pd.DataFrame, title: str = "DataPro Export") -> str:
    latex_table = df_to_latex_table(df)
    latex_content = f"""
\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage{{lmodern}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{geometry}}
\\geometry{{a4paper, margin=1in}}
\\begin{{document}}
\\title{{{title}}}
\\author{{DataPro Analytics}}
\\date{{}}
\\maketitle
\\section{{Dataset}}
The following table contains the dataset exported from DataPro. Rows are limited to 100 for brevity.
\\bigskip
{latex_table}
\\end{{document}}
"""
    return latex_content

# ===================== Main App Logic =====================
if st.session_state.auth_attempts >= st.session_state.max_auth_attempts:
    st.error("🚫 Maximum authentication attempts exceeded. Please refresh the page to try again.")
    st.stop()

if not st.session_state.authenticated:
    st.title("🔐 DataPro: Secure Analytics Platform")
    auth_tab1, auth_tab2 = st.tabs(["🔑 Login", "👤 Register"])
    with auth_tab1:
        login_page()
    with auth_tab2:
        register_user_face()
    st.stop()

if st.session_state.session_expiry and datetime.now() > st.session_state.session_expiry:
    st.session_state.authenticated = False
    st.session_state.current_user = None
    st.session_state.session_expiry = None
    st.error("🕒 Session expired. Please log in again.")
    st.rerun()

# ===================== Main App UI =====================
st.title(f"📊 DataPro: Advanced Analytics Dashboard")
st.markdown(f"👋 Welcome, **{st.session_state.current_user}**! Your session expires at {st.session_state.session_expiry.strftime('%H:%M:%S')}.", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.header("🔧 Dashboard Controls")
st.sidebar.markdown(f"🟢 Logged in as: {st.session_state.current_user}")
if st.sidebar.button("🚪 Logout", key="logout_btn"):
    st.session_state.authenticated = False
    st.session_state.current_user = None
    st.session_state.session_expiry = None
    st.rerun()

st.sidebar.markdown("---")
st.session_state.gemini_key = st.sidebar.text_input(
    "Gemini API Key (optional if set via env)", type="password", value=st.session_state.gemini_key or ""
)
uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"], key="file_uploader")

if st.sidebar.button("🗑️ Clear Data", use_container_width=True):
    for k in ["processed_data", "original_data", "last_uploaded_file", "data_brief"]:
        st.session_state[k] = None
    st.rerun()

# ===================== Load Data =====================
if uploaded_file:
    if st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.processed_data = None
        st.session_state.original_data = None
        st.session_state.last_uploaded_file = uploaded_file.name

    with st.spinner("Loading dataset..."):
        df, err = load_data(uploaded_file)
    if err:
        st.error(f"❌ {err}")
        st.stop()

    st.session_state.original_data = df.copy()
    st.session_state.processed_data = df.copy()
    st.session_state.data_brief = build_data_brief(df)
    st.success("✅ File loaded successfully!")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["📋 Overview", "🔍 Data Quality", "🛠 Cleaning", "📊 Visualizations", "📈 Advanced Analytics", "💾 Export", "🤖 AI Insights"]
    )

    # ---------- Overview ----------
    with tab1:
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)
        cur = st.session_state.processed_data
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(cur):,}", help="Total number of rows")
        c2.metric("Columns", cur.shape[1], help="Total number of columns")
        c3.metric("Memory (MB)", f"{cur.memory_usage(deep=True).sum()/1024**2:.2f}", help="Memory usage")
        c4.metric("Numeric Cols", len(cur.select_dtypes(include=np.number).columns), help="Numeric columns count")
        st.subheader("👀 Data Preview (Editable)")
        show_n = st.slider("Rows to show", 5, min(1000, len(cur)), 10)
        edited = st.data_editor(cur.head(show_n), num_rows="dynamic", use_container_width=True, hide_index=True)
        if not edited.equals(cur.head(show_n)):
            st.session_state.processed_data.update(edited)
            st.session_state.data_brief = build_data_brief(st.session_state.processed_data)
            st.success("✅ Edits applied to in-memory data")
        st.subheader("📊 Column Insights")
        info = pd.DataFrame({
            "dtype": cur.dtypes.astype(str), "non_null": cur.count(),
            "null": cur.isnull().sum(), "null_%": (cur.isnull().sum()/len(cur)*100).round(2),
            "unique": cur.nunique(), "unique_%": (cur.nunique()/len(cur)*100).round(2)
        })
        st.dataframe(info, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Data Quality ----------
    with tab2:
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)
        cur = st.session_state.processed_data
        st.subheader("🕳️ Missing Values Analysis")
        miss = cur.isnull().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        if len(miss):
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = px.bar(x=miss.index, y=miss.values, labels={"x": "Columns", "y": "Missing Count"}, title="Missing Values by Column")
                fig.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.dataframe(pd.DataFrame({"Missing": miss, "Missing_%": (miss/len(cur)*100).round(2)}))
        else:
            st.success("✅ No missing values detected")
        st.subheader("🔄 Duplicate Rows")
        dups = cur.duplicated().sum()
        if dups:
            st.warning(f"⚠️ Found {dups} duplicate rows ({dups/len(cur)*100:.2f}%)")
            if st.button("Show Duplicates"):
                st.dataframe(cur[cur.duplicated(keep=False)].sort_values(cur.columns.tolist()), use_container_width=True)
        else:
            st.success("✅ No duplicate rows found")
        st.subheader("🚨 Outlier Analysis")
        num_cols = cur.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            out_counts = detect_outliers_counts(cur, num_cols)
            out_df = pd.DataFrame(out_counts).T
            st.dataframe(out_df, use_container_width=True)
            sel = st.selectbox("Select column for box plot", num_cols)
            if sel:
                fig = px.box(cur, y=sel, title=f"Box Plot - {sel}", points="all")
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ No numeric columns available.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Cleaning ----------
    with tab3:
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)
        wrk = st.session_state.processed_data.copy()
        st.markdown("### 🕳️ Handle Missing Values")
        if wrk.isnull().sum().sum() > 0:
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                miss_method = st.selectbox(
                    "Missing Value Strategy",
                    ["None", "Drop rows", "Drop columns", "Fill mean", "Fill median", "Fill mode", "Forward fill", "Backward fill"]
                )
            with mcol2:
                thresh_pct = st.slider("Drop-column keep-threshold (%)", 0, 100, 50)
            if miss_method != "None" and st.button("Apply Strategy", type="primary"):
                if miss_method == "Drop rows":
                    wrk = wrk.dropna()
                elif miss_method == "Drop columns":
                    thresh = len(wrk) * (100 - thresh_pct)/100
                    wrk = wrk.dropna(axis=1, thresh=thresh)
                elif miss_method == "Fill mean":
                    nc = wrk.select_dtypes(include=np.number).columns
                    wrk[nc] = wrk[nc].fillna(wrk[nc].mean())
                elif miss_method == "Fill median":
                    nc = wrk.select_dtypes(include=np.number).columns
                    wrk[nc] = wrk[nc].fillna(wrk[nc].median())
                elif miss_method == "Fill mode":
                    for c in wrk.columns:
                        wrk[c] = wrk[c].fillna(safe_mode_fill(wrk[c]))
                elif miss_method == "Forward fill":
                    wrk = wrk.fillna(method="ffill")
                elif miss_method == "Backward fill":
                    wrk = wrk.fillna(method="bfill")
                st.session_state.processed_data = wrk
                st.session_state.data_brief = build_data_brief(wrk)
                st.success("✅ Missing-value strategy applied")
                st.rerun()
        else:
            st.success("✅ No missing values to handle")
        st.markdown("### 🚨 Outlier Management")
        num_cols = wrk.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            cols = st.multiselect("Select columns", num_cols)
            method = st.selectbox("Detection Method", ["IQR", "Z-Score", "Modified Z-Score"])
            treat = st.selectbox("Treatment Method", ["Remove", "Cap to bounds", "Replace with median"])
            if cols and st.button("Apply Outlier Treatment", type="primary"):
                for c in cols:
                    s = wrk[c]
                    if method == "IQR":
                        Q1, Q3 = s.quantile(.25), s.quantile(.75)
                        IQR = Q3 - Q1
                        lb, ub = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                        mask = (s < lb) | (s > ub)
                    elif method == "Z-Score":
                        z = np.abs(stats.zscore(s.dropna()))
                        mask = pd.Series(False, index=s.index)
                        mask.loc[s.dropna().index] = z > 3
                        lb, ub = None, None
                    else:
                        med = s.median()
                        mad = np.median(np.abs(s - med))
                        if mad == 0:
                            mask = pd.Series(False, index=s.index)
                            lb, ub = None, None
                        else:
                            mz = 0.6745*(s - med)/mad
                            mask = np.abs(mz) > 3.5
                            lb, ub = None, None
                    if treat == "Remove":
                        wrk = wrk[~mask]
                    elif treat == "Cap to bounds" and method == "IQR":
                        wrk[c] = np.where(wrk[c] < lb, lb, np.where(wrk[c] > ub, ub, wrk[c]))
                    elif treat == "Replace with median":
                        wrk.loc[mask, c] = s.median()
                st.session_state.processed_data = wrk
                st.session_state.data_brief = build_data_brief(wrk)
                st.success("✅ Outliers handled successfully")
                st.rerun()
        else:
            st.info("ℹ️ No numeric columns available.")
        st.markdown("### 🔢 Data Encoding")
        cat_cols = wrk.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            enc_cols = st.multiselect("Select categorical columns", cat_cols)
            enc_method = st.selectbox("Encoding Method", ["Label Encoding", "One-Hot Encoding"])
            if enc_cols and st.button("Apply Encoding", type="primary"):
                if enc_method == "Label Encoding":
                    le = LabelEncoder()
                    for c in enc_cols:
                        wrk[c] = le.fit_transform(wrk[c].astype(str))
                else:
                    wrk = pd.get_dummies(wrk, columns=enc_cols, drop_first=True)
                st.session_state.processed_data = wrk
                st.session_state.data_brief = build_data_brief(wrk)
                st.success("✅ Encoding applied successfully")
                st.rerun()
        else:
            st.info("ℹ️ No categorical columns available.")
        st.markdown("### 📏 Data Scaling")
        num_cols = st.session_state.processed_data.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            sc_cols = st.multiselect("Select columns to scale", num_cols)
            sc_method = st.selectbox("Scaler", ["StandardScaler (Z-score)", "MinMaxScaler (0-1)"])
            if sc_cols and st.button("Apply Scaling", type="primary"):
                scaler = StandardScaler() if sc_method.startswith("Standard") else MinMaxScaler()
                wrk = st.session_state.processed_data.copy()
                wrk[sc_cols] = scaler.fit_transform(wrk[sc_cols])
                st.session_state.processed_data = wrk
                st.session_state.data_brief = build_data_brief(wrk)
                st.success("✅ Scaling applied successfully")
                st.rerun()
        st.markdown("### 📋 Cleaning Summary")
        oc, pc = st.columns(2)
        oc.metric("Original Rows", f"{st.session_state.original_data.shape[0]:,}")
        oc.metric("Original Cols", st.session_state.original_data.shape[1])
        pc.metric("Current Rows", f"{st.session_state.processed_data.shape[0]:,}")
        pc.metric("Current Cols", st.session_state.processed_data.shape[1])
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Visualizations ----------
    with tab4:
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)
        cur = st.session_state.processed_data
        num_cols = cur.select_dtypes(include=np.number).columns.tolist()
        cat_cols = cur.select_dtypes(include=["object", "category"]).columns.tolist()
        kind = st.selectbox("Visualization Type", [
            "Distribution", "Correlation", "Categorical", "Custom", "Pair Plot",
            "Time Series", "Area", "Heatmap", "Strip", "Bubble", "3D Scatter"
        ])
        if kind == "Distribution" and num_cols:
            cols = st.multiselect("Select numeric columns", num_cols, default=num_cols[:min(4, len(num_cols))])
            ptype = st.radio("Plot Type", ["Histogram", "Box", "Violin", "Density"], horizontal=True)
            for c in cols:
                if ptype == "Histogram":
                    fig = px.histogram(cur, x=c, nbins=30, title=f"Distribution: {c}", marginal="rug")
                elif ptype == "Box":
                    fig = px.box(cur, y=c, title=f"Box Plot: {c}", points="all")
                elif ptype == "Violin":
                    fig = px.violin(cur, y=c, title=f"Violin Plot: {c}", box=True)
                else:
                    fig = px.density_contour(cur, x=c, title=f"Density Plot: {c}", marginal="histogram")
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        elif kind == "Correlation" and len(num_cols) > 1:
            corr = cur[num_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="Viridis", title="Correlation Heatmap")
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        elif kind == "Categorical" and cat_cols:
            c = st.selectbox("Select categorical column", cat_cols)
            vc = cur[c].astype(str).value_counts()
            col1, col2 = st.columns(2)
            col1.plotly_chart(px.bar(x=vc.index, y=vc.values, labels={"x": c, "y": "Count"}, title=f"Counts: {c}"), use_container_width=True)
            col2.plotly_chart(px.pie(values=vc.values, names=vc.index, title=f"Share: {c}", hole=0.3), use_container_width=True)
        elif kind == "Pair Plot" and len(num_cols) > 1:
            cols = st.multiselect("Select columns for pair plot", num_cols, default=num_cols[:min(4, len(num_cols))])
            if cols:
                fig = px.scatter_matrix(cur, dimensions=cols, title="Pair Plot")
                fig.update_traces(diagonal_visible=False)
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        elif kind == "Time Series" and num_cols:
            date_col = st.selectbox("Select date/time column", cur.columns)
            value_col = st.selectbox("Select value column", num_cols)
            if date_col and value_col:
                try:
                    cur[date_col] = pd.to_datetime(cur[date_col])
                    fig = px.line(cur, x=date_col, y=value_col, title=f"Time Series: {value_col} over {date_col}")
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.error("⚠️ Invalid date format in selected column.")
        elif kind == "Custom":
            x = st.selectbox("X-axis", cur.columns)
            y = st.selectbox("Y-axis", cur.columns, index=min(1, len(cur.columns)-1))
            color = st.selectbox("Color (optional)", [None] + list(cur.columns))
            size = st.selectbox("Size (optional)", [None] + num_cols)
            p = st.selectbox("Plot Type", ["Scatter", "Line", "Bar"])
            if p == "Scatter":
                fig = px.scatter(cur, x=x, y=y, color=color, size=size, title=f"{y} vs {x}")
            elif p == "Line":
                fig = px.line(cur, x=x, y=y, color=color, title=f"{y} vs {x}")
            else:
                fig = px.bar(cur, x=x, y=y, color=color, title=f"{y} vs {x}")
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        elif kind == "Area" and num_cols:
            cols = st.multiselect("Select numeric columns", num_cols, default=num_cols[:min(2, len(num_cols))])
            stacked = st.checkbox("Stack areas", value=True)
            if cols:
                fig = px.area(cur, x=cur.index, y=cols, title="Area Plot", line_group=None if stacked else "variable")
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        elif kind == "Heatmap" and len(num_cols) > 1:
            x = st.selectbox("X-axis", num_cols)
            y = st.selectbox("Y-axis", num_cols, index=1)
            if x and y:
                fig = px.density_heatmap(cur, x=x, y=y, title=f"Heatmap: {x} vs {y}", marginal_x="histogram", marginal_y="histogram")
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        elif kind == "Strip" and num_cols:
            y = st.selectbox("Y-axis", num_cols)
            color = st.selectbox("Color (optional)", [None] + cat_cols)
            if y:
                fig = px.strip(cur, x=color, y=y, title=f"Strip Plot: {y}", color=color)
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        elif kind == "Bubble" and len(num_cols) > 1:
            x = st.selectbox("X-axis", num_cols)
            y = st.selectbox("Y-axis", num_cols, index=1)
            size = st.selectbox("Size", num_cols, index=min(2, len(num_cols)-1))
            color = st.selectbox("Color (optional)", [None] + list(cur.columns))
            if x and y and size:
                fig = px.scatter(cur, x=x, y=y, size=size, color=color, title=f"Bubble Plot: {y} vs {x}", size_max=30)
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        elif kind == "3D Scatter" and len(num_cols) > 2:
            x = st.selectbox("X-axis", num_cols)
            y = st.selectbox("Y-axis", num_cols, index=1)
            z = st.selectbox("Z-axis", num_cols, index=2)
            color = st.selectbox("Color (optional)", [None] + list(cur.columns))
            if x and y and z:
                fig = px.scatter_3d(cur, x=x, y=y, z=z, color=color, title=f"3D Scatter: {x}, {y}, {z}")
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Advanced Analytics ----------
    with tab5:
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)
        st.subheader("📈 Advanced Analytics")
        num_cols = st.session_state.processed_data.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            analysis_type = st.selectbox("Analysis Type", ["Descriptive Stats", "Skewness & Kurtosis", "PCA Analysis"])
            if analysis_type == "Descriptive Stats":
                cols = st.multiselect("Select columns", num_cols, default=num_cols[:min(4, len(num_cols))])
                if cols:
                    stats_df = st.session_state.processed_data[cols].describe().round(3)
                    st.dataframe(stats_df, use_container_width=True)
            elif analysis_type == "Skewness & Kurtosis":
                cols = st.multiselect("Select columns", num_cols, default=num_cols[:min(4, len(num_cols))])
                if cols:
                    skew_kurt = pd.DataFrame({
                        "Skewness": st.session_state.processed_data[cols].skew(),
                        "Kurtosis": st.session_state.processed_data[cols].kurtosis()
                    }).round(3)
                    st.dataframe(skew_kurt, use_container_width=True)
            elif analysis_type == "PCA Analysis":
                cols = st.multiselect("Select columns for PCA", num_cols, default=num_cols[:min(4, len(num_cols))])
                n_components = st.slider("Number of components", 2, min(5, len(num_cols)), 2)
                if cols and st.button("Run PCA"):
                    pca = PCA(n_components=n_components)
                    pca_data = pca.fit_transform(st.session_state.processed_data[cols].dropna())
                    explained_var = pca.explained_variance_ratio_
                    st.write(f"Explained Variance Ratio: {explained_var.round(3)}")
                    pca_df = pd.DataFrame(pca_data, columns=[f"PC{i+1}" for i in range(n_components)])
                    st.dataframe(pca_df.head(), use_container_width=True)
                    fig = px.scatter(pca_df, x="PC1", y="PC2", title="PCA Scatter Plot")
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ No numeric columns available for advanced analytics.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Export ----------
    with tab6:
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)
        cur = st.session_state.processed_data
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(cur):,}")
        c2.metric("Columns", cur.shape[1])
        c3.metric("Est. Size (MB)", f"{cur.memory_usage(deep=True).sum()/1024**2:.2f}")
        fmt = st.selectbox("Export Format", ["CSV", "Excel", "JSON", "Parquet", "HTML", "PDF"])
        base = st.text_input("File name (no ext.)", f"cleaned_{(st.session_state.last_uploaded_file or 'data').split('.')[0]}")
        if fmt == "CSV":
            data, mime, ext = cur.to_csv(index=False), "text/csv", ".csv"
        elif fmt == "Excel":
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                cur.to_excel(w, index=False, sheet_name="Cleaned")
            data, mime, ext = buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx"
        elif fmt == "JSON":
            data, mime, ext = cur.to_json(orient="records", indent=2), "application/json", ".json"
        elif fmt == "Parquet":
            buf = io.BytesIO()
            cur.to_parquet(buf, index=False)
            data, mime, ext = buf.getvalue(), "application/octet-stream", ".parquet"
        elif fmt == "HTML":
            data, mime, ext = cur.to_html(index=False), "text/html", ".html"
        else:  # PDF
            latex_content = create_latex_document(cur, title="DataPro Exported Dataset")
            data, mime, ext = latex_content, "text/latex", ".tex"
        st.download_button(
            f"📥 Download {fmt}",
            data=data,
            file_name=f"{base}{ext if fmt != 'PDF' else '.pdf'}",
            mime=mime,
            type="primary",
            use_container_width=True
        )
        if fmt == "PDF":
            st.info("ℹ️ The downloaded file is a LaTeX source (.tex). It will be compiled to PDF upon download.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- AI Insights ----------
    with tab7:
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)
        st.subheader("🤖 AI-Powered Data Insights (Gemini)")
        st.caption("Ask questions about your dataset for intelligent insights. Powered by Gemini AI.")
        chat = ensure_gemini()
        if chat is None:
            st.stop()
        with st.expander("📄 Data Brief Sent to Gemini", expanded=False):
            st.code(st.session_state.data_brief, language="text")
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        for role, msg in st.session_state.chat_messages:
            if role == "user":
                st.markdown(f"**You:** {msg}")
            else:
                st.markdown(f"**Gemini:** {msg}")
        q = st.text_input("Ask about your data (e.g., outliers, correlations, trends):", key="gemini_q")
        col_a, col_b = st.columns([1, 1])
        with col_a:
            send = st.button("Ask Gemini", type="primary", use_container_width=True)
        with col_b:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.gemini_chat = None
                st.session_state.chat_messages = []
                chat = ensure_gemini()
                st.rerun()
        if send and q:
            cur = st.session_state.processed_data
            brief = build_data_brief(cur)
            st.session_state.data_brief = brief
            system_prompt = f"""
You are a senior data analyst. Answer based on the user's question and the dataset brief provided below.
- Reference column names exactly.
- If a direct numeric answer isn't possible, suggest app features to use (e.g., 'Check Visualizations > Pair Plot').
- Use concise bullet points and short formulas when helpful.
- Do not invent columns.
DATASET BRIEF:
{brief}
"""
            try:
                resp = chat.send_message([system_prompt, f"USER QUESTION: {q}"])
                answer = resp.text
            except Exception as e:
                answer = f"Error from Gemini: {e}"
            st.session_state.chat_messages.append(("user", q))
            st.session_state.chat_messages.append(("assistant", answer))
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ===================== Empty State =====================
else:
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    st.markdown("""
    ## Welcome to DataPro!
    Upload a CSV or Excel file from the sidebar to start analyzing your data. Try a sample dataset to explore features!
    """)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Load Sample Iris Dataset", type="primary"):
            from sklearn.datasets import load_iris
            iris = load_iris()
            df = pd.DataFrame(iris.data, columns=iris.feature_names)
            df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
            np.random.seed(7)
            df.loc[np.random.choice(df.index, 10), "sepal length (cm)"] = np.nan
            df.loc[np.random.choice(df.index, 5), "petal width (cm)"] = 10.0
            st.session_state.original_data = df.copy()
            st.session_state.processed_data = df.copy()
            st.session_state.last_uploaded_file = "iris_sample.csv"
            st.session_state.data_brief = build_data_brief(df)
            st.success("Sample dataset loaded")
            st.rerun()
    with c2:
        if st.button("Clear All Data"):
            for k in ["processed_data", "original_data", "last_uploaded_file", "data_brief"]:
                st.session_state[k] = None
            st.success("Data cleared")
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)