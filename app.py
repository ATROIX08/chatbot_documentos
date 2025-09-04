# app.py
# Streamlit + Gemma 3 (1B/12B) por API (OpenRouter/Together) + agente JSON tools
# Caso: Onboarding/Inicio de sesión para conductores y pasajeros

import os
import re
import json
import time
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
from PIL import Image
from PyPDF2 import PdfReader

# -------- LLM client (OpenAI SDK apuntando a OpenRouter o Together) ----------
from openai import OpenAI

def get_llm_client_and_model():
    """
    Selecciona proveedor según secretos disponibles.
    - OPENROUTER_API_KEY (recomendado): base_url OpenRouter, model "google/gemma-3-1b-it" o "google/gemma-3-12b-it:free"
    - TOGETHER_API_KEY: base_url Together, model "google/gemma-3-1b-it" (u otro soportado)
    """
    if "OPENROUTER_API_KEY" in st.secrets:
        client = OpenAI(
            api_key=st.secrets["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
        model = st.secrets.get("OPENROUTER_MODEL", "google/gemma-3-1b-it")
        return client, model, "openrouter"
    elif "TOGETHER_API_KEY" in st.secrets:
        client = OpenAI(
            api_key=st.secrets["TOGETHER_API_KEY"],
            base_url="https://api.together.xyz/v1",
        )
        model = st.secrets.get("TOGETHER_MODEL", "google/gemma-3-1b-it")
        return client, model, "together"
    else:
        raise RuntimeError(
            "Configura un secreto OPENROUTER_API_KEY o TOGETHER_API_KEY en Streamlit Cloud."
        )

# ------------------------ Configuración general ------------------------------
st.set_page_config(page_title="Onboarding Docs — Gemma 3 (1B)", page_icon="📄", layout="wide")

DATA_DIR   = Path(".data")
UPLOAD_DIR = Path("uploads")
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

DB_PATH = str(DATA_DIR / "onboarding.db")  # como str para evitar rarezas en algunos entornos

# ------------------------ Base de datos y esquema ----------------------------
@st.cache_resource(show_spinner=False)
def _get_cached_conn() -> sqlite3.Connection:
    """
    Conexión singleton, segura para reruns de Streamlit.
    - check_same_thread=False para evitar ProgrammingError por hilos distintos.
    - PRAGMA para estabilidad/concurrencia básica en demo.
    """
    conn = sqlite3.connect(
        DB_PATH,
        check_same_thread=False,
        timeout=30,
        isolation_level=None,  # autocommit-like
    )
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")
    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute("PRAGMA synchronous = NORMAL;")
    conn.commit()
    bootstrap_db(conn)
    return conn

def get_conn() -> sqlite3.Connection:
    return _get_cached_conn()

def bootstrap_db(conn: sqlite3.Connection):
    cur = conn.cursor()
    # users
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        role TEXT NOT NULL CHECK(role IN ('driver','rider')),
        created_at TEXT NOT NULL
    )""")
    # requirements
    cur.execute("""
    CREATE TABLE IF NOT EXISTS requirements (
        role TEXT NOT NULL,
        doc_type TEXT NOT NULL,
        description TEXT NOT NULL,
        accepted_mimes TEXT NOT NULL,
        max_mb REAL NOT NULL,
        PRIMARY KEY (role, doc_type)
    )""")
    # uploads
    cur.execute("""
    CREATE TABLE IF NOT EXISTS uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        doc_type TEXT NOT NULL,
        filename TEXT NOT NULL,
        mime TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        status TEXT NOT NULL,
        notes TEXT,
        uploaded_at TEXT NOT NULL
    )""")
    # Semilla mínima (solo si tabla vacía)
    rows = cur.execute("SELECT COUNT(*) AS n FROM requirements").fetchone()
    if rows["n"] == 0:
        reqs_driver = [
            ("driver", "id_oficial", "Identificación oficial vigente", "application/pdf,image/jpeg,image/png", 10),
            ("driver", "licencia_conducir", "Licencia de conducir vigente", "application/pdf,image/jpeg,image/png", 10),
            ("driver", "comprobante_domicilio", "Comprobante de domicilio <= 3 meses", "application/pdf,image/jpeg,image/png", 10),
            ("driver", "tarjeta_circulacion", "Tarjeta de circulación", "application/pdf,image/jpeg,image/png", 10),
            ("driver", "poliza_seguro", "Póliza de seguro vigente", "application/pdf,image/jpeg,image/png", 10),
            ("driver", "foto_perfil", "Foto de perfil (rostro completo, 1:1)", "image/jpeg,image/png", 5),
            ("driver", "selfie", "Selfie de verificación (rostro claro)", "image/jpeg,image/png", 5),
        ]
        reqs_rider = [
            ("rider", "id_oficial", "Identificación oficial vigente", "application/pdf,image/jpeg,image/png", 10),
            ("rider", "foto_perfil", "Foto de perfil (rostro completo, 1:1)", "image/jpeg,image/png", 5),
            ("rider", "selfie", "Selfie de verificación (opcional)", "image/jpeg,image/png", 5),
        ]
        cur.executemany("INSERT INTO requirements VALUES (?,?,?,?,?)", reqs_driver + reqs_rider)
    conn.commit()

def upsert_user(user_id: str, role: str):
    user_id = (user_id or "").strip()
    role = role if role in ("driver", "rider") else "driver"
    if not user_id:
        return
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM users WHERE user_id=?", (user_id,))
    if cur.fetchone() is None:
        cur.execute("INSERT INTO users(user_id, role, created_at) VALUES (?,?,datetime('now'))", (user_id, role))
        conn.commit()
    else:
        cur.execute("UPDATE users SET role=? WHERE user_id=?", (role, user_id))
        conn.commit()

def fetch_requirements(role: str) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    rows = cur.execute("""
      SELECT role, doc_type, description, accepted_mimes, max_mb
      FROM requirements WHERE role=? ORDER BY doc_type
    """, (role,)).fetchall()
    return [dict(r) for r in rows]

def get_role(user_id: str) -> str:
    conn = get_conn()
    cur = conn.cursor()
    r = cur.execute("SELECT role FROM users WHERE user_id=?", (user_id,)).fetchone()
    return r["role"] if r else ""

def get_req(role: str, doc_type: str) -> Dict[str, Any]:
    conn = get_conn()
    cur = conn.cursor()
    r = cur.execute("SELECT * FROM requirements WHERE role=? AND doc_type=?", (role, doc_type)).fetchone()
    return dict(r) if r else {}

def insert_upload(user_id: str, doc_type: str, filename: str, mime: str, size_bytes: int, status: str, notes: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
      INSERT INTO uploads(user_id, doc_type, filename, mime, size_bytes, status, notes, uploaded_at)
      VALUES (?,?,?,?,?,?,?,datetime('now'))
    """, (user_id, doc_type, filename, mime, size_bytes, status, notes))
    conn.commit()

def fetch_status_df(user_id: str) -> pd.DataFrame:
    conn = get_conn()
    cur = conn.cursor()
    q = """
    SELECT r.doc_type,
           r.description,
           r.accepted_mimes,
           r.max_mb,
           COALESCE(u.status,'faltante') AS status,
           u.filename,
           u.uploaded_at,
           u.notes
    FROM requirements r
    LEFT JOIN (
        SELECT user_id, doc_type, status, filename, uploaded_at, notes,
               ROW_NUMBER() OVER (PARTITION BY user_id, doc_type ORDER BY uploaded_at DESC) AS rn
        FROM uploads
        WHERE user_id=?
    ) u ON r.doc_type=u.doc_type AND u.rn=1
    WHERE r.role=(SELECT role FROM users WHERE user_id=?)
    ORDER BY r.doc_type
    """
    rows = cur.execute(q, (user_id, user_id)).fetchall()
    return pd.DataFrame([dict(r) for r in rows])

# -------------------------- Validación de archivos ---------------------------
def sanitize(s: str) -> str:
    return re.sub(r"[^-_.a-zA-Z0-9]+", "_", s)

def save_uploaded(user_id: str, doc_type: str, up) -> Path:
    ext = Path(up.name).suffix.lower()
    folder = UPLOAD_DIR / sanitize(user_id)
    folder.mkdir(parents=True, exist_ok=True)
    fname = f"{sanitize(doc_type)}_{int(time.time())}{ext}"
    path = folder / fname
    with open(path, "wb") as f:
        f.write(up.getbuffer())
    return path

def validate_file_basic(path: Path, mime: str, max_mb: float, doc_type: str):
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        return False, f"El archivo excede {max_mb} MB (tiene {size_mb:.2f} MB)."
    try:
        if mime.startswith("image/"):
            with Image.open(path) as im:
                w, h = im.size
                if w < 600 or h < 600:
                    return False, f"Resolución insuficiente ({w}x{h}). Mínimo recomendado 600×600 px."
                if doc_type in ("foto_perfil", "selfie") and abs(w - h) > max(w, h) * 0.2:
                    return False, f"Se recomienda formato 1:1 (cuadrado). Detectado {w}x{h}."
        elif mime == "application/pdf":
            with open(path, "rb") as f:
                reader = PdfReader(f)
                n = len(reader.pages)
                if n < 1:
                    return False, "El PDF no contiene páginas legibles."
                if n > 20:
                    return False, "El PDF tiene demasiadas páginas (>20)."
    except Exception as e:
        return False, f"No fue posible validar el archivo: {e}"
    return True, "OK"

# ----------------------------- Tools del agente ------------------------------
def tool_get_requirements(role: str) -> List[Dict[str, Any]]:
    return fetch_requirements(role)

def tool_get_status(user_id: str) -> List[Dict[str, Any]]:
    df = fetch_status_df(user_id)
    return df.to_dict(orient="records")

def tool_list_formats(role: str, doc_type: str) -> Dict[str, Any]:
    req = get_req(role, doc_type)
    if not req:
        return {"error": f"Documento '{doc_type}' no reconocido para el rol '{role}'."}
    return {
        "doc_type": doc_type,
        "accepted_mimes": req["accepted_mimes"],
        "max_mb": req["max_mb"],
    }

def tool_register_upload(user_id: str, doc_type: str) -> Dict[str, Any]:
    if "__last_upload" not in st.session_state or st.session_state["__last_upload"] is None:
        return {"error": "No hay archivo cargado en la UI. Sube un archivo en el panel lateral y reintenta."}
    up = st.session_state["__last_upload"]
    role = get_role(user_id)
    req = get_req(role, doc_type)
    if not req:
        return {"error": f"'{doc_type}' no es un documento válido para el rol '{role}'."}
    mime = up.type or ""
    accepted = [m.strip() for m in req["accepted_mimes"].split(",")]
    if mime not in accepted:
        return {"error": f"Formato no permitido. Recibido: {mime}. Permitidos: {', '.join(accepted)}."}
    path = save_uploaded(user_id, doc_type, up)
    ok, msg = validate_file_basic(path, mime, float(req["max_mb"]), doc_type)
    status = "pending" if ok else "rejected"
    notes = "Validación básica aprobada." if ok else f"Rechazado: {msg}"
    insert_upload(user_id, doc_type, str(path), mime, path.stat().st_size, status, notes)
    st.session_state["__last_upload"] = None
    return {"doc_type": doc_type, "status": status, "notes": notes, "path": str(path)}

def tool_next_steps(user_id: str) -> str:
    df = fetch_status_df(user_id)
    faltantes = df[df["status"] == "faltante"]["doc_type"].tolist()
    rechazados = df[df["status"] == "rejected"][["doc_type","notes"]].to_dict(orient="records")
    pendientes = df[df["status"] == "pending"]["doc_type"].tolist()
    partes = []
    if faltantes: partes.append("Faltantes: " + ", ".join(faltantes))
    if rechazados:
        rtxt = "; ".join([f"{r['doc_type']} ({r.get('notes','')})" for r in rechazados])
        partes.append("Rechazados: " + rtxt)
    if pendientes: partes.append("En revisión: " + ", ".join(pendientes))
    if not partes:
        partes.append("Todo cargado y validado de forma básica. Espera aprobación final.")
    return " | ".join(partes)

# ------------------------------ Prompt del agente ----------------------------
SYSTEM_PROMPT = """Eres un asistente de onboarding de documentos para una app de transporte.
Ayudas a "driver" y "rider" a saber qué subir, validar archivos, ver estado y próximos pasos.

Debes responder SIEMPRE en el siguiente formato JSON (sin texto extra):
{
  "action": "call_tool" | "final",
  "tool": "get_requirements" | "get_status" | "list_formats" | "register_upload" | "next_steps" | null,
  "args": { ... },     // argumentos si action=call_tool
  "text": "..."        // mensaje final al usuario si action=final
}

Reglas:
- Si preguntan “qué necesito” o similar => call_tool get_requirements(role).
- Si piden avance/faltantes => call_tool get_status(user_id) y/o next_steps(user_id).
- Si piden formatos/tamaños => call_tool list_formats(role, doc_type).
- Si dicen “subí X” o “registra X” => call_tool register_upload(user_id, doc_type).
- Tras recibir resultados de herramientas, produce una respuesta final clara (action="final") con resumen en español, bullets y, cuando aplique, qué falta o cómo corregir.
- No inventes políticas ni tipos de documentos fuera de la lista.
"""

# ------------------------------ Interfaz Streamlit ---------------------------
st.title("Onboarding de Documentos — Gemma 3 (1B)")
st.caption("Asistente para conductores y pasajeros. Despliegue online (OpenRouter/Together).")

with st.sidebar:
    st.subheader("Identidad")
    role = st.selectbox("Rol", options=["driver", "rider"], index=0)
    user_id = st.text_input("Usuario (email/teléfono/ID)", value="demo@example.com")
    if user_id:
        upsert_user(user_id, role)

    st.divider()
    st.subheader("Subir archivo")
    reqs = fetch_requirements(role)
    opt_docs = [r["doc_type"] for r in reqs] if reqs else []
    doc_type_ui = st.selectbox("Tipo de documento", options=opt_docs)
    uploaded = st.file_uploader("Archivo (PDF/JPG/PNG)", type=["pdf","jpg","jpeg","png"], accept_multiple_files=False)
    if uploaded is not None:
        st.session_state["__last_upload"] = uploaded
        st.info(f"Archivo listo: {uploaded.name} ({uploaded.type}, {uploaded.size/1024/1024:.2f} MB)")
    if st.button("Asociar y validar subida"):
        st.session_state.setdefault("messages", [])
        st.session_state["messages"].append(
            {"role":"user","content": f"He subido {doc_type_ui}. Regístralo y valídalo para {user_id}."}
        )

st.divider()
st.subheader("Estado actual")
if user_id:
    df = fetch_status_df(user_id)
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.write("Selecciona un rol y un usuario para ver el estado.")

st.divider()
st.subheader("Chat")

# Historial
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for m in st.session_state["messages"]:
    if m["role"] in ("user","assistant"):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

user_prompt = st.chat_input("Ej: '¿Qué documentos necesito?', '¿Qué me falta?', 'Subí mi licencia'")
if user_prompt:
    st.session_state["messages"].append({"role":"user","content":user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

# ------------------------------ Utilidades LLM -------------------------------
def extract_json(text: str) -> Optional[dict]:
    # Intenta parsear JSON directo o bloque ```json ... ```
    text = text.strip()
    if text.startswith("```"):
        # Remueve fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        # Fallback: busca primer {...}
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None

def _model_supports_system(provider: str, model_name: str) -> bool:
    """
    Algunos proveedores (p.ej., Google AI Studio vía OpenRouter para gemma/gemini)
    no permiten 'developer/system instruction'. En esos casos devolvemos False.
    """
    m = (model_name or "").lower()
    if provider == "openrouter" and (m.startswith("google/") or "gemini" in m):
        # Incluye google/gemma-3-12b-it:free, etc.
        return False
    return True

def _build_api_messages(history: List[Dict[str,str]], system_prompt: str, supports_system: bool) -> List[Dict[str,str]]:
    """
    Si el modelo permite system => usamos rol 'system'.
    Si NO, incrustamos el system_prompt como prefacio del primer mensaje 'user'.
    """
    if supports_system:
        return [{"role": "system", "content": system_prompt}] + history
    else:
        preface = (
            "INSTRUCCIONES DEL SISTEMA (no revelar al usuario):\n"
            + system_prompt +
            "\n\nSigue estrictamente el formato JSON indicado arriba en todas tus respuestas."
        )
        return [{"role": "user", "content": preface}] + history

# ------------------------------ Lazo de agente -------------------------------
def run_agent():
    client, model, provider = get_llm_client_and_model()
    supports_system = _model_supports_system(provider, model)

    # 1) Pedimos al modelo un JSON (posible llamada a tool)
    history = []
    for m in st.session_state["messages"][-8:]:
        # Solo reenviamos roles user/assistant del historial del chat
        if m["role"] in ("user", "assistant"):
            history.append(m)

    msgs = _build_api_messages(history, SYSTEM_PROMPT, supports_system)

    resp = client.chat.completions.create(
        model=model,
        messages=msgs,
        temperature=0.2
    )
    content = resp.choices[0].message.content or ""
    parsed = extract_json(content)

    # Si el modelo no respetó el formato, devolvemos una ayuda y salimos
    if not parsed or "action" not in parsed:
        assistant_text = "Formato no reconocido. Por favor, reformula o intenta de nuevo."
        with st.chat_message("assistant"):
            st.warning(assistant_text)
        st.session_state["messages"].append({"role":"assistant","content":assistant_text})
        return

    if parsed["action"] == "call_tool":
        tool = parsed.get("tool")
        args = parsed.get("args") or {}
        # Ejecutamos herramienta
        if tool == "get_requirements":
            out = tool_get_requirements(args.get("role", role))
        elif tool == "get_status":
            out = tool_get_status(args.get("user_id", user_id))
        elif tool == "list_formats":
            out = tool_list_formats(args.get("role", role), args.get("doc_type", ""))
        elif tool == "register_upload":
            out = tool_register_upload(args.get("user_id", user_id), args.get("doc_type",""))
        elif tool == "next_steps":
            out = tool_next_steps(args.get("user_id", user_id))
        else:
            out = {"error":"herramienta desconocida"}

        # 2) Pasamos el resultado al modelo para redactar la respuesta final (action=final)
        tool_feedback = {
            "tool": tool,
            "args": args,
            "output": out
        }
        followup_history = history + [{
            "role":"user",
            "content": "Resultado de herramienta (formato JSON). Redacta respuesta final:\n```json\n"
                       + json.dumps(tool_feedback, ensure_ascii=False)
                       + "\n```"
        }]
        msgs2 = _build_api_messages(followup_history, SYSTEM_PROMPT, supports_system)

        resp2 = client.chat.completions.create(
            model=model,
            messages=msgs2,
            temperature=0.2
        )
        content2 = resp2.choices[0].message.content or ""
        parsed2 = extract_json(content2)
        if parsed2 and parsed2.get("action") == "final":
            answer = parsed2.get("text","(sin texto)")
        else:
            answer = content2  # si falla, mostramos lo que devolvió

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state["messages"].append({"role":"assistant","content":answer})

    elif parsed["action"] == "final":
        # El modelo respondió directo
        answer = parsed.get("text","(sin texto)")
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state["messages"].append({"role":"assistant","content":answer})

# Ejecutamos si el último mensaje es del usuario
if st.session_state["messages"] and st.session_state["messages"][-1]["role"] == "user":
    try:
        run_agent()
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Error: {e}")
