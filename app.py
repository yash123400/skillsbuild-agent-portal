import os
import json
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import chromadb
try:
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize ChromaDB connection
try:
    mem_path = os.path.join(BASE_DIR, "skillsbuild_memory")
    tmp_path = "/tmp/skillsbuild_memory"
    
    # In Vercel, the filesystem is read-only except for /tmp.
    # ChromaDB's SQLite needs to write lock files/WAL even for reads.
    # Copy to /tmp to allow Chroma to initialize successfully.
    if os.path.exists(mem_path):
        if not os.path.exists(tmp_path):
            import shutil
            shutil.copytree(mem_path, tmp_path)
        use_path = tmp_path
    else:
        use_path = mem_path # fallback
        
    chroma_client = chromadb.PersistentClient(
        path=use_path,
        settings=chromadb.Settings(anonymized_telemetry=False, is_persistent=True)
    )
    collection = chroma_client.get_or_create_collection("skillsbuild_knowledge")
    print(f"ChromaDB connected successfully. Collection count: {collection.count()}")
except Exception as e:
    print(f"ChromaDB initialization error: {e}")
    collection = None

def deduce_persona_and_query(user_query, chat_history):
    # Use only user messages for persona detection to avoid "context poisoning" 
    # from the assistant's initial question ("are you a student, educator...")
    user_inputs = [m.get("content", "").lower() for m in chat_history if m.get("role") == "user"]
    text = (user_query.lower() + " " + " ".join(user_inputs))
    
    # 1. Intent & Persona Detection
    persona = "unknown"
    
    # Check for specific role claims first
    if any(k in text for k in ["i am a teacher", "i'm a teacher", "i am an educator", "i'm an educator", "educator"]):
        persona = "educator"
    elif any(k in text for k in ["i am a student", "i'm a student", "student"]):
        persona = "student"
    elif any(k in text for k in ["i am an adult", "i'm an adult", "career change", "professional", "adult"]):
        persona = "adult"
    # Fallback to loose keyword match if still unknown
    elif any(k in text for k in ["teacher", "lesson plan", "classroom"]):
        persona = "educator"
    elif any(k in text for k in ["high school", "college", "study"]):
        persona = "student"
    elif any(k in text for k in ["work", "job", "career"]):
        persona = "adult"
        
    query_type = "course"
    if any(k in text for k in ["faq", "terms", "support", "help", "how does"]):
        query_type = "general"
        
    # Skepticism Rule
    is_ambiguous = persona == "unknown" and query_type != "general"
    
    return persona, query_type, is_ambiguous

def query_chromadb(query, persona, query_type):
    if not collection: return []
    
    where_filter = {}
    if query_type == "general":
        where_filter = {"category": "general"}
    elif persona in ["educator", "adult", "student"]:
        where_filter = {"category": persona}
        
    try:
        results = collection.query(
            query_texts=[query],
            n_results=4,
            where=where_filter if where_filter else None
        )
        
        matches = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}]*len(docs)
            for d, m in zip(docs, metas):
                # Clean doc parsing if we injected Docling format
                matches.append({
                    "title": m.get("title", "Resource"),
                    "category": m.get("category", "General"),
                    "url": m.get("url", "#"),
                    "duration": m.get("duration", "Self-paced"),
                    "audience": "All Learners" if "default" not in m else m["default"],
                    "description": d[:150].replace('\n', ' ') + "..."
                })
        return matches
    except Exception as e:
        print(f"Query error: {e}")
        return []

@app.route('/')
def catalog():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/chat')
def chat():
    return send_from_directory(BASE_DIR, 'chat.html')

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    data = request.json or {}
    user_query = data.get("message", "")
    history = data.get("history", [])
    
    if not user_query:
        return jsonify({"reply": "I didn't quite catch that.", "courses": [], "persona": "unknown"})
        
    persona, query_type, is_ambiguous = deduce_persona_and_query(user_query, history)
    
    if is_ambiguous:
        return jsonify({
            "reply": "To give you the right resources, are you looking for classroom materials as a teacher, or a course for yourself as a student?",
            "courses": [],
            "persona": persona
        })
        
    # Otherwise, perform the search
    courses = query_chromadb(user_query, persona, query_type)
    
    reply = "I've checked our catalog and found some matches for you!"
    if persona == "educator":
        reply = "I've pulled some specialized lesson plans and toolkits for your classroom."
    elif persona == "adult":
        reply = "Here are some career-focused resources to help you pivot or upskill."
    elif persona == "student":
        reply = "Awesome! I found some great beginner-friendly courses for you."
    elif query_type == "general":
        reply = "Here is some general information and FAQ regarding your question."
        
    if not courses:
        reply = "I searched our catalog but couldn't find a perfect match. Could you rephrase your interest?"
        
    return jsonify({
        "reply": reply,
        "courses": courses,
        "persona": persona
    })

@app.route('/api/test-db')
def test_db():
    try:
        count = collection.count() if collection else -1
        return jsonify({
            "status": "online" if collection else "offline",
            "count": count,
            "tmp_exists": os.path.exists("/tmp/skillsbuild_memory"),
            "mem_exists": os.path.exists(os.path.join(BASE_DIR, "skillsbuild_memory"))
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# Legacy fallback for backwards compat (if any)
@app.route('/api/search', methods=['POST'])
def legacy_search():
    data = request.json or {}
    query = data.get("query", "")
    courses = query_chromadb(query, "unknown", "course")
    return jsonify({"results": courses})

if __name__ == '__main__':
    app.run(port=5003, debug=True)
