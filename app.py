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

# Hybrid Path Logic: Try relative path (Vercel) then absolute path (Local)
try:
    rel_mem_path = os.path.join(BASE_DIR, "skillsbuild_memory")
    abs_mem_path = "/Users/yashkhemka/.gemini/antigravity/skillsbuild-agent-portal/skillsbuild_memory"
    
    # Choose whichever exists
    source_path = rel_mem_path if os.path.exists(rel_mem_path) else abs_mem_path
    tmp_path = "/tmp/skillsbuild_memory"
    
    # ChromaDB requires write access even for read-only Ops (SQLite lock files)
    # Vercel filesystem is read-only except for /tmp
    if os.path.exists(source_path):
        if not os.path.exists(tmp_path):
            import shutil
            shutil.copytree(source_path, tmp_path)
        use_path = tmp_path
    else:
        use_path = source_path # Fallback to original, likely will error later if doesn't exist
        
    chroma_client = chromadb.PersistentClient(path=use_path)
    # Ensure collection exists
    collection = chroma_client.get_or_create_collection("skillsbuild_knowledge")
    print(f"ChromaDB connected. Source: {source_path}, Effective: {use_path}")
except Exception as e:
    print(f"ChromaDB initialization error: {e}")
    collection = None

import re

def extract_field(text, field_name):
    """Parses Title: or Description: from raw document text."""
    pattern = rf"{field_name}:\s*(.*?)(?:\n|$)"
    match = re.search(pattern, text, re.I)
    return match.group(1).strip() if match else None

def deduce_persona_and_query(user_query, chat_history):
    user_inputs = [m.get("content", "").lower() for m in chat_history if m.get("role") == "user"]
    text = (user_query.lower() + " " + " ".join(user_inputs))
    
    persona = "unknown"
    if any(k in text for k in ["i am a teacher", "i'm a teacher", "i am an educator", "i'm an educator", "educator"]):
        persona = "educator"
    elif any(k in text for k in ["i am a student", "i'm a student", "student"]):
        persona = "student"
    elif any(k in text for k in ["i am an adult", "i'm an adult", "career change", "professional", "adult"]):
        persona = "adult"
    elif any(k in text for k in ["teacher", "lesson plan", "classroom"]):
        persona = "educator"
    elif any(k in text for k in ["high school", "college", "study"]):
        persona = "student"
    elif any(k in text for k in ["work", "job", "career"]):
        persona = "adult"
        
    query_type = "course"
    if any(k in text for k in ["faq", "terms", "support", "help", "how does"]):
        query_type = "general"
        
    is_ambiguous = persona == "unknown" and query_type != "general"
    return persona, query_type, is_ambiguous

def query_chromadb(query, persona, query_type):
    if not collection: return []
    
    def perform_query(p_val, q_type):
        where_filter = {}
        if q_type == "general":
            where_filter = {"category": "general"}
        elif p_val in ["educator", "adult", "student"]:
            where_filter = {"category": p_val}
        
        try:
            results = collection.query(
                query_texts=[query],
                n_results=10,
                where=where_filter if where_filter else None
            )
            return results
        except:
            return None

    try:
        # 1. Attempt filtered search
        results = perform_query(persona, query_type)
        
        # 2. Semantic Fallback if zero results
        if (not results or not results.get("documents") or not results["documents"][0]) and persona != "unknown":
            results = perform_query("unknown", "course")
            
        matches = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metadatas = results["metadatas"][0] if "metadatas" in results else [{}]*len(docs)
            distances = results["distances"][0] if "distances" in results else [0.5]*len(docs)
            
            for d, m, dist in zip(docs, metadatas, distances):
                score = 1.0 - dist
                
                title = m.get("title") or extract_field(d, "Title") or "Data missing in database"
                desc = m.get("description") or extract_field(d, "Description") or (d[:150] + "...")
                url = m.get("url") or "#"
                
                if ":" in desc and len(desc) < 40:
                    desc = d.split("\n")[-1] if "\n" in d else d

                matches.append({
                    "title": title,
                    "category": m.get("category", "General"),
                    "url": url,
                    "duration": m.get("duration", "Self-paced"),
                    "audience": m.get("audience", "All Learners"),
                    "description": desc.replace('\n', ' ').strip(),
                    "score": score
                })
        
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches[:4]
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
    user_query = data.get("message", "").strip()
    history = data.get("history", [])
    
    # Stage One: Strict Persona Identification
    # If history only contains the initial greeting, and user hasn't identified yet, block.
    # We depend on deduce_persona_and_query to tell us if we know the persona.
    persona, query_type, is_ambiguous = deduce_persona_and_query(user_query, history)
    
    # If persona is still unknown and it's not a general metadata query, block.
    if persona == "unknown" and query_type != "general":
        return jsonify({
            "reply": "Welcome to SkillsBuild! To help you best, are you a student, educator, or adult learner?",
            "courses": [],
            "persona": "unknown"
        })
        
    # Stage Four: Self-Correction Guardrail
    # If they just provided the persona keyword, confirm and ask for interest.
    is_persona_keyword = user_query.lower() in ["student", "educator", "teacher", "adult", "adult learner"]
    if is_persona_keyword and persona != "unknown":
        role_map = {"educator": "Educator", "student": "Student", "adult": "Adult Learner"}
        return jsonify({
            "reply": f"I have set your role to {role_map.get(persona, persona)}. What subjects are you interested in studying?",
            "courses": [],
            "persona": persona
        })

    # Stage Two: Intent Classification & Routing
    # Perform search strictly filtered by USER_PERSONA
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

@app.route('/api/skillsbuild/search')
def sovereign_search():
    query = request.args.get('q', '')
    n = int(request.args.get('n', 6))
    # Catalog search is usually general/wide or we can default to 'general' category
    # For a global search, we pass persona='unknown' which disables the category filter in query_chromadb (if not specified)
    # But wait, query_chromadb logic: if persona not in list, where_filter is empty.
    results = query_chromadb(query, "unknown", "course")
    return jsonify({"courses": results})

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

# Legacy fallback for backwards compat
@app.route('/api/search', methods=['POST'])
def legacy_search():
    data = request.json or {}
    query = data.get("query", "")
    courses = query_chromadb(query, "unknown", "course")
    return jsonify({"results": courses})

if __name__ == '__main__':
    # Increase n_results in query_chromadb to 10 for better fuzzy re-ranking
    app.run(port=5003, debug=True)
