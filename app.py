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
    if os.path.exists(source_path):
        if not os.path.exists(tmp_path):
            import shutil
            shutil.copytree(source_path, tmp_path)
        use_path = tmp_path
    else:
        use_path = source_path # Fallback
        
    chroma_client = chromadb.PersistentClient(path=use_path)
    # Ensure both collections are initialized
    courses_col = chroma_client.get_or_create_collection("courses")
    knowledge_col = chroma_client.get_or_create_collection("skillsbuild_knowledge")
    print(f"ChromaDB connected. Courses: {courses_col.count()}, Knowledge: {knowledge_col.count()}")
except Exception as e:
    print(f"ChromaDB initialization error: {e}")
    courses_col = None
    knowledge_col = None

import re

def extract_field(text, field_name):
    """Parses Title: or Description: from raw document text."""
    pattern = rf"{field_name}:\s*(.*?)(?:\n|$)"
    match = re.search(pattern, text, re.I)
    return match.group(1).strip() if match else None

# Mapping for robust metadata filtering across collections
CATEGORY_MAP = {
    "student": ["High School Students", "College Students", "high_school_student", "college_student", "student"],
    "educator": ["Educators", "educator"],
    "adult": ["Adult Learners", "adult_learner", "adult"],
    "general": ["General Support", "FAQ", "general_faq", "general"]
}

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
    if not courses_col and not knowledge_col: return []
    
    def perform_query(col, p_val, q_type):
        if not col: return None
        where_filter = None
        
        if q_type == "general":
            where_filter = {"category": {"$in": CATEGORY_MAP["general"]}}
        elif p_val in CATEGORY_MAP:
            where_filter = {"category": {"$in": CATEGORY_MAP[p_val]}}
            
        try:
            return col.query(query_texts=[query], n_results=10, where=where_filter)
        except Exception:
            # Fallback to unfiltered if $in or category field is missing
            return col.query(query_texts=[query], n_results=10)

    try:
        # 1. Search across BOTH collections
        results_courses = perform_query(courses_col, persona, query_type)
        results_knowledge = perform_query(knowledge_col, persona, query_type)
        
        # 2. Semantic Fallback: if total results are zero, search everything with NO filters
        c_docs = results_courses.get("documents", [[]])[0] if results_courses else []
        k_docs = results_knowledge.get("documents", [[]])[0] if results_knowledge else []
        
        if not c_docs and not k_docs and persona != "unknown":
            results_courses = perform_query(courses_col, "unknown", "course")
            results_knowledge = perform_query(knowledge_col, "unknown", "course")

        matches = []
        for res in [results_courses, results_knowledge]:
            if not res or not res.get("documents") or not res["documents"][0]:
                continue
            
            docs = res["documents"][0]
            metas = res["metadatas"][0] if "metadatas" in res and res["metadatas"] else [{}]*len(docs)
            dists = res["distances"][0] if "distances" in res and res["distances"] else [0.5]*len(docs)
            
            for d, m, dist in zip(docs, metas, dists):
                score = 1.0 - (dist if dist is not None else 0.5)
                title = m.get("title") or extract_field(d, "Title") or "SkillsBuild Resource"
                desc = m.get("description") or extract_field(d, "Description") or (d[:150] + "...")
                url = m.get("url") or "#"
                
                if ":" in desc and len(desc) < 50:
                    desc = d.split("\n")[-1] if "\n" in d else d

                matches.append({
                    "title": title,
                    "category": m.get("category") or m.get("persona") or "General",
                    "url": url,
                    "duration": m.get("duration", "Self-paced"),
                    "audience": m.get("audience", "All Learners"),
                    "description": desc.replace('\n', ' ').strip(),
                    "score": score
                })
        
        # Deduplicate and Sort
        unique_matches = []
        seen_urls = set()
        for m in sorted(matches, key=lambda x: x["score"], reverse=True):
            if m["url"] not in seen_urls:
                unique_matches.append(m)
                seen_urls.add(m["url"])
                
        return unique_matches[:4]
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
    
    persona, query_type, is_ambiguous = deduce_persona_and_query(user_query, history)
    
    if persona == "unknown" and query_type != "general":
        return jsonify({
            "reply": "Welcome to SkillsBuild! To help you best, are you a student, educator, or adult learner?",
            "courses": [],
            "persona": "unknown"
        })
        
    is_persona_keyword = user_query.lower() in ["student", "educator", "teacher", "adult", "adult learner"]
    if is_persona_keyword and persona != "unknown":
        role_map = {"educator": "Educator", "student": "Student", "adult": "Adult Learner"}
        return jsonify({
            "reply": f"I have set your role to {role_map.get(persona, persona)}. What subjects are you interested in studying?",
            "courses": [],
            "persona": persona
        })

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
    results = query_chromadb(query, "unknown", "course")
    return jsonify({"courses": results})

@app.route('/api/test-db')
def test_db():
    try:
        c_count = courses_col.count() if courses_col else -1
        k_count = knowledge_col.count() if knowledge_col else -1
        return jsonify({
            "status": "online" if (courses_col or knowledge_col) else "offline",
            "courses_count": c_count,
            "knowledge_count": k_count,
            "tmp_exists": os.path.exists("/tmp/skillsbuild_memory"),
            "mem_exists": os.path.exists(os.path.join(BASE_DIR, "skillsbuild_memory"))
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/search', methods=['POST'])
def legacy_search():
    data = request.json or {}
    query = data.get("query", "")
    courses = query_chromadb(query, "unknown", "course")
    return jsonify({"results": courses})

if __name__ == '__main__':
    app.run(port=5003, debug=True)
