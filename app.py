import os, json, pickle, numpy as np
from datetime import datetime, timedelta
import tiktoken
from dotenv import load_dotenv, find_dotenv
import streamlit as st
import faiss
from openai import OpenAI
import re
from numpy.linalg import norm
import redis
from streamlit_js_eval import streamlit_js_eval
import hashlib
from typing import Tuple, Dict, Any

#===================================================================================
# 기본 설정
#===================================================================================
#-----------------------
# Redis & OpenAI 설정
#-----------------------
load_dotenv(find_dotenv(), override=True)

api_key = os.environ.get('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

redis_host = os.environ.get("REDIS_HOST")
redis_port = int(os.environ.get("REDIS_PORT"))
redis_password = os.environ.get("REDIS_PASSWORD")

EMBED_MODEL     = "text-embedding-3-small"
CHAT_MODEL_GENERAL      = "gpt-4.1"
CHAT_MODEL_MINI      = "gpt-4o-mini"
TOP_K           = 4
r = redis.Redis(
    host=redis_host,
    port=redis_port,
    decode_responses=True,
    username="default",
    password=redis_password,
)
# Redis 연결 확인
try:
    r.ping()
except Exception as e:
    st.error(f"❌ Redis 연결 실패: {e}")

#-----------------------------------
# 각 가이드라인별 .index와 .pkl 파일 설정
#-----------------------------------
# 1권
IDX_FILE_1        = "data/book1_faiss_chunk_250804.index"
META_FILE_1       = "data/book1_meta_chunk_250804.pkl"
SECTION_IDX_FILE_1 = "data/book1_faiss_section_keywords_250804.index"
SECTION_META_FILE_1 = "data/book1_meta_section_keywords_250804.pkl"
PAGE_IDX_FILE_1 = "data/book1_faiss_page_250804.index"
PAGE_META_FILE_1 = "data/book1_meta_page_250804.pkl"
# 2권
IDX_FILE_2        = "data/book2_faiss_chunk_250804.index"
META_FILE_2       = "data/book2_meta_chunk_250804.pkl"
SECTION_IDX_FILE_2 = "data/book2_faiss_section_keywords_250804.index"
SECTION_META_FILE_2 = "data/book2_meta_section_keywords_250804.pkl"
PAGE_IDX_FILE_2 = "data/book2_faiss_page_250804.index"
PAGE_META_FILE_2 = "data/book2_meta_page_250804.pkl"
# 3권
IDX_FILE_3        = "data/book3_faiss_chunk_250801.index"
META_FILE_3       = "data/book3_meta_chunk_250801.pkl"
SECTION_IDX_FILE_3 = "data/book3_faiss_section_keywords_250801.index"
SECTION_META_FILE_3 = "data/book3_meta_section_keywords_250801.pkl"
PAGE_IDX_FILE_3 = "data/book3_faiss_page_250801.index"
PAGE_META_FILE_3 = "data/book3_meta_page_250801.pkl"
# 4권
IDX_FILE_4        = "data/book4_faiss_chunk_table_250808.index"
META_FILE_4       = "data/book4_meta_chunk_table_250808.pkl"
SECTION_IDX_FILE_4 = "data/book4_faiss_section_keywords_250808.index"
SECTION_META_FILE_4 = "data/book4_meta_section_keywords_250808.pkl"
PAGE_IDX_FILE_4 = "data/book4_faiss_page_250808.index"
PAGE_META_FILE_4 = "data/book4_meta_page_250808.pkl"

with open(PAGE_META_FILE_1, "rb") as f:
    meta_pages_1 = pickle.load(f)
with open(SECTION_META_FILE_1, "rb") as f:
    meta_keywords_1 = pickle.load(f)
with open(META_FILE_1, "rb") as f:
    meta_chunks_1 = pickle.load(f)

with open(PAGE_META_FILE_2, "rb") as f:
    meta_pages_2 = pickle.load(f)
with open(SECTION_META_FILE_2, "rb") as f:
    meta_keywords_2 = pickle.load(f)
with open(META_FILE_2, "rb") as f:
    meta_chunks_2 = pickle.load(f)

with open(PAGE_META_FILE_3, "rb") as f:
    meta_pages_3 = pickle.load(f)
with open(SECTION_META_FILE_3, "rb") as f:
    meta_keywords_3 = pickle.load(f)
with open(META_FILE_3, "rb") as f:
    meta_chunks_3 = pickle.load(f)

with open(PAGE_META_FILE_4, "rb") as f:
    meta_pages_4 = pickle.load(f)
with open(SECTION_META_FILE_4, "rb") as f:
    meta_keywords_4 = pickle.load(f)
with open(META_FILE_4, "rb") as f:
    meta_chunks_4 = pickle.load(f)

PAGE_VOLUME_LIST = [("1권", meta_pages_1), ("2권", meta_pages_2), ("3권", meta_pages_3), ("4권", meta_pages_4)]
SECTION_VOLUME_LIST = [
    ("1권", meta_keywords_1),
    ("2권", meta_keywords_2),
    ("3권", meta_keywords_3),
    ("4권", meta_keywords_4),
]

#---------------------
# 불용어 & 토큰 수 제한
#---------------------
STOPWORDS = ["알려", "수", "있어", "어디", "나오", "는지", "에서", "으로", "하고", "가이드라인", '확인', '확인하고', '싶어', '페이지', '어느', '부분', '데이터']
TOKEN_LIMIT = 277000

#===================================================================================
# 기본 함수 설정
#===================================================================================
#---------------------
# 임베딩 파일 불러오기
#---------------------
def build_or_load():
    loaded = []
    if os.path.exists(IDX_FILE_1) and os.path.exists(META_FILE_1):
        index_1 = faiss.read_index(IDX_FILE_1)
        with open(META_FILE_1, "rb") as f:
            meta_1 = pickle.load(f)
        loaded.append(("1권", index_1, meta_1))
    if os.path.exists(IDX_FILE_2) and os.path.exists(META_FILE_2):
        index_2 = faiss.read_index(IDX_FILE_2)
        with open(META_FILE_2, "rb") as f:
            meta_2 = pickle.load(f)
        loaded.append(("2권", index_2, meta_2))
    if os.path.exists(IDX_FILE_3) and os.path.exists(META_FILE_3):
        index_3 = faiss.read_index(IDX_FILE_3)
        with open(META_FILE_3, "rb") as f:
            meta_3 = pickle.load(f)
        loaded.append(("2권", index_3, meta_3))
    if os.path.exists(IDX_FILE_4) and os.path.exists(META_FILE_4):
        index_4 = faiss.read_index(IDX_FILE_4)
        with open(META_FILE_4, "rb") as f:
            meta_4 = pickle.load(f)
        loaded.append(("4권", index_4, meta_4))
    if not loaded:
        raise FileNotFoundError("인덱스 파일이 존재하지 않습니다.")
    return loaded

#---------------------
# 단일 텍스트 임베딩
#---------------------
def _embed_text(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [np.array(d.embedding, dtype="float32") for d in resp.data]

#===================================================================================
# 위치 / 코드 / 내용 질문 구분
#===================================================================================
def classify_question(question):
    # 1. 코드/구현 관련 질문인가?
    if is_code_question(question) or is_location_or_code_question_llm(question) == "YES":
        return "code"
    # 2. 위치 관련 질문인가?
    if is_location_question(question) or is_location_or_code_question_llm(question) == "NO":
        return "location"
    # 4. 그 외 (임베딩 검색 등)
    return "other"

#---------------------
# 코드 질문인지 확인
#---------------------
def is_code_question(question):
    # 1차: 단순 키워드 체크
    keywords = ["슈도코드", "코드", "구현"]
    if any(k in question for k in keywords):
        return True
    # 2차: 다양한 표기(띄어쓰기, 영어, 오타 등) 커버
    if is_pseudocode(question):
        return True
    return False

#---------------------
# 위치 질문인지 확인
#---------------------
def is_location_question(question):
    keywords = ["어디", "절", "위치", "나와", "포함", "섹션", "부분", "들어있", "언급", "포함된", "수록"]
    return any(k in question for k in keywords)

#-------------------------------------
# gpt-4o-mini로 위치인지 코드 질문인지 판단
#-------------------------------------
def is_location_or_code_question_llm(question):
    prompt = (
        'If the following question is about code, pseudocode, or implementation, answer YES.'
        'or about the location of content '
        '(such as which section, part, where, included, mentioned, etc.) answer NO.\n\n'
        f'Q: {question}\n'
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=3,
    )
    answer = response.choices[0].message.content.strip().upper()
    return answer == "YES"

#===================================================================================
# 임베딩 관련 함수
#===================================================================================
#-----------------------------------------
# 슈도 코드, 수도코드 -> 슈도코드로 target 지정
#-----------------------------------------
def is_pseudocode(query: str, threshold=0.6) -> str | bool:
    target = "슈도코드"
    # target_vec = get_embedding(target)
    target_vec = get_embedding_cached("슈도코드")  # ✅ 캐시 사용

    # 영어 표현을 한글식으로 치환
    normalized_query = query.lower().replace("pseudo", "슈도").replace("code", "코드")

    candidates = normalized_query.split(" ")
    print(f"$$${candidates}")
    for i in range(len(candidates)):
        for j in range(i + 1, min(len(candidates), i + 2)):
            phrase = " ".join(candidates[i:j+1])

            try:
                # vec = get_embedding(phrase)
                vec = get_embedding_cached(phrase)  # ✅ 캐시 사용

                sim = cosine_similarity(vec, target_vec)
                print(f"유사도({phrase} vs 슈도 코드): {sim:.3f}")
                if sim >= threshold:
                    print(target)
                    return target
            except Exception as e:
                print(f"⚠️ 임베딩 오류: {e}")
                continue

    return False

#---------------------------------------------------------------
# 한 번 임베딩을 계산한 텍스트는 다시 API를 호출하지 않고 캐시에서 불러오는 함수
#---------------------------------------------------------------
cache = {}
def get_embedding_cached(text):
    if text in cache:
        return cache[text]  # 👉 저장된 값 재사용
    emb = get_embedding(text)
    cache[text] = emb       # 👉 결과를 캐시에 저장
    return emb

#---------------------------------------------------
# 어떤 단어가 슈도코드 단어와 유사한지 판별하는 임베딩 기반 함수
#---------------------------------------------------
def is_pseudocode_keyword(word: str, threshold=0.4) -> bool:
    # 임베딩 유사도 기반으로 판별
    target = "슈도코드"
    word = word.lower().replace("pseudo", "슈도").replace("code", "코드")
    print(f"word: {word}")
    target_vec = get_embedding(target)
    word_vec = get_embedding(word)

    sim = cosine_similarity(word_vec, target_vec)
    print(f"sim: {sim}")
    return sim >= threshold

#---------------------
# 단일 텍스트 임베딩 함수
#---------------------
def get_embedding(text):
    return _embed_text([text])[0]

#------------------------------------
# 두 임베딩 벡터 간의 코사인 유사도 계산 함수
#------------------------------------
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

#===================================================================================
# RAG 관련 함수
#===================================================================================
#----------------------------------------
# 여러 권에 걸친 문서 DB에서 관련 문맥을 찾아서,
# 그걸 기반으로 LLM이 답변하도록 하는 함수
#----------------------------------------
def rag_chat_multi_volume(query, history, model=CHAT_MODEL_MINI):
    context_blobs = retrieve_multi_volume(query)
    # 각 블록에 출처 표시
    context_text = "\n\n".join(
        f"[{c['volume']}] {c['text']}" for c in context_blobs
    )

    messages = (
        [{"role": "system",
          "content": "Task: \n"
                     "You are a helpful RAG assistant. Given a user question and context, answer appropriately."

                     "Instructions: \n"
                     "1. Use only the provided context to answer the user's question."
                     "2. For the terms '정밀도 (precision)' and '정밀성 (preciseness)':"
                            "- Do NOT ever confuse or mix up these two terms."
                            "- Each term is a distinct metric with its own unique definition and formula."
                            "- If the question is about '정밀도', only provide the definition and formula for precision."
                            "- If the question is about '정밀성', only provide the definition and formula for preciseness."
                     "3. For any formula or equation mentioned in the document:" 
                            "- Show it **exactly** as it appears in the text."    
                            "- Do not modify, rephrase, or re-typeset."
                            "- Just copy and paste the original LaTeX or expression as-is from the document."

                     "Output Format:\n"
                     "1. Write your answer as a concise, direct response."
                     "2. Keep your response brief and clear."
                     "3. If you cannot answer based on the context, reply exactly: '제가 답변하기 어려운 질문입니다. 연구진에게 문의하세요.'"
          }]
        + history
        + [{"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}]
    )

    return client.chat.completions.create(
        model       = model,
        messages    = messages,
        stream      = True,
        max_tokens  = 512, # 고려
        temperature = 0,
    )

#----------------------------------------
# RAG의 검색(R) 부분
#----------------------------------------
def retrieve_multi_volume(query, top_k=4):
    q_emb = np.array(_embed_text([query])[0]).reshape(1, -1)
    results = []
    for label, index, meta in CHUNKS_VOLUME_LIST:
        D, I = index.search(q_emb, top_k)
        for idx in I[0]:
            chunk = meta[idx]
            # 출처 정보 추가 (권)
            chunk = dict(chunk)
            chunk["volume"] = label
            results.append(chunk)
    return results

#===================================================================================
# 답변하기 어려운 질문이라 답변하는 함수
#===================================================================================
def is_insufficient_answer(answer: str) -> bool:
    return (
        "답변하기 어려운 질문입니다. 연구진에게 문의하세요" in answer
    )

#===================================================================================
# 답변에 사용한 모델명 확인 함수
#===================================================================================
def contains_model_tag(text, model):
    # 모델명 문구가 이미 포함됐는지 체크
    tag = f"{model}로 답변"
    tag2 = f"**{model}**로 답변"
    return (tag in text) or (tag2 in text)

#===================================================================================
# LaTeX 관련 함수
#===================================================================================
#--------------------------------------------------
# 텍스트와 LaTeX 수식을 같이 예쁘게 출력하기 위한 렌더링 함수
#--------------------------------------------------
def display_with_latex(text):
    # 블록 수식 구간 split
    blocks = re.split(r'\\\[(.*?)\\\]', text, flags=re.DOTALL)
    for i, block in enumerate(blocks):
        if i % 2 == 0:
            # 설명문, 인라인 텍스트
            st.write(block)
        else:
            # LaTeX 블록 수식
            enhanced = enhance_korean_fraction(block.strip())
            st.latex(enhanced)

#--------------------------------------------------
# LaTeX 수식에서 한글이 포함된 분수 표현을 예쁘게 가공
#--------------------------------------------------
def enhance_korean_fraction(expr: str) -> str:
    # 한글 문자열을 \text{...}로 감싸기
    def wrap_korean(text: str):
        return re.sub(r"([가-힣]+)", r"\\text{\1}", text)

    pattern = r"\\frac\s*{\s*(.+?)\s*}\s*{\s*(.+?)\s*}"

    def repl(match):
        numerator = wrap_korean(match.group(1)) # 분자
        denominator = wrap_korean(match.group(2)) # 분모

        # displaystyle 및 수직 정렬 추가
        numerator = rf"\rule{{0pt}}{{1em}}{numerator}"
        denominator = rf"{denominator}\rule[-1em]{{0pt}}{{0pt}}"
        return rf"\displaystyle \frac{{{numerator}}}{{{denominator}}}"

    return re.sub(pattern, repl, expr)

#===================================================================================
# 질문이 위치, 코드인지 등을 판단 -> 그에 맞는 검색 함수를 실행하도록 분기
#===================================================================================
def query_by_question_subject_location_pseudo(query, question_subject):
    q_type = clean_phrase(query)

    if question_subject in ("location", "location_or_code"):
        return find_in_pages(q_type)
    elif question_subject == "code" or is_pseudocode(query) == "슈도코드":
        return find_pseudocode_sections(q_type)
    else:
        return None

#--------------------
# 조사를 제거하는 함수
#--------------------
def clean_phrase(phrase):
    # "의", "가", "을", "를" 등의 조사를 모두 제거
    return re.sub(r'(의|가 |을|를|은|는|이 |에|와|과|로|으로|,)', ' ', phrase)

#------------------------------------------------------
# 해당 용어가 등장하는 페이지 목록을 문자열로 만들어 반환하는 함수
#------------------------------------------------------
def find_in_pages(q_type):
    keywords_list = extract_nouns(q_type)
    n = len(keywords_list)
    answer_lines = []
    used_phrases = set()
    shown_phrases = set()  # 이미 표시한 표기(대표 표기, 붙여쓰기/띄어쓰기 모두)

    # 2개 이상 단어면 복합어 우선!
    if n >= 2:
        phrase = " ".join(keywords_list)
        phrase_nospace = phrase.replace(" ", "")
        # 대표 표기는 띄어쓰기 있는 쪽으로!
        found = False
        for cand, display_phrase in [(phrase, phrase), (phrase_nospace, phrase)]:
            if display_phrase in shown_phrases:
                continue
            for label, meta_pages in PAGE_VOLUME_LIST:
                matched_pages = find_pages_with_keywords([cand], meta_pages)
                if matched_pages:
                    answer_lines.append(
                        f'**{display_phrase}**은(는) **{label}** {", ".join(map(str, matched_pages))}쪽(페이지)에 나옵니다.\n'
                    )
                    shown_phrases.add(display_phrase)
                    found = True
        if found:
            return "\n".join(answer_lines)

    # 복합어로 못 찾았을 때만 단일어로 각자 검색
    for k in keywords_list:
        if k in shown_phrases:
            continue
        for label, meta_pages in PAGE_VOLUME_LIST:
            matched_pages = find_pages_with_keywords([k], meta_pages)
            if matched_pages:
                answer_lines.append(
                    f'**{k}**은(는) **{label}** {", ".join(map(str, matched_pages))}쪽(페이지)에 나옵니다.\n'
                )
                shown_phrases.add(k)
    return "\n".join(answer_lines) if answer_lines else "해당 페이지를 찾지 못했습니다."

#-----------------------------------
# 질문에서 한글 명사를 추출하는 전처리 함수
#-----------------------------------
def extract_nouns(text):
    # 기존: 모든 2글자 이상 한글 추출
    words = re.findall(r'[가-힣]{2,}', text)
    # 불용어 제거
    return [w for w in words if w not in STOPWORDS]

#-----------------------------------------------
# 주어진 키워드들이 포함된 페이지를 찾아내는 핵심 검색 함수
#-----------------------------------------------
def find_pages_with_keywords(keywords, meta_pages):
    results = []
    if isinstance(keywords, str):
        keywords = [keywords]

    for page_meta in meta_pages:
        text = page_meta["text"]
        text_nospace = re.sub(r'\s+', '', text)  # 모든 공백류 제거
        if all(
            k.replace(" ", "") in text_nospace
            for k in keywords
        ):
            results.append(page_meta["page"])
    return sorted(set(results))

#---------------------------------------
# 슈도코드가 포함된 절을 찾아주는 검색 전용 함수
#---------------------------------------
def find_pseudocode_sections(q_type):
    keywords = extract_keywords(q_type)
    print(f"keywords: {keywords}")

    concept_keywords = [k for k in keywords if not is_pseudocode_keyword(k)]
    print(f"concept_keywords: {concept_keywords}")

    phrase = " ".join(concept_keywords)
    print(f"phrase: {phrase}")

    matched_sections = []

    for label, meta_keywords in SECTION_VOLUME_LIST:
        for meta_kw in meta_keywords:
            # 복합어(띄어쓰기/붙여쓰기) 모두 검사
            candidates = [phrase]
            if " " in phrase:
                candidates.append(phrase.replace(" ", ""))
            for cand in candidates:
                if any(cand in item for item in meta_kw["keywords"]):
                    if "슈도 코드" in meta_kw["text"] or "슈도코드" in meta_kw["text"]:
                        matched_sections.append((label, meta_kw))

    if matched_sections:
        answers = [
            f"**{label}** {section.get('section', '해당 절')} 절에 슈도코드가 있습니다."
            for label, section in matched_sections
        ]
        return "\n\n".join(answers)
    else:
        return "해당 절에는 슈도코드가 없습니다."

#--------------------------------------------------------
# 사용자 질문에서 gpt-4o-mini를 이용하여 핵심 키워드를 추출하는 함수
#--------------------------------------------------------
def extract_keywords(question):
    prompt = (
        "From the question below, extract all the main subject, keyword, or technical term the user is asking about. "
        "Split all compound words and list every technical term separately, separated by commas. "
        "Do not group multiple terms together. "
        "Do not include any other words or explanation.\n\n"
        f"Question: {question}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=20,
    )
    return [kw.strip() for kw in response.choices[0].message.content.strip().split(',')]

#===================================================================================
# 사용자 식별용(fp) 관련 함수
#===================================================================================
TTL = int(os.getenv("CHAT_TTL_SECONDS", "1800"))  # 기본 1800초 (30분)
CHAT_TTL_SECONDS = TTL

# 선택사항: 서버측 솔트 (환경변수로 설정 권장)
FP_SALT = os.getenv("FP_SALT", "please-change-this-salt")

def get_simple_fingerprint() -> Tuple[str, Dict[str, Any]]:
    """
    브라우저에서 (1) persist_id (localStorage) (2) public IP (ipify) (3) navigator.userAgent
    를 가져와서 간단한 fingerprint를 생성하고 (fp, info) 형태로 반환합니다.

    반환:
      - fp: 24자 길이의 hex 문자열 (빈 문자열이면 실패)
      - info: 수집된 원본 정보 딕셔너리 (pid, ip, ua)
    """
    # JS: persist_id 생성/읽기 + ipify 호출 + userAgent 수집
    js = r"""
    (async () => {
      try {
        // 1) persist_id (localStorage)
        let pid = localStorage.getItem("persist_id");
        if (!pid) {
          // crypto.randomUUID() 지원 안 되면 fallback
          pid = (typeof crypto?.randomUUID === "function") ? crypto.randomUUID() : ('p_' + Math.random().toString(36).slice(2,12));
          localStorage.setItem("persist_id", pid);
        }

        // 2) public IP (ipify)
        let ip = "";
        try {
          const res = await fetch('https://api64.ipify.org?format=json');
          const j = await res.json();
          ip = j?.ip || "";
        } catch(e) {
          ip = "";
        }

        // 3) userAgent
        const ua = navigator.userAgent || "";

        return { pid, ip, ua };
      } catch (e) {
        return { pid: "", ip: "", ua: "" };
      }
    })()
    """

    try:
        info = streamlit_js_eval(js_expressions=js, key="simple_fp_collect", want_output=True) or {}
    except Exception:
        # streamlit_js_eval 호출 실패 시 기본값
        info = {"pid": "", "ip": "", "ua": ""}

    # 보정: 타입 안정성 유지
    pid = str(info.get("pid", "") or "")
    ip  = str(info.get("ip", "") or "")
    ua  = str(info.get("ua", "") or "")

    info_clean: Dict[str, Any] = {"pid": pid, "ip": ip, "ua": ua}

    # fingerprint 원문 (순서 고정) + 서버 솔트 포함
    raw = "|".join([pid, ip, ua, FP_SALT])
    fp = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    # 빈값 체크: pid가 전혀 없고 ip도 없으면 실패로 간주할 수 있음
    if not (pid or ip or ua):
        return "", info_clean

    return fp, info_clean

#-----------------------------------------------------------
# fp별로 하루 동안 사용할 수 있는 토큰 수 제한하는 일일 토큰 쿼터 시스템
#-----------------------------------------------------------
def handle_question(prompt, ip: str):
    # ip가 아직 못 잡혔을 수도 있으니 가드
    if not ip:
        return "IP 확인 중입니다. 잠시 후 다시 시도해주세요."

    key = _today_key_for_ip(ip)
    prompt_tokens = count_tokens(prompt)
    current = int(r.get(key) or 0)

    if current + prompt_tokens > TOKEN_LIMIT:
        return "오늘의 토큰 사용량을 초과했습니다. 내일 다시 질문해주세요."

    # 토큰 증가 + 오늘 자정(UTC)까지 TTL 설정
    ttl = _secs_until_utc_midnight()
    pipe = r.pipeline()
    pipe.incrby(key, prompt_tokens)
    pipe.expire(key, ttl)  # 매 호출마다 갱신해서 누락된 키도 정리
    pipe.execute()
    return "질문 처리 완료"

#---------------------------------------------------
# Redis에 저장할 하루 단위 fp별 토큰 사용량 키를 생성하는 함수
#---------------------------------------------------
def _today_key_for_ip(ip: str) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    safe_ip = _normalize_ip(ip) or "unknown"
    return f"tokens:{safe_ip}:{today}"

#----------------------------------------------
# fp 문자열을 Redis 키로 사용할 수 있도록 정제하는 함수
#----------------------------------------------
def _normalize_ip(ip: str) -> str:
    # Redis 키에 안전하도록 특수문자 정리
    return re.sub(r'[^0-9a-zA-Z\.\-_:]', '_', ip or "")

#----------------------------------------------
# 특정 fp 사용자가 오늘 사용한 토큰 수를 조회하는 함수
#----------------------------------------------
def get_token_usage_for_ip(ip: str):
    if not ip:
        return 0
    key = _today_key_for_ip(ip)
    return int(r.get(key) or 0)

#-------------------------------------------------
# 특정 fp에 대한 과거 대화 기록을 Redis에서 불러오는 함수
#-------------------------------------------------
def load_chat(fp: str, max_messages: int = 100) -> list[dict]:
    """
    fingerprint별 대화 내역을 Redis에서 불러옴.
    """
    key = chat_key_by_fp(fp)
    msgs = r.lrange(key, -max_messages, -1) or []
    return [json.loads(m) for m in msgs]

#------------------------------------------------------
# 사용자 fp를 기반으로 Redis에 저장될 채팅 기록 키를 생성하는 함수
#------------------------------------------------------
def chat_key_by_fp(fp: str) -> str:
    """fingerprint별 Redis 채팅 키"""
    return f"chat:{fp}"

#---------------------------------------
# 사용자의 대화 메시지를 Redis에 저장하는 함수
#---------------------------------------
def append_message(role: str, content: str, fp: str):
    """Redis에 메시지를 추가"""
    key = chat_key_by_fp(fp)
    r.rpush(key, json.dumps({"role": role, "content": content}))
    r.expire(key, CHAT_TTL_SECONDS)

#===================================================================================
# 질문이 몇 개의 토큰으로 구성되어 있는지 계산하는 함수
#===================================================================================
def count_tokens(text):
    try:
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text or ""))

#===================================================================================
# 현재 시각으로부터 UTC 기준 오늘 자정까지 남은 시간을 초 단위로 계산하는 함수
#===================================================================================
def _secs_until_utc_midnight() -> int:
    now = datetime.utcnow()
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return int((tomorrow - now).total_seconds())

#===================================================================================
# Streamlit UI
#===================================================================================
CHUNKS_VOLUME_LIST = build_or_load()

st.set_page_config(page_title="chatbot")
st.image("img/logo.png", width=170)
st.error("이 챗봇은 참고용으로 제공되며, 중요한 내용은 반드시 공식 가이드라인을 확인하세요.")

# fingerprint 생성 (IP + UA + persist_id)
fp, info = get_simple_fingerprint()   # ← 기존 get_client_ip() 대신 사용

if fp:
    st.caption(f"ID: {fp[-6:]}")
    st.session_state["fingerprint"] = fp
    st.session_state["client_info"] = info
else:
    st.caption("현재 ID 생성 중…(브라우저 정보 확인)")
    st.stop()

# 과거 대화 전체 출력 (Redis에서 로드)
for h in load_chat(fp):
    with st.chat_message(h["role"]):
        display_with_latex(h["content"])

# 새 질문 입력받기
if prompt := st.chat_input("가이드라인에 대해 질문하세요…"):
    # 토큰(IP 기준)
    result = handle_question(prompt, fp)
    usage = get_token_usage_for_ip(fp)
    st.markdown(f"오늘 사용한 토큰 수: **{usage} / {TOKEN_LIMIT}**")

    # 3. user 질문 즉시 출력
    st.chat_message("user").markdown(prompt)
    append_message("user", prompt, fp)

    with st.chat_message("assistant"):
        # 1) 인디케이터 표시
        typing_box = st.empty()
        typing_box.markdown("""
        <style>
        .typing {font-size: 0.95rem; color: #6b7280;}
        .typing .dot {animation: blink 1.2s infinite;}
        .typing .dot:nth-child(2){animation-delay:0.2s}
        .typing .dot:nth-child(3){animation-delay:0.4s}
        @keyframes blink {0%{opacity:.2} 20%{opacity:1} 100%{opacity:.2}}
        </style>
        <div class="typing">답변 생성 중 <span class="dot">•</span><span class="dot">•</span><span class="dot">•</span></div>
        """, unsafe_allow_html=True)

        if ("토큰 사용량을 초과" in result) or ("IP 확인 중" in result):
            typing_box.empty()  # ← 종료 시 반드시 지워주기
            st.markdown(f"{result}")
            st.stop()

        question = classify_question(prompt)
        print(f"1) question: {question}")

        if question == "other":
            full_response = ""
            for chunk in rag_chat_multi_volume(prompt, load_chat(fp)):
                delta = chunk.choices[0].delta.content or ""
                full_response += delta

            if is_insufficient_answer(full_response):
                full_response = ""
                for chunk in rag_chat_multi_volume(prompt, load_chat(fp), model=CHAT_MODEL_GENERAL):
                    delta = chunk.choices[0].delta.content or ""
                    full_response += delta
                if not contains_model_tag(full_response, CHAT_MODEL_GENERAL):
                    full_response += f"\n\n**{CHAT_MODEL_GENERAL}**로 답변"
            else:
                if not contains_model_tag(full_response, CHAT_MODEL_MINI):
                    full_response += f"\n\n**{CHAT_MODEL_MINI}**로 답변"

            typing_box.empty()  # ← 답변 출력 직전에 제거
            display_with_latex(full_response)
            append_message("assistant", full_response, fp)
            st.stop()

        # 위치/슈도코드 경로
        answer = query_by_question_subject_location_pseudo(prompt, question)
        typing_box.empty()
        print(f"2) answer: {answer}")
        if answer:
            display_with_latex(answer)
            append_message("assistant", answer, fp)
            st.stop()
