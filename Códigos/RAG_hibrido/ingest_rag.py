import os
import re
import shutil
import pickle
from pathlib import Path
import ollama

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

import pymupdf4llm

DATA_PATH = "data/"
DB_PATH = "vectorstore/"
MARKDOWN_PATH = "markdown_output/"
EMBED_MODEL = "nomic-embed-text-v2-moe"
LLM_NAME = "gemma3:12b"
OLLAMA_BASE_URL = "http://localhost:11434"

client = ollama.Client(host=OLLAMA_BASE_URL)

_SKIP_PATTERNS = [
    re.compile(r'Ir para o (conteúdo|menu|busca|rodapé)', re.I),
    re.compile(r'Alto contraste', re.I),
    re.compile(r'Mapa do site', re.I),
    re.compile(r'Selecione o idioma', re.I),
    re.compile(r'Acessar \(/security/', re.I),
    re.compile(r'-{5}\s*(Start|End) of picture text\s*-{5}', re.I),
    re.compile(r'~~\*\*\(HTTPS?://', re.I),
    re.compile(r'^\(https?://', re.I),
]
_BOLETIM_HEADER = re.compile(
    r'ANO\s+[IVXLCDM]+.*?Edição\s+nº\s+\d+.*?BOLETIM\s+DE\s+SERVIÇO',
    re.I | re.S,
)
_LONE_PAGE_NUM = re.compile(r'^\s*\d{1,3}\s*$')
_MULTI_BLANK = re.compile(r'\n{3,}')


def clean_markdown(text: str) -> str:
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        if any(p.search(line) for p in _SKIP_PATTERNS):
            continue
        if _BOLETIM_HEADER.search(line):
            continue
        if _LONE_PAGE_NUM.match(line):
            continue
        cleaned.append(line)
    return _MULTI_BLANK.sub('\n\n', '\n'.join(cleaned)).strip()


def extract_section_headers(text: str) -> str:
    headers = re.findall(r'^#{1,3}\s+.+', text, re.MULTILINE)
    if not headers:
        return ""
    return ' > '.join(h.lstrip('#').strip() for h in headers[-3:])


def generate_chunk_context(chunk_text: str, source: str, section_hint: str) -> str:
    hint = f"\nSeção: {section_hint}" if section_hint else ""
    prompt = (
        f"Documento: {source}{hint}\n\n"
        f"Trecho:\n{chunk_text[:400]}\n\n"
        "Em no máximo 15 palavras, descreva o contexto hierárquico deste trecho "
        "(pró-reitoria, setor, cargo ou tema). Responda apenas a descrição, sem explicações."
    )
    try:
        resp = client.generate(
            model=LLM_NAME,
            prompt=prompt,
            options={"num_ctx": 2048, "temperature": 0},
        )
        return resp["response"].strip()
    except Exception as e:
        print(f"Falha ao gerar contexto: {e}")
        return section_hint


def ingest_docs():
    if os.path.exists(DB_PATH):
        print(f"Limpando banco antigo em {DB_PATH}...")
        shutil.rmtree(DB_PATH)

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Pasta {DATA_PATH} criada. Coloque seus PDFs nela.")
        return

    os.makedirs(MARKDOWN_PATH, exist_ok=True)

    print(f"Extraindo e limpando PDFs de {DATA_PATH}...")
    documents = []

    for file_path in Path(DATA_PATH).glob("**/*.pdf"):
        print(f"\n {file_path.name}")
        try:
            page_docs = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)

            md_clean = "\n\n".join(clean_markdown(p["text"]) for p in page_docs)
            with open(Path(MARKDOWN_PATH) / f"{file_path.stem}.md", "w", encoding="utf-8") as f:
                f.write(md_clean)

            for idx, page_data in enumerate(page_docs):
                cleaned = clean_markdown(page_data.get("text", ""))
                if len(cleaned.strip()) < 50:
                    continue

                page_num = (
                    page_data.get("page")
                    or page_data.get("metadata", {}).get("page")
                    or idx
                )
                documents.append(Document(
                    page_content=cleaned,
                    metadata={
                        "source": file_path.name,
                        "page": int(page_num) + 1,
                        "section_context": extract_section_headers(cleaned),
                    },
                ))
        except Exception as e:
            print(f"Erro ao processar {file_path.name}: {e}")

    if not documents:
        print("Nenhum PDF processado.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n## ", "\n# ", "\n\n", "\n", " "],
    )
    print(f"\nDividindo documentos em chunks...")
    raw_chunks = splitter.split_documents(documents)

    chunks = []
    for doc in raw_chunks:
        cleaned = doc.page_content.replace("\x00", "").strip()
        if len(cleaned) > 50:
            doc.page_content = cleaned
            chunks.append(doc)

    for i, doc in enumerate(chunks):
        doc.metadata["chunk_id"] = i

    print(f"  Páginas: {len(documents)} | Chunks: {len(chunks)}")

    print(f"\nGerando cabeçalhos contextuais para {len(chunks)} chunks")
    contextualized_chunks = []
    for i, doc in enumerate(chunks):
        if i % 20 == 0:
            print(f"  [{i+1}/{len(chunks)}]")
        ctx = generate_chunk_context(
            doc.page_content,
            doc.metadata.get("source", ""),
            doc.metadata.get("section_context", ""),
        )

        contextualized_chunks.append(Document(
            page_content=f"[Contexto: {ctx}]\n\n{doc.page_content}",
            metadata={**doc.metadata, "original_content": doc.page_content},
        ))

 
    print("\nGerando índice BM25...")
    os.makedirs(DB_PATH, exist_ok=True)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    with open(os.path.join(DB_PATH, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25_retriever, f)


    print(f"\nGerando embeddings com {EMBED_MODEL}...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    batch_size = 100
    total_batches = (len(contextualized_chunks) - 1) // batch_size + 1

    print(f"Inserindo lote 1 de {total_batches} no Chroma...")
    vectorstore = Chroma.from_documents(
        documents=contextualized_chunks[:batch_size],
        embedding=embeddings,
        persist_directory=DB_PATH,
    )
    for i in range(batch_size, len(contextualized_chunks), batch_size):
        lote = i // batch_size + 1
        print(f"Inserindo lote {lote} de {total_batches} no Chroma...")
        vectorstore.add_documents(contextualized_chunks[i:i + batch_size])

    print("\nIngestão concluída com sucesso!")


if __name__ == "__main__":
    ingest_docs()
