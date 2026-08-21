import re
import asyncio
import chainlit as cl
import os
import pickle
import ollama

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from sentence_transformers import CrossEncoder

DB_PATH = "vectorstore/"
LLM_NAME = "gemma3:12b"
EMBED_MODEL = "nomic-embed-text-v2-moe"
OLLAMA_BASE_URL = "http://localhost:11434"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

RETRIEVER_K_VECTOR = 30
RETRIEVER_K_BM25 = 20
RERANKER_TOP_K = 12
SUB_QUERIES_COUNT = 4

client = ollama.Client(host=OLLAMA_BASE_URL)


def format_docs(docs):
    parts = []
    seen = set()
    for doc in docs:
        content = doc.metadata.get("original_content") or doc.page_content
        content = re.sub(r'^\[Contexto:.*?\]\n\n', '', content, flags=re.DOTALL).strip()

        if content in seen:
            continue
        seen.add(content)

        source = doc.metadata.get("source", "desconhecido")
        page = doc.metadata.get("page", "?")
        parts.append(f"Fonte: {source} | Página: {page}\n{content}")
    return "\n\n---\n\n".join(parts)


def deduplicate_docs(docs):
    seen = set()
    unique = []
    for doc in docs:
        key = doc.metadata.get("original_content") or doc.page_content
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def rerank_docs(reranker: CrossEncoder, query: str, docs: list, top_k: int = RERANKER_TOP_K):
    if not docs:
        return docs
    pairs = [[query, doc.metadata.get("original_content") or doc.page_content] for doc in docs]
    scores = reranker.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


async def generate_sub_queries(llm, question: str) -> list[str]:
    prompt = (
        f"Decomponha a pergunta abaixo em {SUB_QUERIES_COUNT} sub-perguntas simples e independentes "
        f"para busca em documentos administrativos da UFOPA.\n"
        f"Foque em: nomes de pessoas, cargos, siglas de setores e termos institucionais.\n"
        f"Pergunta: {question}\n\n"
        f"Retorne apenas as sub-perguntas, uma por linha, sem numeração ou explicações."
    )
    result = await llm.ainvoke(prompt)
    sub_queries = [q.strip() for q in result.strip().split('\n') if len(q.strip()) > 5]
    return sub_queries[:SUB_QUERIES_COUNT]


print("Inicializando banco de dados e modelos.")

if not os.path.exists(DB_PATH):
    raise Exception(f"Banco não encontrado em {DB_PATH}. Rode ingest.py primeiro.")

embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

vector_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": RETRIEVER_K_VECTOR},
)

bm25_path = os.path.join(DB_PATH, "bm25.pkl")
if not os.path.exists(bm25_path):
    raise Exception("Índice BM25 não encontrado. Rode o ingest.py novamente.")

with open(bm25_path, "rb") as f:
    bm25_retriever = pickle.load(f)
bm25_retriever.k = RETRIEVER_K_BM25

def clean_query(q):
    return re.sub(r'[^\w\s]', ' ', q)

def combine_documents(results):
    return deduplicate_docs(results["bm25"] + results["vetor"])

hybrid_retriever = (
    RunnableParallel({
        "bm25": RunnableLambda(clean_query) | bm25_retriever,
        "vetor": vector_retriever,
    })
    | RunnableLambda(combine_documents)
)

llm = OllamaLLM(model=LLM_NAME, temperature=0, base_url=OLLAMA_BASE_URL, num_ctx=32768)
llm_for_search = OllamaLLM(model=LLM_NAME, temperature=0, base_url=OLLAMA_BASE_URL, num_ctx=4096)

print("Carregando modelo de reranking (BAAI/bge-reranker-v2-m3)...")
reranker = CrossEncoder(RERANKER_MODEL, max_length=512)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um assistente de IA da Universidade Federal do Oeste do Pará (Ufopa), "
     "restrito exclusivamente aos documentos institucionais fornecidos.\n\n"
     "REGRAS ABSOLUTAS — siga-as sem exceção:\n"
     "0. Verifique se sua resposta faz sentido e se ela está de acordo com a pergunta.\n"
     "1. Responda SOMENTE com informações explicitamente presentes no CONTEXTO abaixo.\n"
     "2. NUNCA invente, suponha, complete ou extrapole informações ausentes no contexto.\n"
     "3. Se a pergunta não puder ser respondida com o contexto, responda EXATAMENTE: "
     "'Não encontrei essa informação nos documentos disponíveis.'\n"
     "4. Se apenas parte da resposta estiver no contexto, responda só essa parte e informe que o restante não foi encontrado.\n"
     "5. Para saudações simples (olá, bom dia, etc.), responda de forma breve e educada sem usar o contexto. Pergunte como pode ajudar.\n"
     "6. Não use conhecimento externo, não cite fontes fora do contexto e não faça suposições.\n"
     "7. PRIORIDADE DE FONTE: Quando o mesmo assunto aparecer em múltiplos documentos, priorize sempre "
     "as informações do 'Relatório de Gestão'. Caso ele não contenha a informação, use os demais "
     "documentos (boletins, portarias, resoluções) e indique claramente a fonte utilizada.\n"
     "8. ISOLAMENTO DE ENTIDADES: NUNCA misture informações de pró-reitorias, reitorias ou setores "
     "diferentes. Se um trecho do contexto menciona a PROGEP, essa informação pertence SOMENTE à PROGEP. "
     "Se outro trecho menciona a PROPLAN, pertence SOMENTE à PROPLAN. "
     "Jamais atribua dados, servidores, metas ou atividades de uma unidade a outra. "
     "Em caso de dúvida sobre a qual unidade uma informação pertence, omita-a e informe que não foi possível confirmar.\n\n"
     "CONTEXTO RECUPERADO DOS DOCUMENTOS:\n{context}"),
    ("human", "{question}"),
])

chain = prompt | llm | StrOutputParser()

print("Recursos globais carregados com sucesso!")

@cl.on_chat_start
async def start():
    await cl.Message(
        content="Oi sou um assistente virtual da UFOPA, como posso ajudar?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="Decompondo pergunta e buscando nos documentos...")
    await msg.send()

    sub_queries = await generate_sub_queries(llm_for_search, message.content)
    all_queries = [message.content] + sub_queries

    retrieval_tasks = [hybrid_retriever.ainvoke(q) for q in all_queries]
    results_per_query = await asyncio.gather(*retrieval_tasks)

    all_docs = [doc for docs in results_per_query for doc in docs]
    unique_docs = deduplicate_docs(all_docs)

    if not unique_docs:
        msg.content = "Não encontrei documentos relevantes para sua pergunta."
        await msg.update()
        return

    msg.content = f"Reranqueando {len(unique_docs)} trechos encontrados..."
    await msg.update()

    loop = asyncio.get_event_loop()
    reranked_docs = await loop.run_in_executor(
        None, rerank_docs, reranker, message.content, unique_docs
    )

    context = format_docs(reranked_docs)

    msg.content = "Gerando resposta..."
    await msg.update()

    response = await chain.ainvoke({"context": context, "question": message.content})

    fontes = sorted(set(
        f"{d.metadata.get('source')} (pág {d.metadata.get('page', '?')})"
        for d in reranked_docs
    ))
    response += "\n\n---\n**Fontes consultadas:**\n- " + "\n- ".join(fontes)

    msg.content = response
    await msg.update()
