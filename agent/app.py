"""
LangGraph Research Assistant Agent — Groq + TF-IDF RAG
=======================================================
Lightweight and fast: no sentence-transformer model, no GPU, ~5 MB RAM for the index.
Designed for low-resource environments and rapid cold-starts.

Dependencies (requirements.txt):
    langchain-groq
    langchain-community
    langchain-core
    langgraph
    arxiv
    pymupdf          # ArxivLoader PDF backend
    scikit-learn     # TF-IDF
    numpy
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import Annotated, TypedDict, Literal
from operator import add

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
import arxiv as arxiv_pkg
from langchain_core.documents import Document
from langchain_community.document_loaders import ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════
#  TF-IDF Vector Store  (replaces FAISS + HuggingFaceEmbeddings)
# ══════════════════════════════════════════════════════════════

@dataclass
class TFIDFStore:
    """
    Lightweight in-memory retriever backed by scikit-learn TfidfVectorizer.

    Why this works well for ArXiv papers:
    - Authors are consistent with their own terminology: "attention", "diffusion",
      "transformer" etc. appear repeatedly — TF-IDF rewards that.
    - No 90 MB model to load, no warm-up time, cold-starts in milliseconds.
    - The whole index lives in a plain numpy array (~1-2 MB for 50 papers).

    Limitation: won't match synonyms ("LLM" vs "large language model").
    Workaround: the LLM reformulates queries before calling search_papers, so
    it can issue multiple slightly different queries if needed.
    """
    _chunks: list[str] = field(default_factory=list)
    _meta:   list[dict] = field(default_factory=list)
    _vectorizer: TfidfVectorizer | None = field(default=None)
    _matrix: np.ndarray | None = field(default=None)   # shape (n_chunks, vocab)

    def add_documents(self, docs: list[Document]) -> None:
        """Append new chunks and refit the TF-IDF index."""
        for doc in docs:
            self._chunks.append(doc.page_content)
            self._meta.append(doc.metadata)
        self._refit()

    def similarity_search(self, query: str, k: int = 5) -> list[tuple[str, dict, float]]:
        """Return top-k (text, metadata, score) tuples for query."""
        if self._matrix is None or len(self._chunks) == 0:
            return []

        q_vec  = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix)[0]
        top_k  = min(k, len(self._chunks))
        top_idx = np.argsort(scores)[::-1][:top_k]

        return [
            (self._chunks[i], self._meta[i], float(scores[i]))
            for i in top_idx
            if scores[i] > 0.0   # drop zero-overlap chunks
        ]

    @property
    def is_empty(self) -> bool:
        return len(self._chunks) == 0

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def reset(self) -> None:
        self._chunks.clear()
        self._meta.clear()
        self._vectorizer = None
        self._matrix = None

    def _refit(self) -> None:
        """Refit vectorizer on the full corpus. Fast for <= 5000 chunks."""
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),      # unigrams + bigrams ("attention mechanism")
            min_df=1,
            max_df=0.95,             # ignore terms appearing in >95% of chunks
            sublinear_tf=True,       # log(1+tf) — dampens very common terms
            strip_accents="unicode",
            analyzer="word",
        )
        self._matrix = self._vectorizer.fit_transform(self._chunks)


# Module-level singleton shared across all tool calls in one process/session
_store = TFIDFStore()

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " "],
)


# ══════════════════════════════════════════════════════════════
#  Agent State
# ══════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages:      Annotated[list[BaseMessage], add]
    papers_loaded: list[str]   # paper titles ingested this session
    rag_context:   str         # last retrieved context (handy for debugging)


# ══════════════════════════════════════════════════════════════
#  Tools
# ══════════════════════════════════════════════════════════════

@tool
def load_arxiv_paper(arxiv_id_or_query: str) -> str:
    """
    Fetch an ArXiv paper by its ID (e.g. '2310.06825') or by a keyword
    search query (e.g. 'retrieval augmented generation 2023').
    The paper is chunked and indexed into the local TF-IDF store so that
    search_papers can retrieve relevant passages afterward.
    Returns a confirmation with titles and chunk count.
    """
    id_pattern = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
    query = arxiv_id_or_query.strip()
    is_id = bool(id_pattern.match(query))

    search = arxiv_pkg.Search(
        id_list=[query] if is_id else [],
        query=query if not is_id else "",
        max_results=1 if is_id else 3,
    )

    results = None
    for attempt in range(3):
        try:
            results = list(arxiv_pkg.Client().results(search))
            break
        except Exception as exc:
            if "429" in str(exc) and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            return f"ArXiv fetch failed: {exc}"

    if not results:
        return f"No papers found for '{arxiv_id_or_query}'."

    docs = []
    for r in results:
        text = f"{r.title}\n\n{r.summary}"
        docs.append(Document(
            page_content=text,
            metadata={
                "Title": r.title,
                "entry_id": r.entry_id.replace("http://", "https://"),
                "Authors": ", ".join(a.name for a in r.authors),
                "Published": str(r.published.date()),
            }
        ))

    chunks = _splitter.split_documents(docs)
    _store.add_documents(chunks)

    papers_info = [f"{r.title}||{r.entry_id.replace('http://', 'https://')}" for r in results]

    return (
        f"Loaded {len(docs)} paper(s) -> {len(chunks)} chunks indexed. "
        f"Store total: {_store.chunk_count} chunks.\n"
        f"Papers: {'; '.join(papers_info)}"
    )


@tool
def search_papers(query: str) -> str:
    """
    Search the TF-IDF index of previously loaded ArXiv papers.
    Returns the top 5 most relevant text passages with their paper titles
    and relevance scores. Call load_arxiv_paper first if no papers are loaded.

    TF-IDF tip: use the exact terminology the paper authors would use.
    If results are weak, try rephrasing with domain-specific synonyms.
    """
    k = 5
    if _store.is_empty:
        return (
            "The paper index is empty. "
            "Call load_arxiv_paper first to fetch relevant papers."
        )

    results = _store.similarity_search(query, k=k)

    if not results:
        return (
            "No relevant chunks found for that query. "
            "Try different terminology or load additional papers."
        )

    parts = []
    for rank, (text, meta, score) in enumerate(results, 1):
        title = meta.get("Title", "Unknown paper")
        entry_id = meta.get("entry_id", "")
        link = entry_id if entry_id else ""
        pct   = f"{score * 100:.1f}%"
        parts.append(f"[{rank}] {title}  (score {pct}){' | ' + link if link else ''}\n{text.strip()}")

    return "\n\n---\n\n".join(parts)


tools     = [load_arxiv_paper, search_papers]
tool_node = ToolNode(tools)


# ══════════════════════════════════════════════════════════════
#  LLM (Groq)
# ══════════════════════════════════════════════════════════════

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    api_key=os.environ["GROQ_API_KEY"],
)

llm_with_tools = llm.bind_tools(tools, tool_choice="auto", parallel_tool_calls=False)

SYSTEM_PROMPT = """You are an expert AI research assistant specialising in scientific papers.

You have two tools:
  • load_arxiv_paper  — fetch & TF-IDF-index a paper by ArXiv ID or keyword search
  • search_papers     — retrieve relevant passages from the indexed papers

Routing rules
─────────────
1. General knowledge (well-known ML concepts, definitions, algorithm comparisons)
   → answer directly. No tools needed.

2. Specific paper / ArXiv ID / post-training-cutoff results
   → call load_arxiv_paper, then search_papers for supporting passages.

3. Open research questions ("what are recent advances in X")
   → load_arxiv_paper with a descriptive keyword query, then search_papers.

TF-IDF search notes (for your own reasoning)
─────────────────────────────────────────────
• The index matches exact words and two-word phrases. Use domain terminology.
• If the first search returns low scores (<5%), try rephrasing with the author's
  likely wording (e.g. "causal language model" instead of "autoregressive LLM").
• You may call search_papers multiple times with varied queries.

Output style
────────────
• Cite paper titles inline when using retrieved passages.
• Be concise and structured. Use bullet points for multi-part answers.
• If retrieved context is insufficient, say so and suggest loading more papers.
"""


# ══════════════════════════════════════════════════════════════
#  Graph nodes
# ══════════════════════════════════════════════════════════════

def agent_node(state: AgentState) -> dict:
    """Main reasoning node: answer directly or emit tool calls."""
    msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    last_exc = None
    for attempt in range(3):
        try:
            response = llm_with_tools.invoke(msgs)
            return {"messages": [response]}
        except Exception as exc:
            if "tool_use_failed" in str(exc) and attempt < 2:
                time.sleep(1)
                continue
            last_exc = exc
            break
    raise last_exc


def route_after_agent(state: AgentState) -> Literal["tools", "end"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


# ══════════════════════════════════════════════════════════════
#  Compile graph
# ══════════════════════════════════════════════════════════════

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "end": END},
    )
    g.add_edge("tools", "agent")   # tool result loops back to the LLM
    return g.compile()


graph = build_graph()


# ══════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════

def ask(question: str, history: list[BaseMessage] | None = None) -> dict:
    """
    Returns a dict with:
      - answer:  final response string
      - steps:   list of agent actions taken (for display)
      - papers:  list of paper titles loaded this run
    """
    msgs = (history or []) + [HumanMessage(content=question)]
    init_state = AgentState(messages=msgs, papers_loaded=[], rag_context="")

    # retry the full graph invoke on tool_use_failed
    last_exc = None
    result = None
    for attempt in range(3):
        try:
            result = graph.invoke(init_state)
            break
        except Exception as exc:
            if "tool_use_failed" in str(exc) and attempt < 2:
                time.sleep(2)
                continue
            last_exc = exc
            break

    if result is None:
        raise last_exc

    # extract trace from all messages produced during the run
    steps = []
    papers_this_run = []
    loaded_papers = {}       # title -> link, from load_arxiv_paper calls
    searched_titles = set()  # titles that actually appeared in search results
    all_msgs = result["messages"]

    for msg in all_msgs:
        # agent decided to call tools
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                args = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                steps.append(f"[agent] → {tc['name']}({args})")
        # tool result
        elif hasattr(msg, "name") and msg.name in ("load_arxiv_paper", "search_papers"):
            content = msg.content or ""
            preview = content[:120] + ("..." if len(content) > 120 else "")
            steps.append(f"[tool]  {preview}")

            if msg.name == "load_arxiv_paper" and "Papers:" in content:
                for entry in content.split("Papers:")[-1].strip().split(";"):
                    entry = entry.strip()
                    if "||" in entry:
                        title, link = entry.split("||", 1)
                        loaded_papers[title.strip()] = link.strip()

            elif msg.name == "search_papers":
                for line in content.splitlines():
                    if line.startswith("[") and "]" in line:
                        part = line.split("]", 1)[-1].strip()
                        title_part = part.split("(score")[0].strip().rstrip("|").strip()
                        if title_part:
                            searched_titles.add(title_part)

    # only keep papers that were actually loaded (agent already filters relevance)
    for title, link in loaded_papers.items():
        papers_this_run.append({"title": title, "link": link})

    answer = all_msgs[-1].content
    new_messages = all_msgs[len(msgs):]
    return {
        "answer": answer,
        "steps": steps,
        "papers": papers_this_run,
        "new_messages": new_messages
    }




# ══════════════════════════════════════════════════════════════
#  CLI demo
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Research Assistant  |  Groq + LangGraph + TF-IDF RAG")
    print("No model downloads — index lives in RAM (~5 MB for 50 papers)")
    print("Commands: 'clear' resets paper index | 'exit' quits\n")

    history: list[BaseMessage] = []

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        if q.lower() == "clear":
            _store.reset()
            history.clear()
            print("Assistant: Paper index and conversation cleared.\n")
            continue

        history.append(HumanMessage(content=q))
        result = ask(q, history[:-1])

        if result["steps"]:
            print("\n── Agent trace ──────────────────────────")
            for s in result["steps"]:
                print(f"  {s}")
            print("─────────────────────────────────────────")

        if result["papers"]:
            print("\n── Papers loaded ────────────────────────")
            for i, p in enumerate(result["papers"], 1):
                print(f"  {i}. {p}")
            print("─────────────────────────────────────────")

        history.extend(result.get("new_messages", []))
        print(f"\nAssistant: {result['answer']}\n")