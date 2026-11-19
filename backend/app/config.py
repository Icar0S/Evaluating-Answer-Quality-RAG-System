"""Configuração centralizada da aplicação, carregada de variáveis de ambiente/.env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Raiz do projeto (dois níveis acima de backend/app/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Versões do índice vetorial do RAG. Cada versão é um acervo (uma pasta de
# documentos) indexado numa coleção própria do Chroma; RAG_INDEX escolhe qual
# o assistente usa. pdf-v1 é o índice histórico (a mesma coleção de sempre).
# Mudar o acervo, o formato ou a extração de texto de uma versão = criar uma
# versão nova, não editar a existente: resultados de avaliação citam a versão
# pelo nome, e o manifesto de cada ingestão (app/rag/ingestion.py) registra o
# que entrou nela.
# O modelo de embedding faz parte da versão: um índice só pode ser consultado
# com o modelo que o construiu (vetores de modelos diferentes não se comparam,
# e a busca erra sem dar erro). Por isso RAG_INDEX fixa também EMBEDDING_MODEL.
#
# Versões diferentes podem ler a mesma pasta (pdf-v1 e pdf-v2 leem
# data/source_pdfs). Quem diz o que entra em cada uma é a lista de documentos
# dela, com o SHA-256 de cada arquivo, em app/rag/index_versions/<versão>.sha256
# (scripts/ingest_documents.py --register). Arquivo da pasta fora da lista fica
# de fora.
#
# "origins": de quais acervos de perguntas de referência (tests/deepeval/goldens:
# "pdf" = dataset.json, "md" = dataset_md.json) a versão contém os artigos. Uma
# pergunta de origem que a versão não contém mede se o RAG se abstém.
#
# Versão derivada (os mesmos documentos de outra versão e uma mudança só, como
# nos braços do plano-benchmark.md): a lista sai de `--register --like <base>`,
# e não da pasta, que pode ter documentos de outras versões. As chaves de
# INDEX_SETTINGS_KEYS são opcionais: presentes, fixam o parâmetro de chunking da
# versão por cima do .env, como o modelo de embedding.
INDEX_SETTINGS_KEYS = (
    "chunk_size_tokens", "chunk_overlap_tokens", "chunk_across_pages", "chunker", "strip_references", "chunk_header",
)

RAG_INDICES: dict[str, dict[str, Any]] = {
    "pdf-v1": {
        "collection": "rag_test_specialist",
        "source_dir": "data/source_pdfs",
        "embedding_model": "nomic-embed-text",
        "origins": ("pdf",),
        "description": "os 39 artigos originais em PDF; texto do PyMuPDF, chunks dentro de cada página",
    },
    "md-v1": {
        "collection": "rag_md_v1",
        "source_dir": "data/new_papers_marks",
        "embedding_model": "nomic-embed-text",
        "origins": ("md",),
        "description": "os 22 artigos originais em Markdown; sintaxe de tabela convertida em texto, documento inteiro "
        "em chunks",
    },
    # Os mesmos 61 artigos nos dois formatos (data/sync_corpora.py, 17/10/2026): comparar pdf-v2 e md-v2 é
    # comparar formatos, não acervos.
    "pdf-v2": {
        "collection": "rag_pdf_v2",
        "source_dir": "data/source_pdfs",
        "embedding_model": "nomic-embed-text",
        "origins": ("pdf", "md"),
        "description": "os 61 artigos em PDF: os 39 do pdf-v1 e os 22 do md-v1 diagramados a partir do Markdown",
    },
    "md-v2": {
        "collection": "rag_md_v2",
        "source_dir": "data/new_papers_marks",
        "embedding_model": "nomic-embed-text",
        "origins": ("pdf", "md"),
        "description": "os 61 artigos em Markdown: os 22 do md-v1 e os 39 do pdf-v1 convertidos pelo pymupdf4llm",
    },
    # Braços do benchmark (plano-benchmark.md): versões derivadas, com a lista da base (--register --like).
    "pdf-v1-aa": {
        "collection": "rag_pdf_v1_aa",
        "source_dir": "data/source_pdfs",
        "embedding_model": "nomic-embed-text",
        "origins": ("pdf",),
        "description": "réplica do pdf-v1 (mesma lista, mesmo modelo, coleção nova) para o ruído da reingestão no E0",
    },
    # Com RAG_INDEX definido, o validador sobrescreve EMBEDDING_MODEL: trocar de embedding é trocar de versão.
    "pdf-v1-bgem3": {
        "collection": "rag_pdf_v1_bgem3",
        "source_dir": "data/source_pdfs",
        "embedding_model": "bge-m3",
        "origins": ("pdf",),
        "description": "o pdf-v1 (mesma lista, mesmo chunking) com o embedding multilíngue bge-m3, para o E1",
    },
    # E6: a janela atravessa a página. No pdf-v1, que corta dentro da página, "chunk 1200" ficaria perto de
    # "um chunk por página"; por isso os braços de chunk têm como baseline o pdf-v1-cont, com 800/120.
    "pdf-v1-cont": {
        "collection": "rag_pdf_v1_cont",
        "source_dir": "data/source_pdfs",
        "embedding_model": "nomic-embed-text",
        "origins": ("pdf",),
        "chunk_size_tokens": 800,
        "chunk_overlap_tokens": 120,
        "chunk_across_pages": True,
        "description": "o pdf-v1 com a janela de 800/120 atravessando a página (baseline dos braços de chunk do E6)",
    },
    "pdf-cont-400": {
        "collection": "rag_pdf_cont_400",
        "source_dir": "data/source_pdfs",
        "embedding_model": "nomic-embed-text",
        "origins": ("pdf",),
        "chunk_size_tokens": 400,
        "chunk_overlap_tokens": 120,
        "chunk_across_pages": True,
        "description": "o pdf-v1-cont com chunk de 400 tokens (E6)",
    },
    "pdf-cont-1200": {
        "collection": "rag_pdf_cont_1200",
        "source_dir": "data/source_pdfs",
        "embedding_model": "nomic-embed-text",
        "origins": ("pdf",),
        "chunk_size_tokens": 1200,
        "chunk_overlap_tokens": 120,
        "chunk_across_pages": True,
        "description": "o pdf-v1-cont com chunk de 1200 tokens (E6)",
    },
    "pdf-cont-o0": {
        "collection": "rag_pdf_cont_o0",
        "source_dir": "data/source_pdfs",
        "embedding_model": "nomic-embed-text",
        "origins": ("pdf",),
        "chunk_size_tokens": 800,
        "chunk_overlap_tokens": 0,
        "chunk_across_pages": True,
        "description": "o pdf-v1-cont sem sobreposição (E6)",
    },
    # E2: os 39 artigos do pdf-v1 em Markdown (--register --like pdf-v1 --by-stem). A conversão pymupdf4llm é a
    # que o md-v2 já trouxe para data/new_papers_marks; o Markdown já corta o documento inteiro, como o pdf-v1-cont.
    "md-p4llm": {
        "collection": "rag_md_p4llm",
        "source_dir": "data/new_papers_marks",
        "embedding_model": "nomic-embed-text",
        "origins": ("pdf",),
        "chunk_size_tokens": 800,
        "chunk_overlap_tokens": 120,
        "description": "os 39 artigos do pdf-v1 convertidos em Markdown pelo pymupdf4llm (degrau 2 do E2)",
    },
    "md-dl": {
        "collection": "rag_md_dl",
        "source_dir": "data/md_dl",
        "embedding_model": "nomic-embed-text",
        "origins": ("pdf",),
        "chunk_size_tokens": 800,
        "chunk_overlap_tokens": 120,
        "description": "os 39 artigos do pdf-v1 convertidos em Markdown pelo Docling (degrau 2 do E2, "
        "data/convert_docling.py)",
    },
    # Degrau 3 do E2: só o chunker muda (corte nos títulos, a janela dentro de cada seção). As duas ficam
    # prontas; a seleção do E2 diz qual vai à confirmação.
    "md-p4llm-sec": {
        "collection": "rag_md_p4llm_sec",
        "source_dir": "data/new_papers_marks",
        "embedding_model": "nomic-embed-text",
        "origins": ("pdf",),
        "chunk_size_tokens": 800,
        "chunk_overlap_tokens": 120,
        "chunker": "section",
        "description": "o md-p4llm com o chunker por seção (degrau 3 do E2)",
    },
    "md-dl-sec": {
        "collection": "rag_md_dl_sec",
        "source_dir": "data/md_dl",
        "embedding_model": "nomic-embed-text",
        "origins": ("pdf",),
        "chunk_size_tokens": 800,
        "chunk_overlap_tokens": 120,
        "chunker": "section",
        "description": "o md-dl com o chunker por seção (degrau 3 do E2)",
    },
    # H5b do E5: a bibliografia é ímã? O bloco cortado é o que a C0 acha (corpus_quality.references_block).
    "pdf-v1-noref": {
        "collection": "rag_pdf_v1_noref",
        "source_dir": "data/source_pdfs",
        "embedding_model": "nomic-embed-text",
        "origins": ("pdf",),
        "strip_references": True,
        "description": "o pdf-v1 sem o bloco de referências de cada artigo (H5b do E5)",
    },
    # E3: o cabeçalho "Artigo: <título> · pág. N" (backend/app/rag/titles.json) só no texto embedado, ou no
    # próprio trecho (embedding, prompt e juízes).
    "pdf-hdr-emb": {
        "collection": "rag_pdf_hdr_emb",
        "source_dir": "data/source_pdfs",
        "embedding_model": "nomic-embed-text",
        "origins": ("pdf",),
        "chunk_header": "embedding",
        "description": "o pdf-v1 com o cabeçalho do artigo só no texto embedado (E3)",
    },
    "pdf-hdr": {
        "collection": "rag_pdf_hdr",
        "source_dir": "data/source_pdfs",
        "embedding_model": "nomic-embed-text",
        "origins": ("pdf",),
        "chunk_header": "text",
        "description": "o pdf-v1 com o cabeçalho do artigo no texto do trecho (E3)",
    },
    # O E3 com o embedding vencedor do E1: com o nomic, 56% das perguntas de dentro em português recebem
    # "não encontrei", e sobra pouca citação para medir a atribuição errada.
    "bgem3-hdr-emb": {
        "collection": "rag_bgem3_hdr_emb",
        "source_dir": "data/source_pdfs",
        "embedding_model": "bge-m3",
        "origins": ("pdf",),
        "chunk_header": "embedding",
        "description": "o pdf-v1-bgem3 com o cabeçalho do artigo só no texto embedado (E3)",
    },
    "bgem3-hdr": {
        "collection": "rag_bgem3_hdr",
        "source_dir": "data/source_pdfs",
        "embedding_model": "bge-m3",
        "origins": ("pdf",),
        "chunk_header": "text",
        "description": "o pdf-v1-bgem3 com o cabeçalho do artigo no texto do trecho (E3)",
    },
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    generation_model: str = "qwen3:8b"
    embedding_model: str = "nomic-embed-text"
    generation_timeout_seconds: int = 120

    # Provider remoto (opcional — API própria do Mac mini, ver llm-api-referencia.md).
    # NÃO é Ollama-compatível: /v1/chat com Bearer, sem rota de embeddings (por
    # isso não há remote_embedding_model — embeddings são sempre locais).
    remote_api_base_url: str | None = None
    remote_api_key: str | None = None
    remote_generation_model: str | None = None
    remote_label: str = "Servidor (Mac mini)"

    # Provider ativo ao iniciar o backend (local | remote); trocável em runtime via /providers/active
    active_provider: str = "local"

    # Chunking
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 120
    # Desloca todas as fronteiras de chunk em N tokens (o primeiro chunk da
    # pagina fica menor, os seguintes ficam alinhados N tokens adiante). 0 = sem
    # deslocamento, que e o comportamento historico. Existe para o operador de
    # mutacao K4 (tests/mutation/), que injeta "corte no meio de regra".
    chunk_boundary_offset_tokens: int = 0
    # PDF: a janela de tokens atravessa a pagina (as paginas viram um texto so,
    # e cada chunk leva a pagina onde comeca). False = chunks dentro de cada
    # pagina, o comportamento historico. Os indices pdf-v1-cont e pdf-cont-* do
    # E6 (plano-benchmark.md) ligam pela versao.
    chunk_across_pages: bool = False
    # Markdown: "token" corta o documento inteiro em janelas (historico);
    # "section" corta nos titulos "#" e corre a janela dentro de cada secao
    # (degrau 3 do E2). No PDF nao muda nada: o texto do PyMuPDF nao tem titulos.
    chunker: Literal["token", "section"] = "token"
    # Corta o bloco de referências de cada documento antes do chunking (o que a
    # C0 acha). False = historico. So a versao pdf-v1-noref (H5b do E5) liga.
    strip_references: bool = False
    # Cabecalho contextual "Artigo: <titulo> · pag. N" (app/rag/titles.py; E3 do
    # plano-benchmark.md): "embedding" poe o cabecalho so no texto embedado (o
    # guardado, o prompt e os juizes nao mudam); "text", no proprio trecho.
    # "none" = historico.
    chunk_header: Literal["none", "embedding", "text"] = "none"

    # Retrieval
    top_k: int = 4
    # Piso de similaridade (0..1) para um chunk entrar no contexto. 0.0 = sem
    # filtro. Acima de 0, uma pergunta fora do corpus tende a chegar na geracao
    # com contexto vazio, o que dispara a frase de abstencao em vez de uma
    # resposta apoiada em trecho irrelevante.
    min_similarity_score: float = 0.0
    # MMR (Maximal Marginal Relevance) na selecao final dos chunks: busca
    # top_k * retrieval_fetch_multiplier candidatos e escolhe top_k balanceando
    # relevancia e diversidade. Desligado por padrao para preservar o
    # comportamento historico do assistente; a configuracao do estudo de
    # mutacao (tests/mutation/config/study.yaml) liga.
    retrieval_use_mmr: bool = False
    retrieval_mmr_lambda: float = 0.7
    retrieval_fetch_multiplier: int = 3
    # Busca hibrida (app/rag/bm25.py; E7 do plano-benchmark.md): a vetorial e a
    # lexical (BM25) fundidas por RRF, com o k do RRF abaixo (60, o da
    # literatura). Desligada por padrao (comportamento historico).
    retrieval_hybrid: bool = False
    retrieval_rrf_k: int = 60
    # Reranker com o LLM local (app/rag/rerank.py; E8 do plano-benchmark.md): o
    # LLM da nota de 0 a 10 a cada candidato, e os top_k de maior nota seguem.
    # Modelo vazio = GENERATION_MODEL. Desligado por padrao (historico).
    retrieval_rerank: bool = False
    retrieval_rerank_model: str | None = None
    retrieval_rerank_candidates: int = 20
    # Traducao da pergunta para o ingles, so na busca (app/rag/translation.py):
    # o acervo e quase todo em ingles. O gerador continua recebendo a pergunta
    # original. Desligada por padrao (comportamento historico); o E1 do
    # plano-benchmark.md liga. Modelo vazio = GENERATION_MODEL, no Ollama local.
    # O cache e um JSON {pergunta: traducao}, relativo a raiz do projeto, gravado
    # uma vez por tests/benchmark/translate_questions.py.
    retrieval_query_translation: bool = False
    retrieval_translation_model: str | None = None
    retrieval_translation_cache: str | None = None

    # Paths (relativos à raiz do projeto)
    source_pdfs_dir: str = "data/source_pdfs"
    vector_store_dir: str = "data/vector_store"
    vector_store_collection: str = "rag_test_specialist"
    logs_dir: str = "logs"
    # Versão do índice (RAG_INDICES). Definida, sobrepõe VECTOR_STORE_COLLECTION,
    # SOURCE_PDFS_DIR e EMBEDDING_MODEL com os valores da versão. Vazia =
    # comportamento histórico.
    rag_index: str | None = None

    # Geracao
    # None = nao envia `options` ao Ollama (mantem o default do modelo, que e o
    # comportamento historico). Fixar em 0.0 nao torna a geracao deterministica,
    # mas evita amplificar a variancia -- e o que o protocolo do estudo pede.
    generation_temperature: float | None = None
    generation_seed: int | None = None
    # Janela de contexto e teto de geracao enviados ao Ollama. None = nao envia
    # (comportamento historico). Descoberto no estudo de mutacao (21/09/2026):
    # sem `num_ctx` o servidor usa 4096 e TRUNCA o prompt em num_ctx/2 tokens,
    # descartando o inicio -- o system prompt inteiro e os primeiros chunks.
    # 288 truncagens silenciosas no baseline antes de alguem olhar o log do
    # servidor. "auto" dimensiona pela configuracao de recuperacao:
    #   num_ctx = ceil_1024(top_k * chunk_size_tokens * 1.25 + 1024 + num_predict)
    # O limite efetivo de prompt no Ollama e num_ctx - num_predict, por isso
    # `num_predict` precisa ser explicito quando `num_ctx` e.
    generation_num_ctx: int | str | None = None
    generation_num_predict: int | None = None
    # Campo `think` do /api/chat do Ollama. None = nao envia (comportamento
    # historico: modelos de raciocinio como o qwen3 pensam antes de responder).
    # False corta esse raciocinio -- medido em 07/10/2026 com o prompt real e o
    # qwen3:8b: 727 -> 224 tokens gerados, 54s -> 10s. Em maquina sem GPU e o
    # que separa responder de estourar GENERATION_TIMEOUT_SECONDS.
    generation_think: bool | None = None

    # Prompt: cada bloco de regra do system prompt e ligavel/desligavel, porque
    # os operadores de mutacao P1-P3 removem exatamente um bloco cada.
    # Os defaults reproduzem o prompt historico (v1).
    prompt_require_grounding: bool = True
    prompt_require_abstention: bool = True
    prompt_require_citation: bool = False
    # Redacao da regra de citacao: "v1" (a de sempre, que o estudo de mutacao
    # le literalmente) ou "v2" (cita [arquivo] quando o trecho nao tem pagina;
    # E3 do plano-benchmark.md). So vale com PROMPT_REQUIRE_CITATION.
    prompt_citation_rule: Literal["v1", "v2"] = "v1"

    # Prompt versioning
    prompt_version: str = "v1"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8010
    cors_origins: str = "http://localhost:5510,http://127.0.0.1:5510,http://localhost:8010"

    # O .env.example traz estes campos vazios ("GENERATION_SEED=") para documentar
    # que existem; vazio significa "nao definido", nao string invalida. Sem isto,
    # `copy .env.example .env` derrubava o backend no boot com erro de validacao.
    # (Nao da para usar env_ignore_empty: os testes dependem de REMOTE_API_*=""
    # sobrescrever o .env real do usuario.)
    @field_validator(
        "generation_temperature",
        "generation_seed",
        "generation_num_ctx",
        "generation_num_predict",
        "generation_think",
        "rag_index",
        "retrieval_translation_model",
        "retrieval_translation_cache",
        "retrieval_rerank_model",
        mode="before",
    )
    @classmethod
    def _vazio_e_none(cls, value: Any) -> Any:
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def _aplica_rag_index(self) -> "Settings":
        if self.rag_index:
            if self.rag_index not in RAG_INDICES:
                raise ValueError(
                    f"RAG_INDEX desconhecido: {self.rag_index!r} (versões: {', '.join(RAG_INDICES)})"
                )
            spec = RAG_INDICES[self.rag_index]
            self.vector_store_collection = spec["collection"]
            self.source_pdfs_dir = spec["source_dir"]
            self.embedding_model = spec["embedding_model"]
            for key in INDEX_SETTINGS_KEYS:
                if key in spec:
                    setattr(self, key, spec[key])
        return self

    @property
    def index_name(self) -> str:
        """Nome da versão do índice em uso (a coleção crua, se não for uma versão registrada)."""
        if self.rag_index:
            return self.rag_index
        return next(
            (name for name, spec in RAG_INDICES.items() if spec["collection"] == self.vector_store_collection),
            self.vector_store_collection,
        )

    @property
    def index_origins(self) -> list[str] | None:
        """Acervos de perguntas de referência que o índice em uso contém (RAG_INDICES, "origins")."""
        spec = RAG_INDICES.get(self.index_name)
        return list(spec["origins"]) if spec else None

    @property
    def source_pdfs_path(self) -> Path:
        return PROJECT_ROOT / self.source_pdfs_dir

    @property
    def vector_store_path(self) -> Path:
        return PROJECT_ROOT / self.vector_store_dir

    @property
    def logs_path(self) -> Path:
        return PROJECT_ROOT / self.logs_dir

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


# Sobrescritas em memoria aplicadas por cima do .env/ambiente. Existem para o
# harness de mutacao (tests/mutation/runner/sut.py), que precisa reconfigurar o
# pipeline entre mutantes sem reescrever o .env do usuario nem subir um processo
# novo por mutante. Nao sao usadas pela API em producao: quem nao chamar
# set_settings_overrides() ve exatamente o comportamento de antes.
_overrides: dict[str, Any] = {}


def set_settings_overrides(values: Mapping[str, Any] | None = None) -> None:
    """Substitui o conjunto de sobrescritas e invalida o cache de Settings."""
    global _overrides
    _overrides = dict(values or {})
    get_settings.cache_clear()


def get_settings_overrides() -> dict[str, Any]:
    return dict(_overrides)


@lru_cache
def get_settings() -> Settings:
    # Argumentos de init tem precedencia sobre env e .env no pydantic-settings.
    return Settings(**_overrides)
