# Prepara o ambiente Ollama para o projeto: define OLLAMA_MODELS, baixa os modelos
# de geracao e embeddings, e valida que os pesos foram salvos no destino esperado.
#
# Uso: powershell -ExecutionPolicy Bypass -File scripts\setup_ollama.ps1

param(
    [string]$ModelsPath = "D:\modelosLLM\models",
    [string]$GenerationModel = "qwen3:8b",
    [string]$EmbeddingModel = "nomic-embed-text"
)

Write-Host "== Verificando instalacao do Ollama ==" -ForegroundColor Cyan
$ollamaVersion = ollama --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Ollama nao encontrado. Instale em https://ollama.com/download antes de continuar." -ForegroundColor Red
    exit 1
}
Write-Host $ollamaVersion

$currentModelsPath = [Environment]::GetEnvironmentVariable("OLLAMA_MODELS", "User")
if ($currentModelsPath -ne $ModelsPath) {
    Write-Host "== Configurando OLLAMA_MODELS = $ModelsPath ==" -ForegroundColor Cyan
    if (-not (Test-Path $ModelsPath)) {
        New-Item -ItemType Directory -Force -Path $ModelsPath | Out-Null
    }
    setx OLLAMA_MODELS "$ModelsPath" | Out-Null
    Write-Host "Variavel definida. Reinicie o servico do Ollama (ou a sessao) para que valha." -ForegroundColor Yellow
} else {
    Write-Host "OLLAMA_MODELS ja aponta para $ModelsPath" -ForegroundColor Green
}

Write-Host "== Baixando modelo de geracao: $GenerationModel ==" -ForegroundColor Cyan
ollama pull $GenerationModel

Write-Host "== Baixando modelo de embeddings: $EmbeddingModel ==" -ForegroundColor Cyan
ollama pull $EmbeddingModel

Write-Host "== Validando modelos com um prompt de teste ==" -ForegroundColor Cyan
$body = @{ model = $GenerationModel; prompt = "Responda em uma frase: o que e RAG?"; stream = $false } | ConvertTo-Json
$resp = Invoke-RestMethod -Uri "http://localhost:11434/api/generate" -Method Post -Body $body -ContentType "application/json"
Write-Host "Resposta do modelo de geracao: $($resp.response)" -ForegroundColor Green

$body2 = @{ model = $EmbeddingModel; prompt = "teste" } | ConvertTo-Json
$resp2 = Invoke-RestMethod -Uri "http://localhost:11434/api/embeddings" -Method Post -Body $body2 -ContentType "application/json"
Write-Host "Dimensao do embedding: $($resp2.embedding.Count)" -ForegroundColor Green

Write-Host "== Confirmando local dos pesos ==" -ForegroundColor Cyan
Get-ChildItem -Path $ModelsPath -Recurse -Depth 1 | Select-Object FullName, Length | Format-Table -AutoSize

Write-Host "Setup concluido." -ForegroundColor Green
