<#
.SYNOPSIS
  Demo local del asistente en una notebook sin GPU (Windows + Docker Desktop).

.EXAMPLE
  .\demo-local.ps1 levantar      # arranca todo y abre http://localhost:8080/demo/
  .\demo-local.ps1 precalentar   # deja en cache las preguntas de la demo (responden al instante)
  .\demo-local.ps1 estado        # contenedores y chequeo de salud
  .\demo-local.ps1 logs          # logs de la API y del modelo en vivo
  .\demo-local.ps1 apagar        # detiene todo (conserva modelos, indice y cache)
#>
param(
    [ValidateSet("levantar", "precalentar", "estado", "logs", "apagar")]
    [string]$Accion = "levantar",
    [switch]$SinNavegador
)

Set-Location $PSScriptRoot
$Compose = @("-f", "docker-compose.yml", "-f", "docker-compose.local.yml")
$Url = "http://localhost:8080"

function Wait-Ready {
    Write-Host "Esperando al asistente (la primera vez descarga ~2,7 GB de modelos)..."
    $deadline = (Get-Date).AddMinutes(20)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri "$Url/health/ready" -UseBasicParsing -TimeoutSec 5
            if ($response.StatusCode -eq 200) {
                Write-Host "Listo: $Url/demo/" -ForegroundColor Green
                return $true
            }
        } catch { }
        Start-Sleep -Seconds 5
    }
    Write-Warning "No respondio a tiempo. Revisar con: .\demo-local.ps1 logs"
    return $false
}

switch ($Accion) {
    "levantar" {
        docker compose @Compose up -d --build
        if ((Wait-Ready) -and -not $SinNavegador) { Start-Process "$Url/demo/" }
    }
    "precalentar" {
        Write-Host "Generando las respuestas de la demo (10-25 s cada una la primera vez)..."
        docker compose @Compose run --rm --no-deps -T -v "${PSScriptRoot}\tests\e2e:/e2e:ro" api python /e2e/demo_check.py
    }
    "estado" {
        docker compose @Compose ps
        try { Invoke-RestMethod "$Url/health/ready" | ConvertTo-Json } catch { Write-Warning "El asistente no responde en $Url" }
    }
    "logs" {
        docker compose @Compose logs --tail 30 -f api llm
    }
    "apagar" {
        docker compose @Compose down
    }
}
