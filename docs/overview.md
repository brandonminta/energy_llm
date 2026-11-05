# Settling into the Wrong Memory
## Degeneración del mecanismo de recuperación en la memoria feed-forward de LLM

Documento de trabajo del proyecto (versión 0.2, noviembre 2026). Sustituye al plan de octubre y a la versión 0.1. Alcance acotado: una afirmación, dos partes, un repositorio que termina en la extracción.

---

## 1. De qué trata

Un modelo de lenguaje de pesos abiertos responde preguntas de hecho. En cada capa leemos cómo el estado oculto que entra al sublayer feed-forward (SwiGLU) interactúa con las filas de la matriz de compuerta, tratadas como banco de memorias de una red de Hopfield moderna. De ahí salen energías, entropías y divergencias que describen la *trayectoria de recuperación* de una respuesta a lo largo de capas y tokens.

La pregunta no es solo si la respuesta es falsa, sino **cómo falla la recuperación**:

| Régimen | Lectura Hopfield | Firma esperada | Literatura previa |
|---|---|---|---|
| Alucinación no asociada (UH) | la query nunca entra en una cuenca | entropía alta sostenida, poca masa top-k, conjunto de memorias inestable entre tokens | Cheang et al. 2026 (UH); Simhi et al. 2024 (HK−) |
| Alucinación asociada (AH) | compromiso temprano y profundo con una cuenca bien formada pero incorrecta | energía y entropía indistinguibles de la recuperación correcta; diferencia posible solo en competencia (varianza de scores) | Cheang et al. 2026 (AH); Simhi et al. 2024 (HK+); Cherukuri y Varshney 2026 |

Lo que nadie tiene: un observable de **un solo pase**, fijado por teoría, sobre la **memoria FFN**, que muestra en qué régimen está una respuesta y explica por qué los detectores internos se degradan en el régimen AH.

---

## 2. La afirmación del artículo

**Los detectores internos se degradan en las alucinaciones asociadas porque lo que leen es compromiso con la memoria paramétrica, y el compromiso es el mismo en una recuperación correcta y en una incorrecta.**

- **H1 (degradación por tipo).** Las métricas de concentración ($E_{\mathrm{lse}}$, $\bar H$, $D_{\mathrm{JS}}$ frente a la referencia) separan FA de UH con AUROC alto y FA de AH cerca del azar. Incluye como resultado descriptivo el signo de $\Delta\bar H$ por tipo (UH más difusa que FA; AH igual o más concentrada).
- **H2 (lo que queda).** En el estrato AH, las métricas de competencia y estabilidad ($\mathrm{Var}_\beta(s)$, Jaccard top-k entre tokens consecutivos) conservan poder discriminante donde la concentración lo pierde.

Falsación: si el diagnóstico de dimensiones atípicas (§6.3) invierte el signo de $\Delta\bar H$, el efecto de energía en capas superficiales se declara artefacto de magnitud y se retira.

Fuera de alcance (otros artículos): alineación de la actualización real con la regla Hopfield, atractor genérico de la plantilla, energía nativa sin $\beta$, localización por capa como hallazgo, escala de modelo.

---

## 3. Marco formal (mínimo)

Capa $l$ y posición $t$; se omite $(l)$ cuando no hay ambigüedad.

- Punto de sonda: el residual $x_t$ a la entrada de la segunda RMSNorm (`post_attention_layernorm`); $\hat x_t = x_t/\sqrt{\mathrm{mean}(x_t^2)+\epsilon}$, con $\|\hat x_t\|_2 \approx \sqrt D$, y $r_t = \mathrm{RMSNorm}(x_t) = \hat x_t \odot \gamma$.
- Sublayer: $\mathrm{FFN}(r) = \big(\mathrm{SiLU}(A r) \odot B r\big)^\top C$, con $A, B \in \mathbb R^{K\times D}$ (compuerta y rama lineal), $C \in \mathbb R^{K\times D}$.
- Banco: $\gamma$ se pliega en la compuerta, $\tilde W = A\,\mathrm{diag}(\gamma)$, filas $\tilde w_i$; banco normalizado con filas $\tilde w_i/\|\tilde w_i\|$ (nunca se materializa: se guardan $\gamma$ y $\|\tilde w_i\|$).
- Pre-activaciones nativas: $g = A r = \tilde W \hat x$.
- Scores (convención del código): $s_i = g_i/(\|\tilde w_i\|\sqrt D) = \cos(\tilde w_i, \hat x)\,\|\hat x\|/\sqrt D \approx \cos(\tilde w_i, \hat x) \in [-1, 1]$. La norma de la query sale del score; no hay término $E_{\mathrm{norm}}$ ($\tfrac12\|\hat x\|^2 \approx D/2$ es constante).
- Distribución de recuperación: $p_i(\beta) = \mathrm{softmax}(\beta s)_i$.
- Energía (Ramsauer et al. 2021): $E_{\mathrm{lse}} = -\tfrac1\beta\log\sum_i e^{\beta s_i}$, sin término cuadrático.
- Termodinámica de la lectura: $\partial(\beta E_{\mathrm{lse}})/\partial\beta = -\langle s\rangle_\beta$ y $\partial^2(\beta E_{\mathrm{lse}})/\partial\beta^2 = -\mathrm{Var}_\beta(s)$.

Tres niveles de correspondencia que el artículo declara: la compuerta como banco es una elección de modelado; el softmax es una lectura, no el mecanismo; no se ejecuta ninguna iteración de Hopfield.

---

## 4. Diseño experimental

```mermaid
flowchart LR
    subgraph Modelos
        Q[Qwen2.5-3B-Instruct]
        L[Llama-3.2-3B-Instruct]
    end
    subgraph Condiciones
        T[TruthfulQA<br/>imitativa, AH esperado]
        K[Dataset de vacío<br/>por fijar: TriviaQA o EntityQuestions<br/>UH esperado]
    end
    Q --> T & K
    L --> T & K
    T & K --> E[Etiqueta factual/alucinada<br/>juez externo + kappa humano]
    E --> R[Etiqueta de tipo por instancia<br/>AH vs UH]
```

- Celdas: 2 modelos x 2 condiciones. Tamaños y segunda condición de vacío se fijan tras el piloto.
- Etiqueta de corrección: juez externo con prompt bloqueado, respuestas archivadas, $\kappa$ frente a subconjunto humano ($n \ge 60$ por celda).
- Etiqueta de tipo **por instancia**: para cada respuesta alucinada, una forward con la atención del sujeto hacia el último token bloqueada (Geva et al. 2023; Cheang et al. 2026). JS de salida por encima de la media observada en FA: AH; por debajo: UH. La condición de dataset es covariable, no asignación de tipo.
- Baselines: confianza de salida (logprob medio, entropía media, media del logit negativo del token elegido); probe lineal sobre el residual en las posiciones de referencia; probe lineal sobre las métricas de recuperación.
- Evaluación: AUROC por estrato (FA vs UH, FA vs AH), validación cruzada repetida y estratificada, bootstrap pareado.

---

## 5. Pipeline y reparto de responsabilidades

```mermaid
flowchart TD
    subgraph REPO[Repositorio energy_llm: forward pass y primeros cálculos]
        S0[Etapa 0: bancos, beta*, diagnóstico de gamma<br/>una vez por modelo] --> S1
        S1[Etapa 1: generación greedy<br/>respuesta, ids, logprob, entropía, logit] --> S2
        S2[Etapa 2: teacher forcing + hooks<br/>métricas por capa y posición, top-k ids]
        I[Opcional: intervene<br/>JS de salida con atención bloqueada]
    end
    subgraph ANALISIS[Código aparte de Brandon]
        S3[Etapa 3: etiquetado<br/>juez, kappa, tipo AH/UH] --> S4
        S4[Etapa 4: análisis<br/>agregaciones, probes, baselines, figuras]
    end
    S2 -->|HDF5 por pregunta| S3
    I -.->|JS por id| S3
```

El repositorio recibe solo identificador y pregunta. No conoce oro, juez ni etiquetas. Todo se une por identificador.

### Etapa 0 (una vez por modelo)
1. Extraer $A^{(l)}$ de los pesos congelados; guardar $\|m_i\|$ y $\hat M^{(l)}$ en float32 con hash.
2. Diagnóstico de $\gamma$: las 5 dimensiones de mayor $|\gamma^{(l)}|$ por capa, guardadas en el manifiesto.
3. $\beta^\star$: uno por modelo, calibrado en 200 preguntas de un corpus neutro ajeno a las condiciones, objetivo $\bar H^\star = 0{,}55$ en la posición de referencia, interpolación en $\log\beta$. Nunca por dataset ni por pregunta.

### Etapa 1
Plantilla de chat fija, greedy, tope de tokens fijo, semilla fija. Se guardan respuesta, ids del prompt y de la respuesta, y por token generado: logprob, entropía de salida, logit del token elegido.

### Etapa 2
Un forward con teacher forcing sobre prompt + respuesta. Hooks: salida de `gate_proj` ($g$) y entrada de `mlp` ($r$). Por capa y posición se calculan las métricas de §6 y se descartan los vectores de tamaño $K_l$. Posiciones de referencia guardadas: último token de contenido de la pregunta, último token del prompt (token de plantilla), primer token generado.

### Etapas 3 y 4 (fuera del repositorio)
Etiquetado, estratificación AH/UH, agregaciones sobre tokens y capas (media, último, máximo, dispersión, pendiente, capa de compromiso, Jaccard entre tokens), probes, AUROC por estrato, sensibilidad a $\beta$ desde la grilla, ablaciones.

---

## 6. Métricas

### 6.1 Núcleo (siempre se calcula)

| Métrica | Definición | Para qué | $\beta$ | Dataset (`traces.h5`) |
|---|---|---|---|---|
| $E_{\mathrm{lse}}$ | $-\tfrac1\beta\log\sum_i e^{\beta s_i}$ | concentración (H1) | grilla | `energy` |
| $\bar H$ | $H(p)/\log K$ | concentración (H1) | grilla | `norm_entropy` |
| masa top-10 | $\sum_{i\in\mathrm{top10}} p_i$ | compromiso (H1) | grilla | `topk_mass` |
| $\langle s\rangle_\beta$ | $\sum_i p_i s_i$ | primera derivada de $\beta E_{\mathrm{lse}}$ | grilla | `mean_score` |
| $\mathrm{Var}_\beta(s)$ | $\sum_i p_i(s_i-\langle s\rangle)^2$ | competencia (H2) | grilla | `var_score` |
| $D_{\mathrm{JS}}(p_t\|p_{\mathrm{ref}})$ | simétrica, acotada | desplazamiento entre fases (H1) | grilla | `js/<ref>` |
| $\Delta_{\mathrm{KL}}$ | $D_{\mathrm{KL}}(p_t\|p_{\mathrm{ref}}) - D_{\mathrm{KL}}(p_{\mathrm{ref}}\|p_t)$ | estrechamiento vs deriva (H1, H2) | grilla | `dkl/<ref>` |
| margen | $s_{(1)} - s_{(2)}$ | separación del ganador | no | `margin` |
| fracción top-1 de $\hat x$ | $\max_j \hat x_j^2/D$ | activaciones masivas (diagnóstico) | no | `xhat_top1_frac` |
| ids top-100 | índices de los 100 mayores $s_i$ | Jaccard frente a referencia y entre tokens consecutivos (H2), en análisis | no | `topk_ids` |
| salida por token | logprob, entropía, logit del elegido | baselines | no | `answers.parquet`: `token_logprobs`, `token_entropies`, `token_logits`; `baselines.parquet` |

`<ref>` ∈ {`question_last`, `prompt_last`, `response_first`}: último token de la pregunta, último token del prompt, primer token generado. Las comparativas se calculan frente a las tres referencias. Hellinger se elimina (redundante con JS; diferencia menor a 0,005 AUROC en los tres experimentos previos).

### 6.2 Extras (no implementados en v2.1.0)
Energía nativa $E_{\mathrm{nat}}$ con $F(z)=z\,\mathrm{softplus}(z)+\mathrm{Li}_2(-e^z)$; razón de participación de los coeficientes nativos $a_i$; $\|\mathrm{FFN}(r)\|$ y cosenos de $\mathrm{FFN}(r)$ con $r$ y con $\bar m - r$; Spearman entre scores; distribución nula de la plantilla vacía. Requieren hooks adicionales (`up_proj`, salida de `mlp`).

### 6.3 Diagnóstico de dimensiones atípicas (piloto y ablación)
Por capa, las 5 dimensiones de mayor $|\gamma|$ y la fracción de $\sum_i\|\tilde w_i\|^2$ que cargan (`bank.h5`, `manifest.json: gamma_diagnostic`). El `smoke` recalcula $\bar H$ con esas 5 entradas de $\gamma$ a cero y reporta, por tercio de capas, $\bar H(\text{respuesta}) - \bar H(\text{prompt\_last})$ intacto frente a ablacionado: si cambia el signo. Solo informe: no hay bandera de ablación en producción.

---

## 7. Tratamiento de datos

```mermaid
flowchart LR
    W[pesos A y gamma<br/>float32] --> N1[normalizar filas de A<br/>guardar normas y hash]
    H[hook x en la entrada de<br/>post_attention_layernorm<br/>bfloat16] --> C1[upcast a float32<br/>x_hat = x / RMS]
    C1 --> S[s = W_g · gamma x_hat / norma / sqrt D]
    S --> P[softmax beta s<br/>grilla de 9 betas más beta*]
    P --> M1[escalares por capa y posición<br/>float16]
    S --> R[ids top-100 por capa y posición<br/>int32]
    C1 --> V[x en las 3 referencias<br/>y en el último token de respuesta<br/>float16]
    M1 & R & V --> F[HDF5 por pregunta<br/>traces.h5, hidden.h5; manifest.json]
```

- Modelo en bfloat16; métricas en float32 tras upcast; almacenamiento float16 salvo ids.
- Grilla de $\beta$: 9 valores absolutos log-espaciados entre $10^0$ y $10^4$ (medias décadas; iguales para los dos modelos porque $s$ es un coseno), más $\beta^\star$ al final: $n_\beta = 10$. Toda métrica dependiente de $\beta$ se guarda para cada valor.
- No se guardan vectores de scores completos ($K_l\times L\times T\times N$ supera los 40 GB). Los ids top-100 bastan para Jaccard e inspección cualitativa.
- Esquema (`$SCRATCH_ROOT/energy_llm/runs/<modelo>/extract/<nombre>/`): `traces.h5` con `/<id>/{energy, norm_entropy, topk_mass, mean_score, var_score, js/<ref>, dkl/<ref>}` de forma $(L,T,n_\beta)$, `/<id>/{margin, xhat_top1_frac}` $(L,T)$, `/<id>/topk_ids` $(L,T,100)$ int32, `/<id>/role_mask` $(T)$ y atributos `betas`, `reference_positions`; `hidden.h5` con `/<id>/{question_last, prompt_last, response_first, response_last}` $(L,D)$; tokens, logprob, entropía y logit en `answers.parquet`; procedencia en `manifest.json`.
- Coste (Qwen 3B, $L=36$, $T\approx 90$, $n_\beta = 10$): unos 2 MB por pregunta antes de gzip.

---

## 8. Invariantes del piloto (50 preguntas, un modelo)

1. Estados con teacher forcing iguales a los de generación: $\|\Delta r\|/\|r\| < 10^{-3}$ en todas las capas.
2. $g$ capturado en `gate_proj` igual a $A r$ con $r$ capturado en la entrada de `mlp` (manejo de $\gamma$ correcto).
3. $\bar H$ en la referencia dentro de $0{,}55\pm0{,}05$ con $\beta^\star$.
4. Diagnóstico de $\gamma$ ejecutado y reportado antes de escalar.
5. Determinismo: dos ejecuciones con la misma semilla producen archivos idénticos en ids y métricas hasta float16.

---

## 9. Referencias que fijan la posición

- Ramsauer et al. 2021, Hopfield networks is all you need.
- Krotov y Hopfield 2016 y 2021, dense associative memory.
- Geva et al. 2021 (FFN como memoria clave-valor); Geva et al. 2023 (flujo sujeto a último token).
- Gupta et al. 2025 (down-projection como memoria asociativa lineal).
- Cheang, Chan, Zhang y Deng 2026 (AH/UH; los estados internos reflejan recuperación, no veracidad).
- Simhi et al. 2024 (HK−/HK+, WACK); Orgad et al. 2025 (tokens de respuesta exacta).
- Cherukuri y Varshney 2026 (cuencas de alucinación).
- Vu et al. 2025, HalluField; Ma et al. 2026, Semantic Energy (energía sobre logits, competidores directos).

