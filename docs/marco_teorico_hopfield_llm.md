# Marco Teórico: Detección de Alucinaciones en LLMs mediante Estados Internos y Redes de Hopfield Modernas

**Investigación:** `hopfield_llm` — Detección de alucinaciones sin clasificador mediante divergencia de patrones de recuperación en capas MLP de transformadores  
**Autor:** Brandon — Yachay Tech University  
**Versión:** 1.0 · Mayo 2026

> **Nota:** este documento es la síntesis temprana en español de la literatura.
> El marco teórico **autoritativo y final** es el capítulo de tesis
> `docs/fundamentals.tex` (y la metodología, `docs/methodology.tex`). Se conserva
> aquí como ayuda de lectura.

---

## 1. Introducción y Motivación

La pregunta central de esta investigación es:

> *¿Difieren los patrones internos de memoria de un LLM entre respuestas correctas y alucinadas — y si es así, en qué capas?*

Para responderla, esta investigación adopta la hipótesis de que las capas feedforward (MLP) de los transformadores actúan como memorias asociativas de tipo Hopfield moderno, y que la alucinación se manifiesta como una divergencia anómala en el patrón de recuperación durante la generación de la respuesta respecto al prefill de la pregunta. Este marco teórico conecta once trabajos que colectivamente fundamentan esta hipótesis, desde la teoría de redes de Hopfield modernas hasta métodos de detección estado del arte.

---

## 2. Fundamentos: Las Capas Feedforward como Memorias Clave-Valor

### 2.1 Geva et al. (2021) — *Transformer Feed-Forward Layers Are Key-Value Memories*

**Resumen.** Este trabajo seminal demuestra empíricamente que las capas feedforward (FF) de los transformadores operan como memorias clave-valor [Geva et al., 2021]. La primera matriz de parámetros $W_1 \in \mathbb{R}^{d_{ff} \times d}$ actúa como banco de **claves**, donde cada fila $k_i$ se activa ante patrones textuales específicos. La segunda matriz $W_2 \in \mathbb{R}^{d \times d_{ff}}$ almacena los **valores**, donde cada columna $v_i$ induce una distribución sobre el vocabulario de salida. La salida de una capa FF es:

$$y^\ell = \sum_i \text{ReLU}(x^\ell \cdot k_i^\ell) \cdot v_i^\ell + b^\ell$$

Los experimentos sobre un modelo de 247M parámetros entrenado en WikiText-103 revelan que: (i) las capas inferiores capturan patrones superficiales (n-gramas), mientras que las capas superiores codifican conceptos semánticos; (ii) los valores en capas superiores predicen tokens con acuerdo no trivial ($3.5\%$ vs. $0.0004\%$ aleatorio); (iii) la predicción final emerge de la composición iterativa de memorias a través de conexiones residuales.

**Impacto en `hopfield_llm`.** Este trabajo es la **piedra angular** de la investigación. Justifica tratar $W_1$ como banco de claves y $W_2$ como banco de valores en la sonda MHN. La observación de que las capas superiores tienen mayor poder predictivo motiva examinar capas $\ell > 14$ de Qwen2.5-1.5B (28 capas) con mayor peso en el AUROC por capa. La composición vía residual implica que la señal de detección no reside en una sola capa sino en la trayectoria completa — lo que valida el diseño multi-capa de la sonda y el uso de regresión logística sobre features de todas las capas.

**Limitación relevante.** El análisis se realizó sobre un modelo ReLU-FFN; Qwen2.5 usa SwiGLU, lo que modifica la forma del coeficiente de activación (ver Sección 3).

---

## 3. La Equivalencia Hopfield–Atención y la Energía de Recuperación

### 3.1 Ramsauer et al. (2021) — *Hopfield Networks Is All You Need*

**Resumen.** Este trabajo introduce las **Redes de Hopfield Modernas (MHN)** con estados continuos y demuestra que la regla de actualización del MHN es *matemáticamente equivalente* al mecanismo de atención de los transformadores [Ramsauer et al., 2021]. La energía del MHN continuo es:

$$E = -\text{lse}(\beta, X^T \xi) + \frac{1}{2}\xi^T\xi + \frac{1}{\beta}\log N + \frac{M}{2}$$

donde $\text{lse}(\beta, z) = \frac{1}{\beta}\log\sum_i e^{\beta z_i}$, $X$ es la matriz de patrones almacenados (claves), $\xi$ es el patrón de consulta (query), $\beta$ es la inversa de temperatura, $N$ es el número de patrones y $M$ el máximo de normas al cuadrado. La regla de actualización que minimiza esta energía es:

$$\xi^{new} = X \cdot \text{softmax}(\beta X^T \xi)$$

que es precisamente la operación de atención con $Q = \xi$, $K = X$, $V = X$. Propiedades clave: capacidad de almacenamiento exponencial $e^{\alpha d}$ (con $d$ la dimensión del espacio asociativo), recuperación con **una sola actualización** y error de recuperación exponencialmente pequeño.

**Impacto en `hopfield_llm`.** Establece el **lenguaje formal** de la investigación. La energía $E$ de una consulta $\xi$ respecto al banco $X$ mide la "facilidad de recuperación" del patrón: energías bajas corresponden a recuperación estable (patrón almacenado), energías altas a recuperación inestable o ambigua. Cuando el modelo genera una alucinación, el vector de activación MLP (la consulta $\xi$) puede no tener un patrón bien separado en el banco $X = W_1^T$, produciendo energía anómala. Los **cuatro términos de la energía** deben mantenerse separados — el término $\text{lse}$ captura la estabilidad de recuperación, $\frac{1}{2}\xi^T\xi$ la norma de la consulta, $\frac{1}{\beta}\log N$ la penalización por capacidad, y $\frac{M}{2}$ la corrección de escala. Colapsar estos términos perdería la semántica física. El parámetro $\beta$ opera como la inversa de temperatura: $\beta \to \infty$ produce recuperación puntual (atención tipo argmax), $\beta \to 0$ produce promedios globales (metaestados).

**Limitación relevante.** La equivalencia exacta con la atención del transformador asume que $Q$, $K$, $V$ son proyecciones lineales del mismo espacio. En la sonda MHN aplicada a MLP de Qwen2.5, los espacios de banco y consulta son distintos (espacio intermedio vs. espacio de entrada), lo que constituye una extensión del marco original. Esta extensión está justificada por los resultados de Gupta et al. (2025) (ver §4.2).

---

## 4. La Naturaleza Asociativa de la Memoria en LLMs

### 4.1 Gupta et al. (2025) — *How Linearly Associative Are Memories in Large Language Models?*

**Resumen.** Este trabajo investiga en qué medida la segunda matriz MLP de los LLMs se comporta como una **Memoria Asociativa Lineal (LAM)** [Gupta et al., 2025]. En el marco LAM, la recuperación es:

$$\hat{y} = Wx, \quad W = \sum_i y_i x_i^T$$

La condición de no-interferencia requiere que los vectores de activación de entrada sean aproximadamente ortogonales entre sí. Los autores miden los ángulos por pares entre vectores de activación usando GPT-2 XL, Llama-2-7B y la familia Pythia (70M–6.9B). Hallazgos principales: (i) para tokens aleatorios de Wikipedia, la segunda matriz MLP exhibe alta ortogonalidad ($\approx 90°$), lo que soporta la interpretación LAM para recuperación genérica; (ii) para tokens **sujeto** en tareas de recuperación factual (CounterFact), la ortogonalidad cae significativamente, indicando mayor interferencia y entrelazamiento entre memorias factuales; (iii) los modelos más grandes (6.9B) mantienen representaciones más estables y distintas a través de capas.

**Impacto en `hopfield_llm`.** Tiene dos implicaciones directas. **Primera:** la baja ortogonalidad en tokens sujeto factual implica que las consultas de hechos específicos atraviesan un espacio con alta interferencia — precisamente donde la alucinación es más probable. Esto provee una justificación geométrica de por qué la energía MHN es anómala durante alucinaciones: la consulta $\xi$ no aterriza en una región bien separada del banco $X$. **Segunda:** justifica el uso de la variante de **espacio de valores** en la sonda (banco `down_values`, hook `mlp_output`) como sonda secundaria, dado que la segunda matriz MLP tiene mayor ortogonalidad y por tanto mejor discriminabilidad para tokens no-factuales. Para Qwen2.5-1.5B (1.5B parámetros), se espera un comportamiento intermedio entre Pythia-1B y Pythia-6.9B.

**Contraste con Geva et al. (2021).** Mientras Geva et al. presentan las FF como memorias clave-valor, Gupta et al. muestran que la interpretación **falla parcialmente** para recuperación factual (alta interferencia). Esta tensión es productiva para la investigación: el detector de alucinaciones aprovecha precisamente esta falla — cuando la recuperación factual interfiere, la energía MHN diverge.

---

## 5. Activaciones Controladas: GLU y SwiGLU

### 5.1 Dauphin et al. (2017) — *Language Modeling with Gated Convolutional Networks*

**Resumen.** Introduce las **Gated Linear Units (GLU)**, definidas como el producto componente-a-componente de dos proyecciones lineales, una de las cuales pasa por una sigmoide [Dauphin et al., 2017]:

$$\text{GLU}(x, W, V) = \sigma(xW) \otimes (xV)$$

La motivación original fue el modelado de lenguaje con redes convolucionales, donde el mecanismo de compuerta reduce el problema del gradiente evanescente y permite paralelización sobre tokens. El trabajo demostró por primera vez que un enfoque no-recurrente podía ser competitivo con LSTMs en benchmarks de gran escala (WikiText-103, Google Billion Words).

**Impacto en `hopfield_llm`.** Provee el antecedente histórico de la compuerta multiplicativa que luego se generalizará en SwiGLU. Para la investigación, importa entender que la compuerta $\sigma(xW)$ introduce una **selección suave de claves**: no todas las columnas del banco contribuyen igualmente, sino solo las que superan el umbral gateado. Esto modifica la interpretación de "coeficiente de activación" respecto al ReLU simple.

---

### 5.2 Shazeer (2020) — *GLU Variants Improve Transformer*

**Resumen.** Propone variantes de GLU para las capas FF del transformador, incluyendo **SwiGLU** [Shazeer, 2020]:

$$\text{SwiGLU}(x, W, V, W_2) = (\text{Swish}_1(xW) \otimes xV) W_2$$

donde $\text{Swish}_\beta(x) = x\sigma(\beta x)$. La operación de compuerta con Swish produce gradientes suaves y no-zero para entradas negativas, a diferencia de ReLU. Para mantener el conteo de parámetros constante respecto a FFN estándar, el tamaño oculto $d_{ff}$ se reduce a $\frac{2}{3}$ del original — generando **tres matrices de peso** ($W$, $V$, $W_2$) en lugar de dos.

**Impacto en `hopfield_llm`.** Qwen2.5 (y Llama-3, Phi, Gemma, Mistral) utilizan SwiGLU. Esto tiene consecuencias directas en el diseño de la sonda MHN:

1. El banco de claves ya no es simplemente $W_1$ sino el resultado $\text{Swish}_1(xW) \otimes xV$, cuya interpretación como "activación de clave" es más compleja que en ReLU-FFN.
2. El coeficiente de activación para la clave $i$ es $a_i = \text{Swish}_1(x W_{:,i}) \cdot (x V_{:,i})$, que mezcla dos proyecciones distintas del input.
3. Para la sonda, se usa `up_proj` output como banco de claves y `gate_proj` output como pesos de compuerta — siguiendo la implementación de Qwen2.5 donde la capa MLP tiene `gate_proj`, `up_proj`, `down_proj`.
4. Los valores (`down_proj` input = `up_proj` output $\otimes$ gate) corresponden al banco de valores en el espacio de dimensión $d_{ff}$.

**Contraste entre Dauphin et al. y Shazeer.** Dauphin propone la compuerta para modelado convolucional; Shazeer la integra directamente en la capa FF del transformador. La diferencia clave es que SwiGLU opera sobre el vector de posición actual (no sobre una ventana convolucional), preservando la interpretación de "memoria posicional" de Geva et al.

---

## 6. Métodos de Detección de Alucinaciones: Estado del Arte

### 6.1 Vu et al. (2025) — *HalluField: Detecting LLM Hallucinations via Field-Theoretic Modeling*

**Resumen.** HalluField introduce un marco de detección basado en **termodinámica y principio variacional** [Vu et al., 2025]. El LLM es modelado como un sistema termodinámico: cada secuencia de tokens generada $\tau$ es un "camino" con energía libre asociada $F_Q(\tau) = -\sum_i \log P(\tau_i | \{\tau_j\}_{j<i}, Q)$ y entropía $H_Q(\tau) = -\sum_i \sum_r P(\tau_i(r,T)) \log P(\tau_i(r,T))$. La detección opera comparando la estabilidad del paisaje energético bajo **perturbaciones de temperatura** $T \to T + \Delta T$: si la respuesta es alucinada, el paisaje energético es inestable (alta variación total $\delta F_Q$). HalluField opera directamente sobre **logits de salida** sin fine-tuning ni redes auxiliares, con tiempo de ejecución de $10^{-4}$ segundos por consulta. En TriviaQA y BioASQ con LLaMA-2 7B: AUC $\approx 0.80$–$0.83$.

**Impacto en `hopfield_llm`.** Es el **trabajo más cercano** metodológicamente — y la diferencia más importante que justifica esta investigación. HalluField opera sobre **logits de salida** (espacio de vocabulario), mientras que `hopfield_llm` opera sobre **estados internos MLP** (espacio de activaciones intermedias). Esta distinción es fundamental:

- HalluField no puede localizar *en qué capa* ocurre la inestabilidad.
- `hopfield_llm` provee diagnóstico **por capa** con métricas AUROC por $\ell$, identificando las capas más informativas para la detección.
- HalluField requiere perturbaciones de temperatura (múltiples forward passes en variantes como HalluFieldSE); `hopfield_llm` requiere un solo forward pass con hooks de activación.
- La energía de HalluField ($-\log P$) es la log-verosimilitud del token; la energía MHN de `hopfield_llm` mide la *facilidad de recuperación del patrón de activación* en la memoria paramétrica — dos conceptos relacionados pero distintos.

**Contraste de resultados.** HalluField reporta AUC 0.80–0.83 en TriviaQA/BioASQ. Esto establece el **baseline externo** contra el cual comparar los resultados de `hopfield_llm`. La hipótesis es que los estados internos MLP proveen señal complementaria, especialmente en capas intermedias donde la recuperación factual está más localizada.

---

### 6.2 Bar-Shalom et al. (2025) — *Beyond Token Probes: Hallucination Detection via Activation Tensors with ACT-ViT*

**Resumen.** ACT-ViT argumenta que los clasificadores de sondeo tradicionales operan sobre **pares aislados capa-token**, ignorando la estructura global del tensor de activaciones $A \in \mathbb{R}^{L \times N \times D}$ [Bar-Shalom et al., 2025]. La propuesta trata el tensor de activaciones como una **imagen** (dimensiones: capas $\times$ tokens $\times$ dimensión oculta ≅ altura $\times$ anchura $\times$ canales) y aplica una arquitectura Vision Transformer (ViT). Adaptadores lineales por modelo ($L_{AM}: A_p \mapsto A_p W_M$) permiten entrenamiento multi-LLM y transferencia zero-shot. En 15 combinaciones LLM-dataset (Mistral-7B, LLaMA-8B, Qwen-7B sobre TriviaQA, HotpotQA, IMDB, Movies), ACT-ViT supera consistentemente las sondas lineales por $5$–$15\%$ AUC. Tiempo de ejecución: $\approx 10^{-5}$ segundos.

**Impacto en `hopfield_llm`.** Provee **dos contribuciones clave**. Primera: confirma experimentalmente que la información de alucinación está distribuida sobre múltiples capas y tokens — justificando la captura de trayectorias token-level en M5 y el diseño per-layer AUROC en M5. Segunda: los resultados de Qwen-7B sobre TriviaQA (AUC ~80–89% según dataset) establecen un **techo de referencia** para comparación directa dado que `hopfield_llm` usa Qwen2.5-1.5B.

**Diferencia crítica con `hopfield_llm`.** ACT-ViT es un **método supervisado** — requiere etiquetas de alucinación para entrenar la ViT y los adaptadores. `hopfield_llm` es fundamentalmente **no supervisado** en la sonda MHN: las métricas de energía, entropía y derivadas se computan sin entrenamiento. Solo la regresión logística en M5 añade un componente supervisado mínimo (para calibración de AUROC), que puede compararse directamente con ACT-ViT en modo de pocos datos.

**Limitación compartida.** Ambos métodos requieren acceso a los pesos del modelo (white-box). ACT-ViT reporta que el pooling inicial de activaciones puede descartar señal informativa — problema que `hopfield_llm` mitiga al operar directamente sobre los vectores de activación sin reducción dimensional previa.

---

### 6.3 Vazhentsev et al. (2026) — *Leveraging LLM Parametric Knowledge for Fact Checking without Retrieval*

**Resumen.** Este trabajo propone **INTRA** (Intrinsic Truthfulness Assessment), un método de verificación de hechos sin recuperación externa que explota las **representaciones internas del LLM** a través de capas y tokens [Vazhentsev et al., 2026]. Evalúa 18 métodos sobre 9 datasets heterogéneos (incluyendo PopQA, X-Fact, AVeriTeC) con Llama 3.1, Ministral y Phi-4. INTRA combina estados ocultos token-level a través de capas alcanzando ROC-AUC promedio de 77.7% (Llama 3.1), superando al segundo mejor método Sheeps en 2.7%. Un resultado notable: los métodos basados en logits **frecuentemente** rinden peor que los que explotan representaciones internas (ROC-AUC de SP: 70.9% vs. INTRA: 77.7%). El análisis por capa muestra que la señal discriminativa varía significativamente entre capas y datasets — con máximos en capas medias-altas para la mayoría de benchmarks.

**Impacto en `hopfield_llm`.** Provee **dos validaciones empíricas** directas para las hipótesis de la investigación. Primera: confirma que las representaciones internas superan a las señales de salida (logits) para detección de alucinaciones — justificando el diseño basado en activaciones MLP. Segunda: el análisis por capa (Figura 2 del paper) muestra perfiles AUC no-monotónicos con picos en capas específicas, exactamente el patrón que `hopfield_llm` busca caracterizar mediante energía MHN por capa. Establece además un **baseline fuerte** en el dominio de verificación sin recuperación: INTRA es el estado del arte no-supervisado, superando métodos de recuperación-RAG (77.7% INTRA vs. 77.5% Verb+RAG en ROC-AUC, con 20× menor costo computacional).

**Contraste metodológico.** INTRA usa los estados ocultos como features directos (vector de embeddings concatenado), con una capa de clasificación ligera. `hopfield_llm` transforma los estados ocultos en métricas energéticas derivadas de la interpretación MHN — introduciendo un prior teórico sobre *cómo* los patrones se almacenan y recuperan, en lugar de tratar las activaciones como vectores arbitrarios.

---

## 7. Detección de Distribución Anómala con Energía Hopfield

### 7.1 Hofmann et al. (2024) — *Energy-based Hopfield Boosting for Out-of-Distribution Detection*

**Resumen.** Hopfield Boosting aplica la **energía MHN** para detección de muestras fuera de distribución (OOD) en visión computacional [Hofmann et al., 2024]. La energía MHN es:

$$E_b(\xi; X, O) = -\text{lse}(\beta, X^T\xi) + \text{lse}(\beta, O^T\xi)$$

donde $X$ es la memoria de patrones in-distribution y $O$ la memoria de patrones auxiliares outlier. Muestras con alta $E_b$ respecto a $X$ (baja similitud con distribución conocida) se identifican como OOD. El método boosting pondera los ejemplos auxiliares según su proximidad a la frontera de decisión, creando weak learners que refuerzan la frontera. En CIFAR-10: FPR95 de $0.92\%$ (desde $2.28\%$ previo); en ImageNet-1K: FPR95 de $36.60\%$ (desde $50.74\%$).

**Impacto en `hopfield_llm`.** Provee la **justificación más directa** para usar energía MHN como detector de anomalías. La analogía es: durante generación correcta, los vectores de activación MLP son "in-distribution" respecto al banco de claves $W_1$ (patrones aprendidos durante preentrenamiento); durante alucinación, son "OOD" — el modelo intenta recuperar información que no está bien consolidada en su memoria paramétrica. La energía $E = -\text{lse}(\beta, W_1^T \xi)$ actúa como score de anomalía. La innovación de `hopfield_llm` respecto a este framework es aplicarlo **dentro del LLM** (a las activaciones intermedias), no a la representación de salida del modelo.

**Contraste.** Hopfield Boosting usa una memoria *aprendida separada* ($X$, $O$ son embeddings de una red separada). `hopfield_llm` usa la memoria *paramétrica del propio modelo* ($W_1$ de cada capa MLP) — lo que elimina la necesidad de entrenamiento adicional y conecta más directamente con la hipótesis clave-valor.

---

## 8. Dinámica Interna: Atractores y Estabilidad Representacional

### 8.1 Anónimo (2025) — *Concept Attractors in LLMs and their Applications*

**Resumen.** Este trabajo establece formalmente que las capas de un LLM implementan un **Sistema de Funciones Iteradas (IFS)** que mapea prompts semánticamente relacionados hacia regiones estables del espacio latente — denominadas **Atractores de Concepto** [Anón., 2025]. Formalmente, si $f_\ell$ es la función de la capa $\ell$, entonces la composición $f_L \circ \cdots \circ f_1$ es una contracción (en el sentido del teorema de punto fijo de Banach) hacia un conjunto atractor $\mathcal{A}_C$ específico al concepto $C$. Experimentos con Llama-3.1-8B muestran convergencia de representaciones para 28 prompts sobre 4 "mundos" (Tolkien, Narnia, Star Wars, Harry Potter) a partir de la capa $\approx 24$. Aplicaciones: detección de conceptos, desvío de atractores para traducción de código (transpiler), reducción de alucinaciones en VLMs (reducción $>20\%$ en CHAIR), y generación de datos sintéticos con mejor factualidad.

**Impacto en `hopfield_llm`.** Provee el **marco dinámico** complementario a la visión estática de energía MHN. En el lenguaje de atractores: una respuesta **correcta** converge a un atractor estable asociado al concepto factual relevante; una **alucinación** indica que el sistema no converge — o converge a un atractor incorrecto (concepto factualmente equivocado) o a un estado metaestable (promedio de múltiples conceptos parcialmente compatibles). El análogo MHN de "convergencia al atractor" es "energía baja en el punto fijo"; "falta de convergencia" corresponde a energía alta o alta entropía de la distribución softmax. La observación de que la convergencia ocurre en capas altas ($\ell \approx 24$ en Llama-3.1-8B, ~$85\%$ de profundidad) sugiere que las capas intermedias-altas de Qwen2.5-1.5B (~capas 18–26 de 28) serán las más discriminativas para la sonda MHN.

**Limitación.** El paper usa Llama-3.1-8B; la transferencia de estas observaciones a Qwen2.5-1.5B (un modelo $\approx 5\times$ más pequeño) no está garantizada — y constituye una pregunta empírica que `hopfield_llm` puede responder.

---

## 9. Recuperación con Hopfield en Sistemas de QA

### 9.1 Anónimo (2026) — *Conv-CoA: Open-Domain Question Answering via Conversational Chain-of-Action with Hopfield Retriever*

**Resumen.** Conv-CoA propone un framework de QA conversacional abierto que combina razonamiento encadenado (Chain-of-Action) con un **recuperador Hopfield eficiente** [Anón., 2026]. El recuperador Hopfield modela el espacio de documentos como patrones almacenados en una MHN, y la consulta reformulada como el patrón de entrada. La ventaja respecto a recuperadores densos tradicionales es la capacidad de actualización incremental del conjunto de memorias (Contextual Knowledge Set, CKS) sin re-entrenamiento. Introduce además el Conv-MRFS (conversational multi-reference faithfulness score) para verificar coherencia entre conocimiento recuperado y respuesta generada. En comparación con 23 métodos sobre dos benchmarks públicos, Conv-CoA supera en accuracy y eficiencia.

**Impacto en `hopfield_llm`.** Este trabajo es **contextual y motivacional** más que metodológicamente central. Demuestra que las redes Hopfield son una herramienta práctica y competitiva para recuperación en QA — reforzando la coherencia de usar el mismo formalismo para *diagnosticar* la recuperación interna del LLM. El dominio de aplicación (QA) es idéntico al de TriviaQA, el dataset principal de `hopfield_llm`. La noción de "faithfulness score" en Conv-CoA es análoga a la "divergencia de patrones de recuperación" en `hopfield_llm`: ambas buscan medir si la respuesta es coherente con el conocimiento disponible.

---

## 10. Síntesis: Conexión entre los Papers y la Investigación

```
Geva et al. (2021) ──────────────────────────────────────────────────────────┐
[FF layers = key-value memories]                                             │
         │                                                                   │
         ▼                                                                   │
Ramsauer et al. (2021) ──────────────────────────────────────────────────────┤
[MHN ≡ attention; E = -lse(β, X^Tξ) + ½ξ^Tξ + ...]                        │
         │                                                                   │
         ├──► Gupta et al. (2025) ──────────────────────────────────────────►│
         │    [LAM: ortogonalidad alta genérica,                             │
         │     baja para recall factual → interferencia en alucinaciones]    │
         │                                                                   │
         ├──► Shazeer (2020) + Dauphin et al. (2017) ────────────────────►  │
         │    [SwiGLU: W_gate ⊗ W_up → modifica coeficientes de clave]     │
         │                                                                   │
         ▼                                                                   │
HIPÓTESIS CENTRAL: E_MHN por capa discrimina alucinaciones ◄────────────────┘
         │
         ├──► Hofmann et al. (2024)          [OOD detection via E_MHN]
         │    [Justifica E como score de anomalía]
         │
         ├──► Concepto Attractors (2025)     [Dinámica de convergencia]
         │    [Capas altas = convergencia a atractor = baja E]
         │
         ├──► HalluField (2025)              [Baseline termodinámico]
         │    [AUC 0.80–0.83 sobre logits; referencia de comparación]
         │
         ├──► ACT-ViT (2025)                 [Baseline supervisado]
         │    [AUC ~80–89% sobre activaciones completas; techo de referencia]
         │
         ├──► INTRA / Vazhentsev (2026)      [Baseline no supervisado]
         │    [AUC ~77.7%; confirma señal en capas intermedias]
         │
         └──► Conv-CoA (2026)                [Hopfield en QA factual]
              [Motivación aplicada; mismo dominio que TriviaQA]
```

---

## 11. Contraste de Resultados y Discusión Teórica

### 11.1 Señal en logits vs. estados internos

Existe una tensión empírica entre HalluField (que opera sobre logits y reporta AUC 0.80–0.83) y INTRA (que usa estados internos y reporta AUC 0.77–0.78 en el mismo dominio). Sin embargo, INTRA no accede al espacio **MLP específicamente** — usa los estados del residual stream completo. La hipótesis de `hopfield_llm` es más fuerte: la señal está localizada en la componente MLP del residual stream, interpretable como recuperación de memoria.

La comparación correcta no es "logits vs. estados" sino "¿qué componente del estado interno es más informativa?" Vazhentsev et al. [2026] muestran que los estados de capas medias-altas son más discriminativos (Figura 2 del paper). Ramsauer et al. [2021] y Geva et al. [2021] sugieren que es en las capas altas donde los patrones MLP son más semánticamente ricos. La convergencia de estas dos líneas apoya examinar capas $\ell \in [14, 28]$ de Qwen2.5-1.5B.

### 11.2 Supervisado vs. no supervisado

ACT-ViT [Bar-Shalom et al., 2025] y INTRA [Vazhentsev et al., 2026] requieren etiquetas de entrenamiento. HalluField [Vu et al., 2025] y `hopfield_llm` son fundamentalmente no supervisados en su componente core — las métricas de energía se computan directamente de los pesos y activaciones del modelo.

La ventaja de `hopfield_llm` es su **interpretabilidad teórica**: cada métrica (energía, entropía de softmax, activación máxima, delta energía) corresponde a un concepto bien definido en el marco MHN. Esto permite generar hipótesis falsificables sobre el comportamiento del modelo, no solo clasificar correctamente.

### 11.3 La contribución diferencial respecto a HalluField

HalluField [Vu et al., 2025] es el trabajo más cercano en espíritu: ambos usan un formalismo físico (termodinámico vs. energético), ambos son computacionalmente eficientes, ambos no requieren fine-tuning. Las diferencias son:

| Dimensión | HalluField | `hopfield_llm` |
|-----------|-----------|----------------|
| **Señal** | Logits de salida ($P(\text{token})$) | Activaciones MLP intermedias |
| **Marco** | Termodinámica: $F = U - TS$ | MHN: $E = -\text{lse}(\beta, W_1^T \xi)$ |
| **Perturbación** | Temperatura de muestreo $T$ | Divergencia prefill vs. generación (KL) |
| **Localización** | Sin resolución por capa | Per-layer AUROC + regresión logística |
| **Interpretabilidad** | Estabilidad del vocabulario | Estabilidad de recuperación de memoria |
| **Forward passes** | $\geq 2$ (base + perturbado) | 1 (con hooks) |

### 11.4 Síntesis sobre SwiGLU y la sonda MHN

La adopción de SwiGLU en Qwen2.5 [Shazeer, 2020] implica que la coeficiente de activación de la clave $i$ no es simplemente $\text{ReLU}(x \cdot k_i)$ sino $\text{Swish}(x W_{gate,i}) \cdot (x W_{up,i})$. Esto tiene consecuencias para la energía MHN:

$$a_i = \text{Swish}(x W_{gate}^{:,i}) \cdot (x W_{up}^{:,i})$$

$$y = \sum_i a_i \cdot W_{down}^{i,:}$$

El banco de claves en la sonda debe ser $W_{up}^T$ (proyección directa, sin compuerta), y la compuerta $W_{gate}$ actúa como modulador de la "temperatura" efectiva de recuperación — exactamente la modulación que el parámetro $\beta$ controla en el MHN teórico. Esto sugiere que capas con mayor actividad de compuerta ($\|W_{gate}^T x\|$) tendrán energía MHN más discriminativa.

### 11.5 Sobre la ortogonalidad y la capacidad del detector

Gupta et al. [2025] muestran que para tokens sujeto en recall factual, la ortogonalidad del banco de claves cae. En el formalismo MHN, baja ortogonalidad implica patrones no bien separados, lo que produce distribuciones softmax más difusas (alta entropía) y menor concentración en el patrón recuperado (baja energía relativa al patrón correcto). Esto crea exactamente el perfil de anomalía que `hopfield_llm` intenta detectar: durante alucinación, $\xi$ (activación MLP del token de respuesta) no encuentra un patrón bien separado en el banco $W_1$, resultando en alta entropía de la distribución $p = \text{softmax}(\beta W_1^T \xi)$.

---

## 12. Implicaciones para la Validación del Código

### 12.1 Corrección de la energía MHN (M1)

La energía debe incluir los cuatro términos de Ramsauer et al. [2021]:

$$E = -\text{lse}(\beta, W_1^T \xi) + \frac{1}{2}\|\xi\|^2 + \frac{1}{\beta}\log N + \frac{M}{2}$$

Colapsar a solo $-\text{lse}$ elimina la corrección de norma y capacidad. El término $M/2$ (máximo de normas al cuadrado de los patrones) es especialmente importante cuando los vectores de $W_1$ tienen normas heterogéneas — lo que es esperable en modelos SwiGLU donde la norma de las columnas de $W_{up}$ varía por especialización semántica.

### 12.2 Espacio de banco vs. espacio de consulta (M1, secundario)

La sonda secundaria usando `down_values` (banco) y `mlp_output` (hook) corresponde al espacio de **valores**: $V = W_{down}$, $\xi_{val} = y = \sum_i a_i W_{down}^{i,:}$. Este espacio tiene interpretación LAM según Gupta et al. [2025] — la segunda matriz MLP exhibe mayor ortogonalidad genérica. La energía en espacio de valores captura si la *salida* de la MLP corresponde a patrones de valor bien consolidados, complementando la sonda en espacio de claves.

### 12.3 La métrica KL de drift (M2)

La divergencia KL entre la distribución $p_{gen}^{(\ell)} = \text{softmax}(\beta W_1^{(\ell)T} \xi_{gen})$ (durante generación) y $p_{pre}^{(\ell)} = \text{softmax}(\beta W_1^{(\ell)T} \xi_{pre})$ (durante prefill) captura el **desalineamiento de patrones de recuperación** entre la pregunta y la respuesta. La hipótesis es que durante alucinación, el modelo "cambia de región de memoria" de forma abrupta: los patrones recuperados durante la generación divergen de los patrones consultados durante el prefill de la pregunta. Esta idea conecta directamente con el concepto de atractor de Anón. [2025]: una alucinación puede corresponder a convergencia hacia un atractor incorrecto.

### 12.4 Restricción de scores al subconjunto de tokens de respuesta (M3)

El análisis debe restringirse al span de respuesta generada, no al prefill. Vazhentsev et al. [2026] y Bar-Shalom et al. [2025] coinciden en que las señales más discriminativas aparecen en los tokens de respuesta. La lógica teórica: durante el prefill, el modelo procesa la pregunta de entrada (tokens de referencia), donde los patrones MLP corresponden al texto de la pregunta; durante la generación de respuesta, el modelo debe recuperar conocimiento factual — es aquí donde la energía MHN diverge si el conocimiento es incierto o ausente.

---

## 13. Conclusión del Marco Teórico

Esta investigación se sitúa en la intersección de tres líneas de trabajo:

1. **Interpretabilidad de LLMs** [Geva et al., 2021; Gupta et al., 2025]: Las capas MLP almacenan memoria paramétrica como memorias clave-valor.
2. **Redes de Hopfield Modernas** [Ramsauer et al., 2021; Hofmann et al., 2024]: La energía MHN es un detector natural de anomalías en recuperación de patrones.
3. **Detección de alucinaciones vía estados internos** [Bar-Shalom et al., 2025; Vazhentsev et al., 2026; Vu et al., 2025]: Las representaciones internas contienen señal discriminativa para alucinación, superior a los logits de salida.

La contribución de `hopfield_llm` es conectar estas tres líneas mediante una sonda interpretable, sin entrenamiento de clasificador, con localización por capa, operando sobre el espacio semánticamente rico de las activaciones MLP bajo el formalismo MHN. Los resultados de AUROC por capa y la regresión logística sobre features MHN proveen tanto diagnóstico cuantitativo (¿qué capas discriminan mejor?) como interpretación cualitativa (¿qué aspecto de la dinámica de recuperación difiere entre respuestas correctas y alucinadas?).

---

## Referencias

- Bar-Shalom, G., Frasca, F., Galron, Y., Ziser, Y., & Maron, H. (2025). Beyond Token Probes: Hallucination Detection via Activation Tensors with ACT-ViT. *NeurIPS 2025*.
- Dauphin, Y. N., Fan, A., Auli, M., & Grangier, D. (2017). Language Modeling with Gated Convolutional Networks. *ICML 2017*.
- Geva, M., Schuster, R., Berant, J., & Levy, O. (2021). Transformer Feed-Forward Layers Are Key-Value Memories. *EMNLP 2021*, 5484–5495.
- Gupta, A., Sindhu, N., & Anumanchipalli, G. (2025). How Linearly Associative Are Memories in Large Language Models? *New Frontiers in Associative Memory Workshop, ICLR 2025*.
- Hofmann, C., Schmid, S., Lehner, B., Klotz, D., & Hochreiter, S. (2024). Energy-based Hopfield Boosting for Out-of-Distribution Detection. *NeurIPS 2024*.
- Ramsauer, H., Schäfl, B., Lehner, J., Seidl, P., Widrich, M., Adler, T., ... & Hochreiter, S. (2021). Hopfield Networks Is All You Need. *ICLR 2021*.
- Shazeer, N. (2020). GLU Variants Improve Transformer. *arXiv:2002.05202*.
- Vazhentsev, A., Marina, M., Moskovskiy, D., Pletenev, S., Seleznyov, M., Salnikov, M., ... & Moskvoretskii, V. (2026). Leveraging LLM Parametric Knowledge for Fact Checking without Retrieval. *arXiv:2603.05471*.
- Vu, M., Tran, B. K., Shah, S. A., Zollicoffer, G., Hoang, X. N., & Bhattarai, M. (2025). HalluField: Detecting LLM Hallucinations via Field-Theoretic Modeling. *arXiv:2509.10753*.
- Anónimo (2025). Concept Attractors in LLMs and their Applications. *Submitted to NeurIPS 2025*.
- Anónimo (2026). Conv-CoA: Open-Domain Question Answering via Conversational Chain-of-Action with Hopfield Retriever. *Under review, ICLR 2026*.
