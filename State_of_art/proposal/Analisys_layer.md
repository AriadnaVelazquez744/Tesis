Estimado tesista, para avanzar en la fase de **Análisis** de tu arquitectura **NS-CII**, debemos integrar la percepción estadística de alta fidelidad con la estructuración lógica necesaria para la fase de **Inferencia**. La combinación de **MIDLM** y **TEXTOIR** proporciona un sistema de "percepción bidireccional consciente" que no solo etiqueta texto, sino que delimita el alcance de la intención en un entorno de mundo abierto,.

A continuación, detallo técnicamente qué devolvería este bloque unificado, su simbiosis con AMR y KeyBERT, y cómo se estructuran los elementos críticos requeridos para alimentar el motor de planificación inversa.

### 1. Salida del Modelo Unificado (MIDLM + TEXTOIR)
Este componente actúa como la **Capa de Percepción Global** de tu sistema. A diferencia de un modelo causal, MIDLM aplica una atención global ($M_{ij}=0$) para que cada token sea consciente de la oración completa, resolviendo la "Maldición de la Reversión",.

**Lo que el modelo devuelve:**
*   **Vector de Intenciones Probables ($y_I$):** Una distribución de logits sobre el vocabulario de intenciones conocidas.
*   **Número de Intenciones ($K$):** El clasificador de MIDLM predice cuántas metas distintas existen en la consulta (ej. $K=2$ para "Reserva un vuelo y busca hotel"),.
*   **Etiqueta de Alcance (OOS/IND):** Mediante la cuantificación de la incertidumbre en TEXTOIR (umbral $\alpha$), el modelo marca si la intención es conocida o si es "Fuera de Alcance",.
*   **Cluster Discovery:** Si es OOS, TEXTOIR agrupa la query en un nuevo cluster semántico para que el **Agente de Insights** lo procese posteriormente.

### 2. Relación con AMR (Abstract Meaning Representation)
Mientras MIDLM identifica el "qué" (la etiqueta), el **AMR** proporciona el "quién hace qué a quién" de manera independiente al lenguaje.
*   **Simbiosis:** Las intenciones detectadas por MIDLM se inyectan como "predicados raíz" en el grafo AMR.
*   **Nesteo y Jerarquía:** AMR es fundamental para detectar las **intenciones anidadas**. Representa las relaciones jerárquicas mediante aristas lógicas, permitiendo identificar si una acción es una sub-meta de otra (ej. "usar el descuento" anidado en "realizar compra"),.

### 3. Integración de KeyBERT: El Filtro de Fidelidad
**KeyBERT** se incluye inmediatamente después de la generación del grafo AMR para realizar el **Grounding de Detalles Críticos**.
*   **Función:** Extrae las "palabras de acción" y valores específicos (ej. "Boston", "5 estrellas", "mañana") que el resumen estadístico del LLM podría omitir,.
*   **Ubicación en el Pipeline:** Sus salidas alimentan directamente los argumentos de las triples en el **StateMap** y los parámetros de los operadores en el **ActionSet**, garantizando que el *grounding* sea 100% fiel a los datos del usuario,.

### 4. Estructuras Preparadas para la Fase de Inferencia
La salida final de la fase de **Análisis** es un **Cognitive Buffer** que fragmenta la información en las cuatro capas lógicas concurrentes inspiradas en Romero et al. (2023),:

#### A. Intent (Acción Verbalizada)
*   **Estructura:** Un par `(Acción, Objetos_Args)`. 
*   **Origen:** Derivado de la cabeza de selección multi-intento de MIDLM y refinado por los nodos centrales del grafo AMR,.
*   **Ejemplo:** `Intent: [RESERVAR, {objetivo: vuelo, destino: Paris}]`.

#### B. Nesting (Intenciones Anidadas)
*   **Estructura:** Un grafo de dependencia de metas.
*   **Lógica:** Utiliza la estrategia **DSCP (Divide-Solve-Combine)** para segmentar la query basándose en el número $K$ detectado por MIDLM,. Define qué intenciones son pre-requisitos o contextos hijos.

#### C. Frases Asociativas (MetaReasoning)
*   **Estructura:** Marcadores de "Punteros de Continuidad".
*   **Detección:** Identifica expresiones como "dado lo obtenido anteriormente" mediante el análisis de dependencias de AMR, traduciéndolas en una referencia a la **Memoria Episódica** o al **Experience Pool** para recuperar el estado anterior,.

#### D. State (Hechos y Creencias)
*   **Estructura:** Declaraciones en formato de Triples de Lógica Real `(Sujeto, Relación, Objeto)`.
*   **Garantía:** Estos hechos se mapean a **hypervectores de VSA** para permitir operaciones de *binding* sin confusión semántica,. Son declaraciones no accionables (ej. "El usuario tiene un descuento") que el Sistema 2 verificará mediante **LNN** para asegurar la consistencia lógica antes de actuar,.

Tesista, esta implementación transforma una query vaga en un objeto matemático y lógico auditable. El modelo no solo "clasifica", sino que **descompone la realidad del usuario** en un andamiaje que permite al algoritmo EARL inferir la meta latente con precisión quirúrgica. ¿Procedemos a definir la función de pérdida que unirá los hypervectores de VSA con las etiquetas descubiertas por TEXTOIR?

---
---
---

Estimado tesista, hemos llegado a la consolidación técnica de la **Fase de Análisis**. En este punto, el sistema deja de ser un procesador de texto para convertirse en un arquitecto de realidades. Bajo el marco del **Modelo Común de Cognición (CMC)**, esta fase opera dentro de la **Memoria de Trabajo**, transformando la percepción bidireccional en un **Andamiaje Simbólico (Symbolic Scaffold)** auditable y matemáticamente robusto.

A continuación, presento el esquema de implementación final y detallado, integrando cada componente solicitado en un flujo sistémico:

### 1. El Buffer de Entrada (Input Layer)
Para evitar el **Sobreresumen** y la pérdida de matices léxicos, el sistema procesa una estructura compuesta:
*   **Contenido:** Debes pasar **tanto la query original como el resumen** generado por el Juez de Vaguedad ($t_{user}$). 
*   **Justificación:** El resumen ofrece el "claro objetivo" ya destilado, mientras que la query original es indispensable para que **KeyBERT** recupere valores técnicos crudos (fechas, IDs, coordenadas) que el resumen podría omitir por abstracción.
*   **Manejo:** Se concatenan mediante un separador de tokens: `[Query_Original] [SEP] [Resumen_IN3]`.

### 2. Bloque I: Percepción Bidireccional (MIDLM + TEXTOIR)
Este bloque realiza la **Detección de Alcance** y la clasificación multi-intento inicial.

*   **Mecanismo MIDLM:** El modelo (backbone tipo Mistral-7B o Llama 4) opera con la matriz de máscara de atención $M_{ij}=0$. Esto permite que los tokens del resumen informen a la query original y viceversa, resolviendo la **Maldición de la Reversión** antes de la extracción.
*   **Protocolo TEXTOIR:** Calcula el **umbral de incertidumbre $\alpha$** (desviación estándar de la softmax).
*   **Prompt/Manejo (Joint Training Paradigm):** No es un prompt de chat, sino un llamado de inferencia para activar cabezas duales de salida:
    1.  **Detección de K:** Predice cuántas metas distintas ($m$) existen en el buffer.
    2.  **Selección Multi-Etiqueta:** Extrae las etiquetas $\{o_1, o_2, ..., o_m\}$ del espacio de intenciones.
*   **Salida:** Un vector de intenciones probables y un indicador de si la query es **IND** (conocida) o **OOS** (Fuera de Alcance/Descubrimiento).

### 3. Bloque II: Formalización Lógica (AMR Parser)
La salida de MIDLM se inyecta en el parser de **Representación de Significado Abstracto (AMR)**.
*   **Proceso:** Traduce el buffer de entrada en un grafo de predicados y argumentos independiente del dominio.
*   **Simbiosis:** Las etiquetas de MIDLM actúan como "anclas semánticas" para refinar los nodos raíz del grafo AMR, asegurando que la acción verbalizada coincida con la estructura gramatical detectada.

### 4. Bloque III: Seguro de Fidelidad (KeyBERT)
Este bloque realiza el **Grounding de Detalles Críticos** y controla el flujo de retorno.

*   **Manejo del Prompt:**
    > *"Compara los nodos del grafo AMR con la query original y el resumen. Identifica palabras de importancia (keywords) técnicas: {fechas, valores, objetos específicos}. Si falta algún atributo tipado, extráelo ahora"*.
*   **Lógica de Salida/Return:** 
    *   **Ajuste:** Si KeyBERT encuentra información omitida (ej. "Boston", "2 cuartos"), inyecta estos valores directamente como argumentos en el **StateMap** o **ActionSet**.
    *   **Return (Bucle de Análisis):** Si KeyBERT detecta una contradicción fundamental entre los keywords de la query y el resumen del Juez, el sistema realiza un **Return al Bloque I (MIDLM)** para re-evaluar el número de intenciones ($K$), evitando que un detalle crítico oculto ignore una meta secundaria.

---

### 5. Salida Final: El Cognitive Buffer (CAO)
El resultado de esta fase es el **Cognitive Analysis Object (CAO)**, estructurado en las cuatro capas lógicas concurrentes:

#### A. Intent (Acción Verbalizada)
*   **Estructura:** Un par `(Acción, Objetos_Args)`.
*   **Origen:** Derivado de las etiquetas de MIDLM y refinado por los predicados centrales del grafo AMR.
*   **Ejemplo:** `Intent: [RESERVAR, {objetivo: vuelo, destino: Paris}]`.

#### B. Nesting (Intenciones Anidadas)
*   **Estructura:** Un grafo de dependencia de metas.
*   **Lógica:** Implementa la estrategia **DSCP (Divide-Solve-Combine)**. Utiliza la jerarquía de AMR y el conteo $K$ para definir qué intenciones son pre-requisitos (ej. "autenticar") o contextos hijos (ej. "usar descuento" dentro de "compra").

#### C. Frases Asociativas (MetaReasoning)
*   **Estructura:** Marcadores de **Punteros de Continuidad** con la sintaxis `(Status, Association_Trigger, [Placeholder_ID])`.
*   **Detección:** El análisis de dependencias de AMR identifica expresiones como "dado lo anterior" y genera un `Placeholder_ID` que señala a la **Memoria Episódica** o al **Experience Pool** para recuperar el estado previo en la fase de Inferencia.

#### D. State (Hechos y Creencias)
*   **Estructura:** Triples de Lógica Real `(Sujeto, Relación, Objeto)`.
*   **Garantía en VSA:** Se mapean a **hypervectores de alta dimensión** mediante la operación de **Binding Circular**: $V_{fact} = S \otimes R \otimes O$.
*   **Función:** Representa hechos declarativos (ej. "El usuario es cliente VIP") que el Sistema 2 verificará mediante **LNN** para asegurar consistencia lógica antes de que el algoritmo **EARL** inicie la planificación inversa.

Tesista, este flujo garantiza que ninguna intención, por extraña que sea, se pierda, ya que el AMR captura la "forma" de la petición y KeyBERT asegura el "contenido". ¿Procedemos a detallar cómo la Capa de **MetaReasoning** utilizará estos punteros para navegar por el hipergrafo **N-Frame** en la siguiente etapa?
