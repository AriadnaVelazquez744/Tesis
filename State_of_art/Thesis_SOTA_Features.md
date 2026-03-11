# Características del estado del arte de la tesis

## I. Categoría: Aproches (Arquitecturas)

* **Neural Propose → Symbolic Verify (Sequential Pipeline):** Arquitectura de flujo secuencial donde un modelo neuronal (ej. LLM) genera candidatos a solución o pasos de razonamiento, los cuales son validados por un verificador simbólico formal para garantizar su corrección\.  
* **Symbolic Scaffold → Neural Fill (Structural Decomposition):** Enfoque donde un componente simbólico genera una estructura de alto nivel (como invariantes o planes) y la red neuronal se encarga de completar los detalles de bajo nivel o tareas preceptuares dentro de ese marco\.  
* **Joint Neural+Symbolic Inference (End-to-End Integration):** Integración profunda donde las restricciones simbólicas se embeben directamente en la función de pérdida o en el procedimiento de inferencia, permitiendo que el sistema sea diferenciable de extremo a extremo\.  
* **Multi-Agent or Modular Systems (Parallel Integration):** Sistema compuesto por diversos agentes o módulos especializados (algunos simbólicos y otros neuronales) que colaboran en paralelo e intercambian resultados para resolver subtareas complejas\.

## II. Categoría: Integración (Taxonomía de Kautz)

* **Type 1 (Simple I/O):** Sistemas de aprendizaje profundo estándar donde las entradas y salidas son símbolos (ej. traducción de texto), pero el procesamiento interno es puramente neuronal\.  
* **Type 2 (Subroutines):** Sistemas híbridos donde una red neuronal actúa como una subrutina para ayudar a un motor de búsqueda o razonador simbólico (ej. AlphaGo usando MCTS con redes de evaluación)\.  
* **Type 3 (Collaboration):** Sistemas donde un componente neuronal (ej. percepción visual) y un sistema simbólico (ej. motor de consulta) interactúan como corrutinas para lograr una tarea común\.  
* **Type 4 (Symbolic Compilation):** Enfoque donde el conocimiento simbólico (reglas *if-then*) se traduce o compila directamente en la arquitectura inicial y los pesos de una red neuronal\.  
* **Type 5 (Symbolic Loss):** Integración donde las reglas lógicas se mapean en *embeddings* que funcionan como restricciones suaves o regularizadores dentro de la función de pérdida de la red neuronal\.  
* **Type 6 (Full Hybridization):** El objetivo final de la IA general: un sistema capaz de realizar razonamiento simbólico verdadero (incluyendo razonamiento combinatorio) enteramente dentro de un motor neuronal\.

## III. Categoría: Learning & Strategy Mechanisms

* **Thought (CoT, ToT, GoT, DoT):** Capacidad de un modelo para construir pasos intermedios de pensamiento que forman un puente lógico estructurado y coherente entre la entrada y la salida final.  
* **Chain-of-Action:** Secuencia de acciones ejecutadas por un agente (a menudo multimodal) basadas en la percepción continua y el razonamiento sobre el estado del entorno\.  
* **Training (SFT/RFT):** Procesos de optimización de parámetros; SFT (*Supervised Fine-Tuning*) usa pares instrucción-respuesta, mientras RFT (*Reinforced Fine-Tuning*) usa señales de recompensa para fortalecer el razonamiento. Es un cambio estructural y permanente en la memoria paramétrica\.  
* **use Knowledge Graph (KG):** Representación estructurada de hechos en forma de triples (Sujeto, Predicado, Objeto) que capturan relaciones semánticas entre entidades\.  
* **use Search/Exploration methods:** Algoritmos (ej. MCTS, búsqueda en haz) que navegan por espacios de decisión o rutas de pensamiento para optimizar el resultado\.  
* **Symbolic Formulation:** Traducción de descripciones en lenguaje natural a lenguajes formales como Lógica de Primer Orden (FOL), Lógica Proposicional o lenguajes de programación.  
* **External Deterministic Solver:** Herramientas externas (ej. Z3, Prover9) que realizan inferencias lógicas exactas basándose en algoritmos deterministas para evitar alucinaciones\.  
* **In-context Learning:** Capacidad de los modelos para aprender y adaptarse a nuevas tareas a partir de unos pocos ejemplos proporcionados directamente en el *prompt*, sin actualizar sus pesos.  
* **Self-refinement:** Proceso iterativo donde el modelo utiliza mensajes de error o su propia reflexión crítica para corregir y mejorar sus producciones iniciales\.  
* **Rule-base/LLM-base Interpreter:** Módulo que traduce los resultados simbólicos crudos de un solver o motor lógico de vuelta a un lenguaje natural entendible por el humano\.  
* **LLM problem representation**
* **Behavior Learning:** Se refiere a la adquisición de políticas de acción y estrategias de comportamiento, comúnmente mediante Aprendizaje por Refuerzo (RL), para que el agente interactúe de forma autónoma en entornos dinámicos\.  
* **Symbolic Agent (Search rules):** Agente que busca de forma autónoma reglas lógicas o rutas de conocimiento dentro de una base interna para apoyar el razonamiento\.  
* **Conocimiento de Procesos (Process Knowledge):** A diferencia de un Grafo de Conocimiento (que es estático), el conocimiento de procesos impone un orden y un flujo conceptual (ej. protocolos médicos o reglas de diagnóstico). Es el "pegamento" que permite que la IA entienda la secuencia lógica de lo que se le pide\.  

## IV. Categoría: Memory & Meta-Cognition

* **Memory: Parametric:** Conocimiento almacenado de forma implícita y distribuida dentro de los pesos del modelo neuronal a través del proceso de entrenamiento.  
* **Memory: Non-Parametric:** Memoria almacenada en medios externos (ej. bases de datos vectoriales, RAG) que permite actualizaciones rápidas y manejo de información de "larga cola" sin reentrenar el modelo\.  
* **Meta-Cognition / Self-Monitoring:** Proceso de orden superior que permite al sistema monitorear, evaluar y ajustar sus propios procesos de razonamiento y aprendizaje. Es el "controlador" que se sitúa por encima de las tareas para decidir qué sistema (S1 o S2) debe manejar la tarea, monitorear recursos y ajustar la estrategia global antes o durante el proceso.  
* **Active Forgetting:** Proceso deliberado de eliminar redundancias o información obsoleta para optimizar el espacio de almacenamiento y mejorar la eficiencia de la recuperación\.  
* **Omnidirectional/Bidirectional Inference:** Capacidad de un sistema lógico (ej. LNN) para realizar inferencia en cualquier dirección (de premisas a conclusión y viceversa) a través de los nodos de la red.  
* **Bounded Rationality:** Concepto que establece que el razonamiento de un sistema opera bajo restricciones finitas de tiempo, conocimiento incompleto y capacidad computacional\.  
* **Trace-based Grounding:** Uso de los pasos introspectivos ("traza") de un proceso de razonamiento para instruir y corregir al modelo, asegurando la fidelidad lógica\.  
* **Razonamiento de Teoría de la Mente (Theory of Mind Reasoning):** Capacidad del sistema para atribuir estados mentales (creencias, deseos, metas) a otros agentes para predecir y explicar su comportamiento\.  

## V. Categoría: Herramientas y Estructuras de Agentes

* **External Tools:** Integración de software, interfaces de programación (APIs) o recursos externos (como calculadoras o motores de búsqueda) que permiten al modelo extender sus límites de conocimiento factual y ejecutar tareas especializadas fuera de su arquitectura base 1-3.  
* **Programming Code:** Uso de lenguajes de programación (como Python o Lisp) como carriers de planificación o síntesis de programas, permitiendo que el razonamiento sea expresado en una sintaxis ejecutable y verificable 4-6.  
* **Symbolic Reasoners:** Motores de inferencia deterministas (como solvers SAT, CSP o probadores de teoremas FOL) que aplican reglas lógicas estrictas para derivar conclusiones exactas a partir de premisas formalizadas.
* **Multimodal Agents:** Sistemas de IA capaces de procesar y generar información integrando múltiples modalidades sensoriales (texto, imagen, video y audio) para una comprensión holística del entorno.  
* **Multi-agent Systems:** Arquitecturas compuestas por múltiples módulos o agentes especializados que colaboran mediante protocolos de comunicación y debate para resolver tareas complejas que un modelo monolítico no podría abordar con eficiencia.

## VI. Categoría: Racionalidad

* **Axioms of Rationality:** Principios matemáticos y lógicos (como la transitividad, la negación y la implicación) que rigen la coherencia interna de un sistema de decisión, asegurando que las elecciones del agente sigan reglas sólidas de inferencia.  
* **Information Grounding:** Conexión semántica entre símbolos abstractos y datos sensoriales del mundo real (ej. vincular la palabra "cubo" con un objeto visual). Proceso de anclar símbolos abstractos y representaciones semánticas en datos del mundo real o contextos específicos, mitigando la desconexión entre el lenguaje y la realidad física\.  
* **Logical Consistency:** Requisito de que las respuestas del modelo no se contradigan entre sí ni violen los principios lógicos o las bases de conocimiento establecidas\.  
* **Invariance from Irrelevant Context:** Capacidad de un modelo para mantener su desempeño y juicios constantes a pesar de la presencia de información distractora o cambios superficiales en el formato de la instrucción que no alteran la lógica del problema.  
* **Orderability of Preference:** Capacidad del sistema para jerarquizar múltiples soluciones candidatas basándose en funciones de utilidad, restricciones de presupuesto o preferencias aprendidas del comportamiento humano\.  
* **Explicit Intermediate Representation (Transparency):** Uso de formatos legibles (como grafos o lógica formal) durante los pasos intermedios del procesamiento para que el razonamiento sea auditable.  

## VII. Categoría: Evaluación y Sesgos de Decisión

* **Hallucinations:** Fenómeno en el que el modelo genera contenido que parece plausible pero es fácticamente incorrecto, lógicamente inconsistente o no está respaldado por los datos de entrada\.  
* **modify inatruction templates**
* **parapphrasing task descriptions**
* **alter the order of examples**
* **change prompt language**
* **Risk Aversion / Loss Aversion:** Sesgos en la toma de decisiones donde el sistema prioriza evitar resultados negativos o inciertos por encima de maximizar ganancias potenciales, emulando patrones psicológicos humanos bajo incertidumbre\.  
* **Small Probability under Uncertainty:** Evaluación de cómo el sistema procesa eventos de baja frecuencia o información de "larga cola", donde la falta de datos estadísticos requiere un razonamiento basado en reglas más que en patrones\.

## VIII. Categoría: Parámetros y Baselines de Evaluación

* **Correctness and Soundness:** Métricas que garantizan la ausencia de falsos positivos y aseguran que cada paso deductivo sea válido dentro del sistema lógico adoptado.  
* **Accuracy and Success Rate:** Medida cuantitativa del emparejamiento exacto entre la salida del modelo y la solución objetivo en tareas de razonamiento estructurado\.  
* **Efficiency and Scalability:** Optimización del tiempo de ejecución y el uso de recursos de memoria, permitiendo que los métodos de razonamiento simbólico (usualmente costosos) funcionen en aplicaciones de gran escala.  
* **Interpretability and Explanation Quality:** Evaluación de la capacidad del sistema para extraer representaciones compactas y fieles que justifiquen una decisión de manera comprensible para el usuario \.  
* **Generality and Coverage:** Alcance del sistema para manejar una amplia variedad de tareas y tipos de lógica (temporal, espacial, modal) sin requerir ajustes manuales específicos para cada caso.  
* **Systematic Generalization (OOD):** Capacidad de aplicar reglas aprendidas a ejemplos que están fuera de la distribución estadística del entrenamiento, demostrando un entendimiento de los principios subyacentes.

## IX. Baselines Comparativos

* **Neural-only methods:** Modelos basados exclusivamente en aprendizaje profundo (ej. Transformers estándar) utilizados para contrastar la falta de rigor lógico y la susceptibilidad a alucinaciones\.  
* **Symbolic-only methods:** Sistemas tradicionales basados en reglas (GOFAI) utilizados para comparar la fragilidad ante datos ruidosos y la incapacidad de aprendizaje autónomo.  
* **Heuristic explainers:** Métodos de explicabilidad post-hoc (como LIME o SHAP) que se comparan con las explicaciones intrínsecas de los modelos neuro-simbólicos para medir la fidelidad y la minimidad de la explicación\.
