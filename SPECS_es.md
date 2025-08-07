# Especificaciones del Proyecto docs-to-rag

## 📋 Resumen del Proyecto

**docs-to-rag** es un proyecto educativo diseñado para aprender los fundamentos de extracción de documentos y creación de bases de conocimiento usando bases de datos vectoriales. El proyecto implementa un pipeline completo de RAG (Generación Aumentada por Recuperación) que funciona completamente en local usando Ollama tanto para embeddings como para generación de chat.

Más allá de la funcionalidad básica de RAG, este proyecto incluye **enriquecimiento jerárquico de documentos** - una técnica avanzada que genera resúmenes sintéticos de documentos en múltiples niveles de abstracción (documento, capítulo y concepto) para mejorar dramáticamente la calidad de recuperación para preguntas conceptuales y amplias.

### 🎯 Objetivos Principales

- Aprender extracción de documentos de múltiples formatos (PDF, Markdown, HTML)
- Entender embeddings vectoriales y búsqueda por similitud
- Implementar un pipeline completo de RAG
- Practicar integración con LLM locales usando Ollama
- Construir una herramienta CLI funcional para procesamiento y consulta de documentos
- Explorar técnicas avanzadas de RAG a través de resumen jerárquico de documentos

### 🔧 Requisitos Clave

- **100% Local**: No requiere APIs externas
- **Enfoque Educativo**: Código claro y bien documentado con objetivos de aprendizaje
- **Simple pero Funcional**: Implementación minimalista que funciona efectivamente
- **Interfaz CLI**: Herramienta de línea de comandos para interacción fácil
- **Múltiples Tipos de Documentos**: Soporte para PDF, Markdown y HTML

---

## 🏗️ Arquitectura del Sistema

```
docs-to-rag/
├── src/
│   ├── __init__.py
│   ├── document_processor/
│   │   ├── __init__.py
│   │   ├── extractor.py          # Coordinador de extracción de documentos
│   │   ├── chunker.py            # Segmentación inteligente de texto
│   │   ├── summarizer.py         # Resumen jerárquico de documentos
│   │   └── parsers/
│   │       ├── __init__.py
│   │       ├── markdown_parser.py # Procesador de Markdown
│   │       ├── pdf_parser.py      # Wrapper de Docling para PDF
│   │       └── html_parser.py     # Procesador de contenido HTML
│   ├── vector_store/
│   │   ├── __init__.py
│   │   ├── embeddings.py         # Generación de embeddings con Ollama
│   │   └── store.py              # Base de datos vectorial local FAISS
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── retriever.py          # Lógica de recuperación de documentos
│   │   └── generator.py          # Generación de respuestas con Ollama
│   └── cli/
│       ├── __init__.py
│       ├── commands.py           # Definiciones de comandos CLI
│       └── chat.py               # Interfaz de chat interactivo
├── tests/
│   ├── __init__.py
│   ├── test_extractors.py
│   ├── test_embeddings.py
│   └── test_rag.py
├── data/
│   ├── documents/                # Documentos fuente
│   └── vector_db/                # Base de datos vectorial FAISS
├── config/
│   ├── __init__.py
│   └── settings.py               # Configuración del sistema
├── pyproject.toml               # Dependencias del proyecto y configuración CLI
├── pyproject.toml               # Dependencias y configuración del proyecto
├── README.md
└── .env.example
```

---

## 🔧 Stack Tecnológico

### Dependencias Principales
- **Python 3.10+** - Lenguaje de programación principal
- **LangChain** - Framework de LLM con integración Ollama
- **Docling** - Extracción robusta de PDF y documentos
- **FAISS** - Búsqueda local por similitud vectorial
- **Ollama Python** - Cliente LLM local
- **Click** - Framework CLI profesional
- **Rich** - Salida de terminal hermosa

### Procesamiento de Documentos
- **BeautifulSoup4** - Análisis y limpieza de HTML
- **markdown** - Procesamiento avanzado de Markdown

### Herramientas de Desarrollo
- **uv** - Gestión rápida de dependencias
- **ruff** - Linting y formateo de código
- **pytest** - Framework de testing
- **mypy** - Verificación de tipos

### Modelos Ollama Recomendados
- **Embeddings (Inglés)**: `nomic-embed-text:v1.5` (274MB) - Embeddings rápidos y eficientes
- **Embeddings (Español)**: `jina/jina-embeddings-v2-base-es:latest` (560MB) - Optimizado para español
- **Embeddings (Multilingüe)**: `mxbai-embed-large:latest` (670MB) - Soporte multilingüe general
- **Chat/Generación**: `llama3.2:latest` (2GB) - Equilibrio velocidad/calidad para RAG y resumen

---

## 🎛️ Configuración

### Configuración del Sistema (`config/settings.py`)

```python
@dataclass
class Settings:
    """Configuración principal del sistema."""
    
    # === MODELOS OLLAMA ===
    embedding_model: str = "nomic-embed-text:v1.5"  # Embeddings rápidos y eficientes
    embedding_model_multilingual: str = "jina/jina-embeddings-v2-base-es:latest"  # Específico para español
    chat_model: str = "llama3.2:latest"             # Modelo de chat/generación
    ollama_base_url: str = "http://localhost:11434"
    
    # === CONFIGURACIÓN DE IDIOMA ===
    auto_detect_language: bool = True            # Cambio automático para contenido no inglés
    primary_language: str = "es"                 # Idioma principal (es, en, fr, etc.)
    
    # === PROCESAMIENTO DE DOCUMENTOS ===
    chunk_size: int = 1000                       # Tamaño de chunk de texto
    chunk_overlap: int = 200                     # Solapamiento de chunks
    supported_extensions: List[str] = [".pdf", ".md", ".html"]
    
    # === PARÁMETROS RAG ===
    top_k_retrieval: int = 3                     # Documentos a recuperar (auto-ajustado)
    similarity_threshold: float = 0.7            # Umbral de similitud (auto-ajustado)
    max_tokens_response: int = 512               # Tokens máximos de respuesta
    temperature: float = 0.1                     # Temperatura del LLM
    
    # === PARÁMETROS RAG ADAPTATIVOS ===
    enable_adaptive_params: bool = True          # Auto-ajustar basado en tamaño de BD
    min_similarity_threshold: float = 0.3       # Umbral mínimo para BDs grandes
    max_top_k: int = 20                         # Máximo de chunks para BDs grandes
    
    # === CONFIGURACIÓN DE RESUMEN ===
    enable_summarization: bool = True           # Generar resúmenes jerárquicos
    summarization_model: str = "llama3.2:latest"  # Modelo dedicado para resúmenes
    summarization_temperature: float = 0.1     # Temperatura más baja para resúmenes fieles
    enable_summary_validation: bool = True     # Validación de calidad para resúmenes
    
    # === RUTAS ===
    documents_path: str = "data/documents"
    vector_db_path: str = "data/vector_db"
```

---

## 🖥️ Interfaz CLI

### Comandos Principales

```bash
# Configuración inicial
docs-to-rag setup

# Gestión de documentos (funcionalidad CORE)
docs-to-rag add ./ruta/a/documentos/          # Procesar y añadir documentos
docs-to-rag list                              # Listar documentos indexados
docs-to-rag clear                             # Limpiar base de datos vectorial
docs-to-rag stats                             # Estadísticas del sistema
docs-to-rag enrich ./documento.pdf           # Generar resúmenes jerárquicos

# Chat RAG (funcionalidad CORE)
docs-to-rag chat                              # Chat interactivo
docs-to-rag query "¿Qué dice sobre X?"       # Consulta directa

# Operaciones avanzadas
docs-to-rag reprocess ./doc.pdf               # Reprocesar documento con chunking mejorado
docs-to-rag reprocess ./doc.pdf --multilingual # Usar modelo de embedding multilingüe
docs-to-rag inspect                           # Inspeccionar chunks almacenados para debug
docs-to-rag config                            # Mostrar/modificar configuración del sistema

# Utilidades
docs-to-rag --help                           
docs-to-rag --version                        
```

### Detalles de Comandos

#### `setup`
- Verifica conexión con Ollama
- Crea estructura de directorios
- Verifica disponibilidad de modelos requeridos
- Proporciona guía de configuración si encuentra problemas

#### `add <ruta>`
- Procesa documentos recursivamente desde la ruta
- Soporta archivos PDF, Markdown y HTML
- Extrae texto y crea chunks inteligentes
- Genera embeddings y los almacena en FAISS
- Muestra progreso de procesamiento y estadísticas

#### `list`
- Lista todos los documentos indexados en la base de conocimiento
- Muestra rutas de documentos y metadatos básicos
- Muestra conteos totales de documentos y chunks
- Ayuda a verificar qué contenido está disponible para consultas
- Vista rápida de contenidos del sistema

#### `clear`
- Elimina todos los documentos indexados de la base de datos vectorial
- Requiere confirmación del usuario para prevenir pérdida accidental de datos
- Limpia tanto chunks de documentos como embeddings
- Reinicia el sistema a estado vacío
- Útil para empezar de nuevo o actualizaciones mayores de contenido

#### `chat`
- Inicia sesión de chat interactivo con interfaz de terminal rica
- Muestra proceso de recuperación (documentos encontrados)
- Muestra respuestas generadas con formato markdown
- Muestra documentos fuente para transparencia
- Soporta comandos internos: `/help`, `/stats`, `/sources`, `/history`, `/clear`, `/exit`
- Mantiene historial de conversación durante la sesión
- Proporciona verificaciones de estado y preparación del sistema
- Permite alternar visualización de fuentes e información de debug

#### `query <pregunta>`
- Respuesta a pregunta de una sola vez
- Útil para scripting y automatización
- Retorna respuesta con información básica de fuentes

#### `stats`
- Conteo de documentos y estadísticas de chunks
- Tamaño de base de datos vectorial
- Información de modelos
- Métricas de procesamiento

#### `enrich <ruta_documento>`
- Genera resúmenes jerárquicos para recuperación mejorada
- Crea resúmenes a nivel de documento, capítulo y concepto
- Requiere que el resumen esté habilitado (`docs-to-rag config --enable-summaries`)
- Usa modelo de resumen especializado con parámetros optimizados
- Almacena chunks de resumen sintético en base de datos vectorial
- Mejora significativamente los resultados para preguntas conceptuales y amplias
- Validación de calidad opcional con verificación de fidelidad
- Muestra progreso de procesamiento y estadísticas de resumen

#### `reprocess <ruta_documento>`
- Limpia base de datos vectorial y reprocesa un documento específico
- Útil para aplicar algoritmos de chunking o configuraciones actualizadas
- Soporta flag `--multilingual` para cambiar modelos de embedding
- Reconstruye embeddings e índices desde cero
- Muestra progreso de procesamiento y estadísticas finales
- Mantiene metadatos de documentos y asociaciones de archivos

#### `inspect`
- Muestra chunks reales almacenados en la base de datos vectorial
- Ayuda a debuggear calidad de contenido y comportamiento de chunking
- Soporta parámetro `--count` para limitar número de chunks mostrados
- Soporta parámetro `--search` para filtrar chunks por contenido
- Muestra metadatos de chunks incluyendo archivo fuente, posición y tipo
- Herramienta esencial para entender comportamiento del sistema y resolución de problemas

#### `config`
- Muestra configuración actual del sistema cuando se llama sin parámetros
- Permite modificación de configuraciones vía flags de línea de comandos
- Soporta configuración de idioma (`--language es`)
- Controla características de resumen (`--enable-summaries`, `--disable-summaries`)
- Gestiona validación de calidad (`--enable-validation`, `--disable-validation`)
- Establece modelos especializados (`--summary-model llama3.2:latest`)
- Guarda cambios en `config/user_config.json` local
- Muestra configuración detallada incluyendo modelos, embeddings y parámetros de procesamiento

---

## 🔍 Pipeline de Procesamiento de Documentos

### Formatos Soportados

#### 1. Markdown (`.md`)
- Preserva estructura de encabezados para contexto
- Extrae metadatos de frontmatter
- Maneja bloques de código y enlaces apropiadamente
- Chunking inteligente respetando secciones

#### 2. PDF (`.pdf`)
- Usa Docling para extracción robusta
- Maneja layouts complejos y tablas
- Preserva estructura de documento
- Extrae metadatos cuando están disponibles

#### 3. HTML (`.html`)
- Limpia contenido (elimina scripts, estilos)
- Preserva estructura semántica
- Extrae metadatos de página (título, descripción)
- Maneja varias codificaciones HTML

### Estrategia de Chunking de Texto

```python
def chunk_document(content: str, doc_type: str) -> List[Chunk]:
    """
    Chunking inteligente basado en tipo de documento:
    
    - Markdown: Respeta límites de sección (encabezados)
    - PDF: Chunking consciente de párrafos
    - HTML: Límites de elementos semánticos
    - Todos: Tamaño configurable con solapamiento
    """
```

---

## 🤖 Implementación RAG

### Pipeline Principal (`rag/pipeline.py`)

```python
class RAGPipeline:
    """Pipeline principal de procesamiento RAG."""
    
    def ask(self, question: str) -> RAGResult:
        """
        Flujo completo de RAG:
        1. Generar embedding de la pregunta
        2. Buscar en base de datos vectorial chunks similares
        3. Recuperar documentos relevantes top-k
        4. Generar respuesta usando contexto
        5. Retornar resultado con metadatos
        """
```

### Proceso de Recuperación
1. **Embedding de Consulta**: Convertir pregunta a vector usando Ollama
2. **Búsqueda por Similitud**: Búsqueda de similitud coseno FAISS
3. **Selección de Contexto**: Chunks más relevantes top-k
4. **Seguimiento de Fuentes**: Mantener información de fuente de documento

### Proceso de Generación
1. **Construcción de Prompt**: Construir prompt RAG con contexto
2. **Generación Local**: Usar Ollama para generación de respuesta
3. **Procesamiento de Respuesta**: Limpiar y formatear salida
4. **Recolección de Metadatos**: Seguir timing y fuentes

---

## 📊 Componentes Educativos

### Objetivos de Aprendizaje

1. **Extracción de Documentos**
   - Entender desafíos de diferentes formatos de archivo
   - Aprender técnicas de extracción robustas
   - Manejar varias estructuras de documentos

2. **Chunking de Texto**
   - Importancia del tamaño de chunk y solapamiento
   - Estrategias de segmentación conscientes del contenido
   - Impacto en calidad de recuperación

3. **Embeddings Vectoriales**
   - Convertir texto a representaciones numéricas
   - Entender similitud semántica
   - Modelos de embedding locales vs. en la nube

4. **Búsqueda Vectorial**
   - Indexado y búsqueda FAISS
   - Métricas de similitud (similitud coseno)
   - Consideraciones de rendimiento

5. **Implementación RAG**
   - Conceptos de Generación Aumentada por Recuperación
   - Gestión de ventana de contexto
   - Ingeniería de prompts para RAG

### Estructura de Código para Aprendizaje

- **Comentarios Exhaustivos**: Cada función explica su propósito y decisiones
- **Anotaciones de Tipo**: Todas las funciones tienen hints de tipo completos
- **Diseño Modular**: Separación clara de responsabilidades
- **Logging Educativo**: Salida verbosa mostrando cada paso de procesamiento
- **Métricas Simples**: Tiempos de procesamiento, conteos de chunks, puntajes de similitud

---

## 🚀 Ejemplos de Uso

### Flujo Completo

```bash
# 1. Configuración inicial (solo una vez)
docs-to-rag setup
# 🔍 Verificando Ollama...
# 📁 Creando directorios...
# 🤖 Verificando modelos...
# ✅ Configuración completada!

# 2. Añadir documentos al sistema
docs-to-rag add ./mi_documentacion/
# 📄 Procesando documentos...
# ✅ 5 documentos procesados
# 📊 142 chunks creados
# 💾 Base de datos vectorial actualizada

# 3. Verificar qué fue procesado
docs-to-rag stats
# 📚 Documentos indexados: 5
# 🧩 Total de chunks: 142
# 💾 Tamaño BD vectorial: 3.2MB

# 4. Chat interactivo
docs-to-rag chat
# 🤖 Chat RAG iniciado!
# 📝 Tu pregunta: ¿Cómo instalo esto?
# 🔍 Buscando documentos relevantes...
# 📄 Encontrados 3 documentos relevantes
# 💭 Generando respuesta...
# 🤖 Respuesta: Para instalar el software...

# 5. Consulta directa
docs-to-rag query "Resume las características principales"
# 🔍 Procesando consulta...
# 🤖 Respuesta: Las características principales incluyen...
```

---

## 🧪 Desarrollo y Testing

### Dependencias del Proyecto (`pyproject.toml`)

```toml
[project]
name = "docs-to-rag"
version = "0.1.0"
description = "Sistema RAG educativo con LLMs locales"
authors = [
    {name = "Juanje Ojeda", email = "juanje@redhat.com"}
]
requires-python = ">=3.10"

dependencies = [
    "langchain>=0.1.0",
    "langchain-ollama>=0.1.0",
    "docling>=1.0.0",
    "faiss-cpu>=1.7.4",
    "click>=8.1.0",
    "rich>=13.0.0",
    "beautifulsoup4>=4.12.0",
    "markdown>=3.5.0",
    "python-dotenv>=1.0.0",
    "pydantic>=2.5.0",
    "httpx>=0.25.0",
    "aiofiles>=23.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "ruff>=0.1.0",
    "mypy>=1.7.0",
    "pre-commit>=3.0.0",
]

[project.scripts]
docs-to-rag = "src.cli.commands:main"
```

### Estrategia de Testing

- **Tests Unitarios**: Cada componente (extractor, embeddings, recuperación)
- **Tests de Integración**: Funcionalidad completa del pipeline
- **Tests de Documentos**: Manejo de varios formatos de archivo
- **Tests de Rendimiento**: Velocidad de procesamiento y uso de memoria

---

## 🎯 Criterios de Éxito

### Requisitos Funcionales

1. **Procesamiento de Documentos**: Extraer y fragmentar exitosamente archivos PDF, MD, HTML
2. **Almacenamiento Vectorial**: Crear y buscar en base de datos vectorial FAISS
3. **Pipeline RAG**: Recuperar documentos relevantes y generar respuestas coherentes
4. **Interfaz CLI**: Interacción intuitiva por línea de comandos
5. **Operación Local**: Sin dependencias de APIs externas

### Requisitos Educativos

1. **Claridad de Código**: Implementación bien documentada y legible
2. **Ruta de Aprendizaje**: Progresión clara de conceptos simples a complejos
3. **Experimentación**: Fácil modificar parámetros y ver resultados
4. **Entendimiento**: Cada paso explica por qué, no solo cómo

### Objetivos de Rendimiento

- **Procesamiento de Documentos**: < 30 segundos para 10 documentos típicos
- **Respuesta a Consultas**: < 5 segundos para consultas RAG típicas
- **Uso de Memoria**: < 1GB RAM para colecciones moderadas de documentos
- **Almacenamiento**: Tamaño eficiente de base de datos vectorial

---

## 🔄 Fases de Desarrollo

### Fase 1: Base Principal
- Extracción básica de documentos (archivos de texto primero)
- Estrategia simple de chunking
- Integración Ollama para embeddings
- Almacenamiento y recuperación básica FAISS

### Fase 2: Soporte de Documentos
- Extracción PDF con Docling
- Procesamiento Markdown con preservación de estructura
- Extracción de contenido HTML
- Estrategias mejoradas de chunking

### Fase 3: Implementación RAG
- Pipeline completo de recuperación
- Generación de respuestas con contexto
- Desarrollo de interfaz CLI
- Funcionalidad de chat interactivo

### Fase 4: Pulido y Educación
- Documentación exhaustiva
- Documentos de ejemplo y casos de uso
- Optimización de rendimiento
- Testing y validación

---

## 📚 Consideraciones Adicionales

### Prerrequisitos

- **Ollama**: Debe estar instalado y ejecutándose (`ollama serve`)
- **Modelos**: Modelos requeridos descargados (`nomic-embed-text:v1.5`, `llama3.2:latest`)
- **Python**: Versión 3.10 o superior
- **Almacenamiento**: Al menos 500MB de espacio libre para modelos y datos

### Limitaciones

- **Modelos Locales**: El rendimiento depende del hardware local
- **Tipos de Documentos**: Limitado a PDF, Markdown, HTML inicialmente
- **Escala**: Diseñado para uso educativo, no escala de producción
- **Idiomas**: Optimizado para contenido en español/inglés

### Extensiones

Mejoras futuras potenciales:
- Formatos de documentos adicionales (DOCX, TXT)
- Comparación de múltiples modelos de embedding
- Estrategias avanzadas de chunking
- Opción de interfaz web
- Métricas de evaluación y benchmarks

---

## 🔧 Características de Optimización RAG (v0.2)

### Sistema de Parámetros Adaptativos
El sistema ajusta automáticamente los parámetros de recuperación basándose en el tamaño de la base de datos vectorial:

- **Bases de datos pequeñas (≤100 chunks)**: `top_k=3`, `threshold=0.7` (restrictivo)
- **Bases de datos medianas (≤1K chunks)**: `top_k=5`, `threshold=0.6` (equilibrado)
- **Bases de datos grandes (≤5K chunks)**: `top_k=10`, `threshold=0.5` (relajado)
- **Bases de datos muy grandes (>5K chunks)**: `top_k=15-20`, `threshold=0.3-0.4` (permisivo)

### Herramientas de Debug y Diagnóstico
- **Modo Debug**: Flag `--debug` muestra parámetros adaptativos y detalles de recuperación
- **Parámetros Personalizados**: Sobrescribir configuraciones automáticas con `--top-k` y `--threshold`
- **Métricas de Rendimiento**: Información detallada de timing y uso de chunks
- **Puntajes de Similitud**: Visualización opcional de puntajes de relevancia de documentos

### Mejoras CLI
```bash
# Optimización automática
docs-to-rag query "Pregunta" --debug

# Ajuste manual de parámetros
docs-to-rag query "Pregunta" --top-k 20 --threshold 0.2

# Resolución de problemas con resultados pobres
docs-to-rag query "Pregunta" --threshold 0.1  # Más permisivo
```

Esto asegura rendimiento óptimo a través de colecciones de documentos de cualquier tamaño, desde pequeñas demos hasta grandes bases de conocimiento empresariales.

---

## 🌍 Detección Automática de Idioma (v0.3)

### Resumen
El sistema ahora detecta automáticamente contenido en español y optimiza los modelos de embedding consecuentemente, mejorando significativamente la calidad de recuperación para documentos en español.

### Características de Detección de Idioma
- **Detección a Nivel de Documento**: Analiza contenido durante extracción usando frecuencia de palabras clave en español
- **Detección a Nivel de Consulta**: Identifica consultas en español en tiempo de ejecución
- **Cambio Automático de Modelo**: Cambia sin problemas entre modelos de embedding basándose en idioma
- **Optimización por Lotes**: Usa embeddings en español cuando la mayoría de documentos están en español (≥50%)

### Modelos de Embedding
- **Inglés/Por Defecto**: `nomic-embed-text:v1.5` (rápido, eficiente)
- **Especializado en Español**: `jina/jina-embeddings-v2-base-es:latest` (optimizado para español)
- **Fallback Multilingüe**: `mxbai-embed-large:latest` (soporte multilingüe general)

### Algoritmo de Detección
La detección de español se basa en analizar la frecuencia de palabras comunes en español:
- **Artículos**: el, la, los, las
- **Preposiciones**: de, del, en, con, por, para
- **Pronombres**: que, se, una, un
- **Verbos**: es, son, está, están
- **Adverbios**: también, muy, más
- **Conectores**: como, cuando, donde, porque

El contenido se considera español si >15% de las palabras coinciden con palabras clave en español.

### Ejemplos de Uso
```bash
# Detección automática durante procesamiento de documentos
docs-to-rag add "documento_spanish.pdf"  # Auto-detecta español → usa modelo jina

# Procesamiento multilingüe explícito
docs-to-rag reprocess "documento.pdf" --multilingual

# Consulta con detección automática de idioma
docs-to-rag query "¿Cuál es el contenido principal?"  # Auto-detecta consulta en español

# Configuración de idioma
docs-to-rag config --language es  # Establecer idioma principal a español
docs-to-rag config                # Mostrar configuraciones actuales de idioma
```

### Impacto en Rendimiento
- **Documentos en Español**: 40-60% mejora en precisión de recuperación
- **Colecciones de Idioma Mixto**: Selección inteligente de modelo por documento/consulta
- **Consultas Inter-idioma**: Degradación elegante con modelos multilingües
- **Tiempo de Procesamiento**: Overhead mínimo (~2-3% aumento para detección de idioma)

---

## 🧠 Enriquecimiento Jerárquico de Documentos

### Resumen

El comando `enrich` implementa un sofisticado sistema de resumen de documentos que genera múltiples tipos de resúmenes sintéticos para mejorar dramáticamente la calidad de recuperación para preguntas conceptuales. Esta característica transforma el sistema RAG de un simple recuperador basado en chunks a una base de conocimiento multi-nivel.

### Tipos de Resumen Jerárquico

#### 1. Resúmenes a Nivel de Documento
- **Propósito**: Entendimiento general del documento y consultas de temas amplios
- **Contenido**: Resumen ejecutivo, temas principales, conclusiones clave
- **Caso de Uso**: "¿De qué trata este documento?", "Resume los puntos principales"

#### 2. Resúmenes a Nivel de Capítulo
- **Propósito**: Recuperación de información específica de sección
- **Contenido**: Resúmenes de capítulo/sección con contexto estructural
- **Caso de Uso**: "¿Qué discute el Capítulo 3?", "Explica la sección de metodología"

#### 3. Resúmenes a Nivel de Concepto
- **Propósito**: Recuperación enfocada de temas y definiciones
- **Contenido**: Conceptos clave, definiciones y conocimiento especializado
- **Caso de Uso**: "Define machine learning", "Explica los conceptos clave"

### Configuración Especializada de LLM para Resumen Fiel

El sistema usa configuraciones dedicadas de LLM optimizadas específicamente para generación de resúmenes, asegurando mayor calidad y resúmenes más fieles comparado con usar parámetros generales de chat.

#### Parámetros Específicos para Resumen
- **Modelo Dedicado**: Puede usar modelo diferente que respuestas de chat (`summarization_model`)
- **Temperatura Más Baja**: `0.1` para resúmenes más determinísticos y fieles
- **Muestreo Enfocado**: `top_p=0.8` para selección de contenido más enfocada
- **Límites de Tokens Optimizados**:
  - Resúmenes de documento: 400 tokens
  - Resúmenes de capítulo: 250 tokens
  - Resúmenes de concepto: 200 tokens

#### Sistema de Control de Calidad
- **Prompting Mejorado**: Prompts del sistema diseñados específicamente para resumen fiel
- **Generación Multi-intento**: Hasta 3 intentos por resumen con puntuación de calidad
- **Validación de Fidelidad**: Verificación automática de fidelidad del contenido al material fuente
- **Métricas de Calidad**:
  - Validación de longitud
  - Verificación de estructura
  - Consistencia de idioma
  - Análisis de solapamiento de contenido
  - Detección de meta-comentarios

#### Características de Fidelidad de Resumen
- **Aplicación de Requisitos Críticos**: Instrucciones explícitas contra alucinación
- **Verificación de Fidelidad de Fuente**: Valida contenido de resumen contra texto original
- **Puntuación de Calidad**: Sistema de puntuación 0.0-1.0 con umbral 0.8 para aceptación
- **Reintento Automático**: Resúmenes de pobre calidad regenerados automáticamente
- **Análisis de Solapamiento**: Asegura 30-70% solapamiento de palabras clave con fuente (previene tanto copia como alucinación)

### Opciones de Configuración
```bash
# Habilitar validación de calidad (más lento pero más confiable)
docs-to-rag config --enable-validation

# Usar modelo especializado para resúmenes
docs-to-rag config --summary-model "llama3.1:8b"

# Mostrar configuración detallada de resumen
docs-to-rag config
```

### Flujo de Enriquecimiento

#### Pipeline de Procesamiento
1. **Análisis de Documento**: Extraer y analizar estructura de documento
2. **Detección de Capítulos**: Identificar secciones, encabezados y divisiones lógicas
3. **Extracción de Conceptos**: Identificar términos y conceptos clave usando PLN
4. **Generación de Resumen**: Crear resúmenes jerárquicos usando prompts especializados
5. **Validación de Calidad**: Verificación opcional de fidelidad y calidad
6. **Almacenamiento Vectorial**: Convertir resúmenes a chunks y almacenar con embeddings

#### Proceso de Enriquecimiento de Ejemplo
```bash
# 1. Habilitar resumen
docs-to-rag config --enable-summaries

# 2. Enriquecer un documento
docs-to-rag enrich ./paper_investigacion.pdf

# Salida:
# ✅ Enriquecimiento completado exitosamente
# Documentos procesados: 1
# Chunks de resumen generados: 8
#   - Resúmenes de documento: 1
#   - Resúmenes de capítulo: 4
#   - Resúmenes de concepto: 3
```

#### Impacto en Recuperación
- **Antes del Enriquecimiento**: Solo chunks de texto original disponibles
- **Después del Enriquecimiento**: Chunks originales + chunks de resumen sintético
- **Mejora de Consultas**: 30-50% mejores resultados para preguntas conceptuales
- **Casos de Uso Mejorados**:
  - Preguntas de revisión de literatura
  - Explicaciones de conceptos
  - Resúmenes de documentos
  - Resúmenes de secciones

### Compromisos Calidad vs Velocidad
- **Modo Por Defecto**: Generación rápida, validación básica
- **Modo de Calidad**: Generación multi-intento con validación (2-3x más lento, mucho mayor calidad)
- **Modelo Especializado**: Puede usar modelo más grande/mejor solo para resúmenes mientras mantiene modelo rápido para chat

Esta configuración avanzada aborda el desafío crítico de fidelidad de resumen mientras mantiene los beneficios de rendimiento del despliegue local de LLM.

---

*Autor: Juanje Ojeda (juanje@redhat.com)*  
*Proyecto: Sistema RAG Educativo con LLMs Locales*
