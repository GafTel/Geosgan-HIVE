# geosgan-hive

Setup base de un **sistema multi-agente con IA** en Python: un **supervisor** enruta la
tarea entre **agentes especialistas**, y cada agente puede usar **herramientas** en bucle
hasta dar su respuesta final.

```
usuario ──► Supervisor ──┬──► researcher ──┐
                         ├──► writer ──────┼──► respuesta final
                         └──► reviewer ───────┘
```

Cada agente ejecuta su propio bucle: si el modelo responde con una llamada a herramienta
(en JSON), se ejecuta y se reintenta, hasta un máximo de pasos.

## Estructura

```
src/geosgan_hive/
├── messages.py       # Message: unidad de la conversación/transcripción
├── llm.py            # LLMClient (protocolo), LiteLLMClient (real) y FakeLLM (tests/demos)
├── tools.py          # Herramientas (calculadora segura, fecha UTC, conteo de palabras)
├── agents.py         # Agent: bucle de tool-calling por agente
├── orchestrator.py   # Supervisor: decide qué agente sigue o termina (FINISH)
├── factory.py        # Equipo por defecto: researcher, writer, reviewer
├── config.py         # Settings (pydantic-settings, prefijo HIVE_)
└── cli.py            # Punto de entrada `geosgan-hive` / `python -m geosgan_hive`
tests/                # Tests con FakeLLM (no necesitan clave de API)
examples/demo_offline.py  # Demo completa sin clave de API
```

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # y añade la clave de tu proveedor (OPENAI_API_KEY, etc.)
```

## Uso

```bash
# Demo sin clave de API (respuestas guionizadas):
python examples/demo_offline.py

# Con un modelo real (usa HIVE_MODEL del .env y la clave del proveedor):
geosgan-hive "Investiga las ventajas de los agentes multi-modelo y redacta un resumen"
python -m geosgan_hive "Explica qué es un supervisor de agentes" --model gpt-4o-mini
```

La salida muestra la **ruta** de agentes que atendieron la tarea, los pasos del supervisor
y la respuesta final.

## Tests y lint

```bash
pytest
ruff check .
```

## Cómo añadir un agente

```python
from geosgan_hive import Agent, Supervisor, LiteLLMClient, Settings, load_settings

settings = load_settings()
llm = LiteLLMClient(model=settings.model)
agentes = {
    "traductor": Agent(
        name="traductor",
        description="Traduce textos al idioma indicado.",
        system_prompt="Eres un traductor profesional...",
        llm=llm,
    ),
}
supervisor = Supervisor(agentes, llm, max_steps=settings.max_steps)
resultado = supervisor.run("Traduce 'hola' al inglés")
print(resultado.final_answer, resultado.route)
```

Para añadir una herramienta, crea una clase con `name`, `description`, `parameters`
(JSON Schema) y `run(args)`, y regístrala en el `ToolRegistry` del agente.

## Configuración (variables de entorno / .env)

| Variable               | Por defecto   | Descripción                                  |
| ---------------------- | ------------- | -------------------------------------------- |
| `HIVE_MODEL`          | `gpt-4o-mini` | Modelo de litellm (cualquier proveedor)      |
| `HIVE_API_KEY`        | —             | Clave genérica opcional para tu endpoint     |
| `HIVE_MAX_STEPS`      | `8`           | Máximo de pasos del supervisor               |
| `HIVE_MAX_TOOL_STEPS` | `5`           | Máximo de llamadas a herramientas por agente |
| `HIVE_TEMPERATURE`    | `0.2`         | Temperatura del modelo                       |

Las claves de proveedor (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...) las lee litellm
directamente del entorno. Nunca las imprimas ni las subas a git (`.env` está en `.gitignore`).
