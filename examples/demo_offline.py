"""Demo del sistema multi-agente sin clave de API (FakeLLM con respuestas guionizadas).

Ejecutar desde la raíz del proyecto:  python examples/demo_offline.py
"""

import json

from agent_setup.config import Settings
from agent_setup.factory import build_supervisor
from agent_setup.llm import FakeLLM


def main() -> None:
    respuestas = [
        # 1. el supervisor delega en el investigador
        json.dumps({"next": "investigador", "reason": "hay que calcular el dato primero"}),
        # 2. el investigador llama a la calculadora
        json.dumps({"tool": "calculator", "args": {"expression": "12 * 8 + 4"}}),
        # 3. el investigador responde con el dato calculado
        "He calculado que 12 * 8 + 4 = 100.",
        # 4. el supervisor delega en el redactor
        json.dumps({"next": "redactor", "reason": "ya tenemos el dato"}),
        # 5. el redactor escribe el texto
        "Texto redactado con el dato calculado: el resultado es 100.",
        # 6. el supervisor termina con la respuesta final
        json.dumps({"next": "FINISH", "answer": "El resultado final es 100 (12 * 8 + 4)."}),
    ]
    llm = FakeLLM(responses=respuestas)
    supervisor = build_supervisor(llm, Settings(max_steps=8))
    result = supervisor.run("Calcula 12 * 8 + 4 y escribe una frase con el resultado")

    print("Ruta:", " -> ".join(result.route))
    print("Pasos del supervisor:", result.steps)
    print("Respuesta final:", result.final_answer)
    print("\nTranscripción:")
    for msg in result.transcript:
        print(f"[{msg.agent or msg.role}] {msg.content[:90]}")


if __name__ == "__main__":
    main()
