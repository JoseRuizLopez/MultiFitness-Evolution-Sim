# Guía para contribuciones de Codex

Este repositorio implementa una simulación evolutiva con múltiples funciones objetivo. Sigue estas indicaciones al modificar el código.

## Estilo de código
- Cumple la guía PEP8 con indentación de 4 espacios.
- Utiliza `snake_case` para funciones y variables y `PascalCase` para clases.
- Los comentarios y docstrings deben estar en español.

## Pruebas
1. Instala las dependencias con:
   ```bash
   pip install -r requirements.txt
   ```
2. Ejecuta todos los tests con:
   ```bash
   pytest -q
   ```
   Asegúrate de que no se reporten fallos antes de confirmar.

## Commits y Pull Requests
- Los mensajes de commit deben ser breves y en imperativo ("Añade soporte para X").
- Las descripciones de Pull Request deben incluir dos apartados:
  - **Summary**: explica los cambios principales.
  - **Testing**: detalla la ejecución de `pytest` u otros comandos relevantes.
