# claude-times

![License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Version](https://img.shields.io/badge/version-1.0.0-green)

Analiza el historial JSONL de Claude Code para generar reportes
detallados de:

- ⏱ Tiempo de trabajo por sesión e instrucción
- 🔢 Consumo de tokens (input, cache read, cache write, output)
- 💰 Costos estimados por modelo
- 🛠 Uso de herramientas (Read, Write, Bash, etc.)
- 📊 Exportación a CSV y JSON

---

## 🎯 Objetivo

Proveer transparencia y control técnico sobre el uso real de Claude
Code:

- Identificar instrucciones costosas
- Optimizar uso de modelos
- Detectar desperdicio por errores API
- Analizar eficiencia del cache
- Medir productividad asistida por IA

---

## 📦 Requisitos

- Python 3.8+
- Claude Code instalado
- Acceso al directorio `~/.claude/projects`

---

## 📦 Instalacion

### Recomendado (CLI Tool)

Instalar con **pipx** (recomendado para CLI tools):

````bash
pipx install claude-times

---

Si no tienes instalado pipx:

brew install pipx
pipx ensurepath
Alternative: pip
pip install claude-times

Después de la instalación:

claude-times --help

---

Development Setup

Clone the repository:

git clone https://github.com/ATuManera/claude-times.git
cd claude-times

Create a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install in editable mode with development tools:

pip install -e ".[dev]"

Run linting:

ruff check .
🔄 Upgrade

If installed with pipx:

pipx upgrade claude-times

If installed with pip:

pip install --upgrade claude-times

---

## 🚀 Uso

```bash
claude-times
````

### Opciones principales

```bash
--list                 # Lista proyectos
--project NOMBRE       # Filtrar proyecto
--all                  # Todas las sesiones
--days 30              # Últimos 30 días
--detail               # Timeline detallado
--csv                  # Exportar CSV
--json                 # Exportar JSON
```

Ejemplo:

```bash
claude-times --project lego --days 30 --detail
```

---

## 🧠 Modelos soportados

Incluye pricing estimado para:

- Claude Opus 4.6
- Claude Opus 4.5
- Claude Sonnet 4.5
- Claude Haiku 4.5

Los costos mostrados corresponden a precios API públicos como referencia
comparativa.

---

## 🔐 Licencia

Este proyecto está licenciado bajo:

**GNU Affero General Public License v3.0 (AGPL-3.0-or-later)**

Esto significa que:

- ✔ Puedes usarlo
- ✔ Puedes modificarlo
- ✔ Puedes redistribuirlo
- ✔ Puedes usarlo comercialmente
- ✔ Puedes integrarlo en SaaS

Pero:

- ⚠ Debes mantener esta misma licencia
- ⚠ Debes publicar el código fuente si lo distribuyes o lo ofreces
  como servicio

Texto completo: https://www.gnu.org/licenses/agpl-3.0.html

---

## 👤 Autor

**J. Fernando Gallarday V.**\
Innova a tu Manera Soluciones Digitales S.A.C. (A Tu Manera Digital)\
Perú

---

## ⚠ Descargo de responsabilidad

Este software se proporciona "tal cual", sin garantía de ningún tipo.
Los cálculos de costo son estimaciones basadas en precios públicos y
pueden no reflejar el costo real en planes de suscripción.

---

## ⭐ Contribuciones

Pull requests y mejoras son bienvenidas. Al contribuir aceptas que tu
código será distribuido bajo AGPL-3.0.
