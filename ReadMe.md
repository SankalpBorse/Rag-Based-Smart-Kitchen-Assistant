# Rag Based Smart Kitchen Assistant

[cite_start]An intelligent, stateful, and voice-enabled cooking companion built with **Retrieval-Augmented Generation (RAG)**, **Large Language Models (LLMs)**, and a custom stateful cooking orchestration engine[cite: 6, 26]. 

Unlike static recipe chatbots, this system is an interactive partner. [cite_start]It retrieves recipes from a massive culinary database, personalizes them to your dietary needs and pantry inventory, and guides you step-by-step through the cooking process—handling mid-cook substitutions and questions in real-time[cite: 29, 30].

---

## Architecture

`![System Architecture Diagram](docs/Architecture_Diagram.png)`

[cite_start]The system is designed with a modern, decoupled client-server model[cite: 165]:
1. [cite_start]**Frontend (Interface Layer):** A vanilla HTML/CSS/JS web application featuring a chat interface, dynamic step-by-step cooking panels, a sliding pantry management drawer, and WebRTC-based voice recording[cite: 294, 295, 297].
2. [cite_start]**Backend (FastAPI):** Exposes REST endpoints (`/chat`, `/chat/voice`, `/next-step`, `/pantry`) to handle all client requests[cite: 293].
3. [cite_start]**Control Layer (State Machine):** A custom Python controller (`controller.py`) that manages session memory and transitions between `IDLE`, `INGREDIENT_CONFIRM`, and `COOKING` states[cite: 271].
4. **Intelligence Layer (LLM Router):** A multi-tier LLM client. [cite_start]It routes fast, simple tasks (intent classification, extractions) to **Groq (Llama 3.3 70B)**, and complex reasoning tasks (recipe generation, personalization) to **Gemini 2.0 Flash**[cite: 250, 253, 254].
5. [cite_start]**Retrieval Layer (RAG):** Uses `sentence-transformers` and ChromaDB to semantically fetch recipes[cite: 237, 238].
6. [cite_start]**Data Layer:** SQLite for strict inventory/pantry tracking, and ChromaDB for soft behavioral preference tracking[cite: 231, 233].

---

## ✨ Key Features

* [cite_start]**Zero-Latency Step Advancement:** During a cooking session, pressing "Enter" or saying "Next Step" advances the recipe using purely local state logic **(0 LLM calls)**[cite: 279, 386]. This ensures zero latency when your hands are full.
* [cite_start]**Pantry-Aware Personalization:** Before starting a recipe, the system cross-references the ingredients with your SQLite pantry, alerts you to missing items, and offers context-aware substitutions[cite: 276, 277].
* **Dual-Memory Preference Tracking:**
  * [cite_start]*Hard Preferences (SQLite):* Dietary restrictions, spice levels, allergies[cite: 231].
  * [cite_start]*Soft Preferences (ChromaDB):* Behavioral habits extracted dynamically by the LLM (e.g., "User prefers air-frying over deep-frying")[cite: 233, 234].
* [cite_start]**Voice-First Interaction:** Integrated Speech-to-Text via **Groq Whisper Large V3** and Text-to-Speech via **Edge TTS** (with support for Sarvam Bulbul for Indian/Hinglish contexts)[cite: 297, 519]. 

---

## ⚙️ How It Works: The State Machine

[cite_start]The core of the assistant is a rigid Python state machine that eliminates the hallucination loops common in agentic frameworks[cite: 271]. 

1. **`IDLE` Mode:** The default state. The orchestrator classifies user input into one of 13 intents. [cite_start]The user can add items to the pantry, ask for meal suggestions based on available ingredients, or chat generally[cite: 273, 274].
2. **`INGREDIENT_CONFIRM` Mode:** Triggered when the user asks to cook a dish. [cite_start]The system performs RAG retrieval, personalizes the recipe, compares it against the SQLite pantry using fuzzy name matching, and lists missing ingredients[cite: 275, 276]. [cite_start]It waits in this state until the user confirms they are ready[cite: 277].
3. [cite_start]**`COOKING` Mode:** The system guides the user one step at a time[cite: 279]. 
   * [cite_start]**Fast Path:** Blank inputs (Enter) or "Next" trigger the `advance_step` method instantly (0 LLM calls)[cite: 279].
   * [cite_start]**LLM Path:** Keyword detection for phrases like "missing" or "don't have" routes to a targeted LLM call to suggest a substitute based on the *current* step's context[cite: 280, 281].

---

## 🗄️ RAG Database Creation Process

The intelligence of the assistant is grounded in a massive, custom-built Vector Database. [cite_start]The primary recipe collection was synthesized from **6 distinct Kaggle datasets** to ensure diverse, highly detailed culinary coverage[cite: 150, 197]:

1. [cite_start]**Datasets Used:** * Indian Food Dataset (250+ traditional recipes)[cite: 198].
   * [cite_start]Nutritional Values Dataset[cite: 201].
   * [cite_start]Multi-Cuisine Recipes Dataset (20+ international cuisines)[cite: 204].
   * INDORI Dataset (street food)[cite: 205].
   * [cite_start]RecipeNLG Subset (50,000 records)[cite: 206].
   * [cite_start]Indian Recipes with Detailed Nutrition[cite: 207].
2. **Data Cleaning & Standardization:** Using Pandas, the datasets were loaded, cleaned, and normalized. [cite_start]Records with incomplete steps or missing ingredients were purged[cite: 209].
3. [cite_start]**Fuzzy Matching Integration:** To enrich the recipes, nutritional data was mapped to specific recipes using fuzzy string matching via the `rapidfuzz` library[cite: 210].
4. [cite_start]**Chunking Strategy:** Each recipe was consolidated into a single, cohesive text chunk containing the dish name, description, exact ingredients, and step-by-step instructions to preserve full context during retrieval[cite: 211].
5. **Embedding & Storage:** The chunks were embedded using the `all-MiniLM-L6-v2` sentence-transformer model and stored locally in a **ChromaDB** collection (`indian_recipes`). [cite_start]The final production database contains **11,577 high-quality recipe documents**[cite: 212, 213].

[cite_start]*(Note: A secondary Knowledge Base was also constructed from 7 culinary science books, such as "Masala Lab" and "Salt Fat Acid Heat", chunked using PyMuPDF to answer fundamental food science queries but due the problems faced during chunking and retrieval plus I didn't think this would actually help the system I did not use it.)*[cite: 215, 216, 219, 227].

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend Framework** | FastAPI, Uvicorn | [cite_start]REST APIs, Audio handling [cite: 519] |
| **Fast LLM Tier** | Groq (Llama 3.3 70B) | [cite_start]Intent routing, extractions, STT [cite: 516] |
| **Quality LLM Tier** | Google Gemini (2.0 Flash) | [cite_start]Recipe generation, personalization [cite: 516] |
| **Vector Database** | ChromaDB (>= 0.5.0) | [cite_start]Storing recipes & soft preferences [cite: 516] |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | [cite_start]Dense semantic search [cite: 516] |
| **Relational Database** | SQLite | [cite_start]Pantry inventory & hard preferences [cite: 516] |
| **Voice (STT / TTS)** | Whisper Large V3 (Groq) / Edge TTS | [cite_start]Real-time audio interaction [cite: 519] |
| **Frontend UI** | HTML5, CSS3, Vanilla JS | [cite_start]Interactive Chat & Pantry UI [cite: 519] |

---

