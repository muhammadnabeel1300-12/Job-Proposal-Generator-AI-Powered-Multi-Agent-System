# AI Job Description Analyzer & Proposal Generator

An AI-powered Streamlit application that reads a job description and a candidate CV
(normal or scanned documents), compares them, and writes a personalized, reviewed job proposal.
Built with Python, Streamlit and the Anthropic Claude API.

## Problem statement
Writing a good proposal for every job takes time. Candidates often miss key requirements,
copy generic text, or accidentally exaggerate their experience. Documents also arrive as scans
or photos that normal parsers cannot read.

## Objectives
- Read PDF, DOCX, TXT, JPG, JPEG and PNG files, using OCR only when needed.
- Extract **explicit** and **inferred** job requirements separately.
- Compare the CV with the requirements using a **documented scoring method**.
- Write a tailored proposal and **review it against the CV** so nothing is invented.

## Features
- Smart PDF handling: selectable text is read directly; OCR runs only on scanned pages (mixed PDFs supported).
- Page-level extraction report, OCR quality warnings, editable extracted text.
- Four independent AI agents sharing one Claude client.
- Every AI answer is validated with Pydantic; malformed JSON is retried once.
- Match score calculated in Python (not by the AI); CV quotes are verified against the CV text.
- "Not Mentioned in CV" (unknown) is kept separate from "Confirmed gap".
- Reviewer loop with a maximum of two revisions; unresolved unsupported claims block approval.
- Prompt-injection protection: documents are treated as untrusted data.
- Results kept in session state, so no repeated API calls. Edit, copy and download (TXT) the proposal.

## The four agents
| Agent | File | Job |
|---|---|---|
| 1. Job Analyzer | `agents/job_analyzer.py` | Explicit keywords (with evidence) + inferred requirements (with reasoning) |
| 2. Profile Matcher | `agents/profile_matcher.py` | Per-requirement CV assessment, verified evidence, score |
| 3. Proposal Writer | `agents/proposal_writer.py` | Writes the proposal from verified facts only |
| 4. Proposal Reviewer | `agents/reviewer.py` | Checks grammar, tone, relevance, claims; max 2 revisions |

### Scoring methodology
```
weights : required = 3.0 | preferred = 1.5 | inferred = 1.0
credit  : matched = 1.0 | partial = 0.5 | not_mentioned = 0 | confirmed_missing = 0
score   = 100 x sum(weight x credit) / sum(weight)
```
Claude only judges each requirement and quotes the CV. If the quote is not found in the CV text,
the item is not counted. A second figure shows the potential score if all unverified items were confirmed.

## Technology stack
Python, Streamlit, Anthropic Claude API, Pydantic, PyMuPDF, pypdf, python-docx,
Tesseract OCR + pytesseract, Pillow, python-dotenv.

## Folder structure
```
job-description-analyzer/
├── app.py                  Streamlit interface (5 pages)
├── schemas.py              Pydantic models
├── agents/                 job_analyzer, profile_matcher, proposal_writer, reviewer
├── utils/
│   ├── file_parser.py      PDF / DOCX / TXT / image parsing
│   ├── ocr_processor.py    Tesseract OCR (separate module)
│   ├── claude_client.py    Reusable Claude client + error handling
│   ├── prompts.py          All prompts
│   └── text_checks.py      Evidence checking + prompt-injection defence
├── tests/                  pytest test suite
├── requirements.txt / requirements-dev.txt
├── .env.example  .env  .gitignore
└── README.md
```

## Installation (Windows, PowerShell)
Requires Python 3.10+.

On Windows, install the [Microsoft Visual C++ 2015-2022 Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe)
for PyMuPDF's native PDF and OCR support. If it is unavailable, the app can still
read selectable PDF text using the fallback parser, but scanned PDFs will not be OCR'd.

```powershell
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it (if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)
venv\Scripts\Activate.ps1

# 3. Install requirements
pip install -r requirements.txt
```

### 4. Configure the `.env` file
Open `.env` and fill in:
```
ANTHROPIC_API_KEY=your_real_key
CLAUDE_MODEL=claude-sonnet-5-5
```
Get a key at https://console.anthropic.com. `CLAUDE_MODEL` can be any model ID your account supports.
Never share or commit `.env` (it is already in `.gitignore`).

### 5. Install Tesseract OCR (separate from pip!)
`pip install pytesseract` only installs the Python wrapper. The Tesseract program itself must be installed:
1. Download the Windows installer: https://github.com/UB-Mannheim/tesseract/wiki
2. Install with default options (`C:\Program Files\Tesseract-OCR\`).
3. Add that folder to your Windows **PATH**, then reopen the terminal and run `tesseract --version`.
4. Alternatively, add this line to `.env`:
   `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe`

The app also checks the default Windows install folders automatically.
Without Tesseract, text-based PDFs, DOCX and TXT still work; only scans and images need OCR.

### 6. Run the application
```powershell
streamlit run app.py
```
Open http://localhost:8501 if the browser does not open by itself.

## How to use
1. **Document Upload**: upload the job description and the CV, click *Extract text*.
2. **Document Extraction**: check the text (fix OCR mistakes), click *Confirm*.
3. **Job Analysis**: run Agent 1 and review explicit vs. inferred requirements.
4. **Profile Matching**: run Agent 2 to see score, strengths, unverified items and gaps.
5. **Proposal Generation**: generate, read the reviewer feedback, edit, copy or download.

## Testing
```powershell
pip install -r requirements-dev.txt
python -m pytest -v
```
The tests need **no API key and cost nothing**: Claude is replaced by a fake client.
OCR tests are skipped automatically if Tesseract is not installed.

| # | Test case | Where |
|---|---|---|
| 1-3 | Text PDF, scanned PDF, mixed PDF | `tests/test_file_parser.py` |
| 4-6 | DOCX, TXT, JPG/PNG | `tests/test_file_parser.py` |
| 7-9 | Empty, corrupted, unsupported files | `tests/test_file_parser.py` |
| 10-11 | Missing API key, invalid Claude response | `tests/test_claude_client.py` |
| 12-13 | CV with matching / missing skills | `tests/test_agents.py` |
| 14 | Proposal with unsupported claims | `tests/test_agents.py` |
| - | Full upload-to-proposal UI flow | `tests/test_app_flow.py` |

## Security notes
- API keys live only in `.env`; they are never shown in the interface.
- Uploads are validated (extension, size, empty/corrupt files) and never executed.
- Document text is sanitized and wrapped in tags; agents are told to treat it as data, never as instructions.
- Claude's quotes are verified against the source documents before they affect results.

## Troubleshooting
| Problem | Fix |
|---|---|
| "Tesseract OCR was not found" | Install Tesseract, add to PATH or set `TESSERACT_CMD` |
| "API key was rejected" | Check `ANTHROPIC_API_KEY` in `.env`, restart the app |
| "Model ... was not found" | Set a valid `CLAUDE_MODEL` |
| "Rate limit reached" | Wait a minute and try again |
| OCR text is poor | Use a clearer/higher-resolution scan, or correct the text on page 2 |
