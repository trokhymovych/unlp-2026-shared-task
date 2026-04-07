"""
Question generation pipeline for UNLP 2026 Shared Task.

Extracts text from domain PDFs page-by-page, uses OpenAI gpt-4o-mini to
decide whether a page is question-worthy, and generates Ukrainian MCQ
questions grounded to specific pages. Outputs CSV in the same format as
data/dev_questions.csv.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import threading
import time

from dotenv import load_dotenv
import fitz  # pymupdf
import openai
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
DOMAINS = ["domain_1", "domain_2"]

SYSTEM_PROMPT = """\
You are an expert Ukrainian-language exam question writer.

DOMAIN CONTEXT:
{domain_description}

YOUR TASK:
You will receive the text content of a single page from a Ukrainian PDF document \
belonging to the domain described above. You must:

1. First, IDENTIFY the specific subject of the document from the page text: \
the exact sport name (e.g. "стронгмен", "самбо", "баскетбол") or the exact drug \
name (e.g. "ретаболіл", "фервекс", "амоксикомб"). This is the ENTITY NAME.

2. DECIDE whether this page contains specific factual content suitable for \
generating an exam question. Good pages contain concrete facts such as: \
specific rules of a sport, drug dosages or interactions, procedural requirements, \
penalties, contraindications, composition details, etc.
   - Pages that are tables of contents, title pages, lists of abbreviations, \
publisher/manufacturer info, general introductions without specific facts, or \
pages with mostly diagrams/tables without explanatory text should be SKIPPED.

3. If the page IS suitable, generate up to 10 multiple-choice questions that:
   - Are written entirely in Ukrainian.
   - CRITICAL: Every question MUST explicitly mention the ENTITY NAME (the specific \
sport or drug) in the question text. Generic questions like "Який вид змагань \
визначає результати?" are FORBIDDEN — write "Який вид змагань зі стронгмену \
визначає результати?" instead. A reader must be able to identify which sport or \
drug the question is about WITHOUT seeing the source document.
   - Are answerable ONLY from the provided page text.
   - Have exactly 6 answer options labeled A through F.
   - Have exactly one correct answer.
   - Have 5 plausible but incorrect distractors.
   - NEVER use quotation marks (", «, ») or braces in questions or answer options.
   - Match the style of these examples (notice how each question names the sport or drug):

{few_shot_examples}

4. If the page is NOT suitable, return an empty questions list.

RESPONSE FORMAT (strict JSON):
{{
  "entity_name": "the sport or drug name extracted from the page",
  "questions": [
    {{
      "question": "...",
      "A": "...",
      "B": "...",
      "C": "...",
      "D": "...",
      "E": "...",
      "F": "...",
      "correct_answer": "A"
    }}
  ]
}}

Return ONLY valid JSON. The "correct_answer" field must be exactly one of: A, B, C, D, E, F.
If the page is not suitable, return: {{"entity_name": "", "questions": []}}
"""

USER_PROMPT = """\
Domain: {domain}
Document page text:

{page_text}
"""


@dataclass
class PageInfo:
    domain: str
    doc_id: str
    n_pages: int
    page_num: int  # 1-based
    text: str


@dataclass
class GeneratedQuestion:
    domain: str
    n_pages: int
    question: str
    a: str
    b: str
    c: str
    d: str
    e: str
    f: str
    correct_answer: str
    doc_id: str
    page_num: int


def extract_pages(data_dir: Path) -> list[PageInfo]:
    """Extract text from every page of every PDF across all domains."""
    pages: list[PageInfo] = []
    for domain in DOMAINS:
        domain_dir = data_dir / domain
        if not domain_dir.exists():
            log.warning("Domain directory not found: %s", domain_dir)
            continue
        pdf_files = sorted(domain_dir.glob("*.pdf"))
        log.info("Domain %s: found %d PDFs", domain, len(pdf_files))
        for pdf_path in pdf_files:
            try:
                doc = fitz.open(pdf_path)
            except Exception:
                log.exception("Failed to open %s", pdf_path)
                continue
            n_pages = len(doc)
            for page_idx in range(n_pages):
                page = doc[page_idx]
                text = page.get_text("text")
                pages.append(
                    PageInfo(
                        domain=domain,
                        doc_id=pdf_path.name,
                        n_pages=n_pages,
                        page_num=page_idx + 1,
                        text=text,
                    )
                )
            doc.close()
    return pages


def load_domain_descriptions(data_dir: Path) -> dict[str, str]:
    """Load readme-en.txt for each domain."""
    descriptions: dict[str, str] = {}
    for domain in DOMAINS:
        readme_path = data_dir / domain / "readme-en.txt"
        if readme_path.exists():
            descriptions[domain] = readme_path.read_text(encoding="utf-8").strip()
        else:
            descriptions[domain] = f"Domain: {domain}"
    return descriptions


def load_few_shot_examples(data_dir: Path, n_per_domain: int = 3) -> dict[str, str]:
    """Sample a few existing questions from dev_questions.csv as few-shot examples."""
    csv_path = data_dir / "dev_questions.csv"
    examples: dict[str, str] = {}
    if not csv_path.exists():
        for domain in DOMAINS:
            examples[domain] = "(No examples available)"
        return examples

    df = pd.read_csv(csv_path)
    for domain in DOMAINS:
        domain_df = df[df["Domain"] == domain]
        if domain_df.empty:
            examples[domain] = "(No examples available)"
            continue
        sample = domain_df.sample(n=min(n_per_domain, len(domain_df)), random_state=42)
        lines: list[str] = []
        for _, row in sample.iterrows():
            lines.append(
                f"Question: {row['Question']}\n"
                f"A: {row['A']}\n"
                f"B: {row['B']}\n"
                f"C: {row['C']}\n"
                f"D: {row['D']}\n"
                f"E: {row['E']}\n"
                f"F: {row['F']}\n"
                f"Correct: {row['Correct_Answer']}"
            )
        examples[domain] = "\n\n".join(lines)
    return examples


def generate_questions_for_page(
    client: openai.OpenAI,
    page: PageInfo,
    system_prompt: str,
    model: str,
    max_retries: int = 6,
) -> list[GeneratedQuestion]:
    """Call OpenAI API to generate questions for a single page."""
    user_msg = USER_PROMPT.format(domain=page.domain, page_text=page.text[:6000])

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=2000,
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            questions_raw = data.get("questions", [])

            results: list[GeneratedQuestion] = []
            for q in questions_raw:
                correct = q.get("correct_answer", "").strip().upper()
                if correct not in {"A", "B", "C", "D", "E", "F"}:
                    continue
                required_keys = {"question", "A", "B", "C", "D", "E", "F"}
                if not required_keys.issubset(q.keys()):
                    continue
                results.append(
                    GeneratedQuestion(
                        domain=page.domain,
                        n_pages=page.n_pages,
                        question=q["question"],
                        a=q["A"],
                        b=q["B"],
                        c=q["C"],
                        d=q["D"],
                        e=q["E"],
                        f=q["F"],
                        correct_answer=correct,
                        doc_id=page.doc_id,
                        page_num=page.page_num,
                    )
                )
            return results

        except (openai.RateLimitError, openai.APIConnectionError):
            wait = min(2 ** (attempt + 2), 120)
            log.warning(
                "Rate limit / connection error, retrying in %ds (attempt %d/%d)...", wait, attempt + 1, max_retries
            )
            time.sleep(wait)
        except (json.JSONDecodeError, KeyError) as e:
            log.warning(
                "Failed to parse response for %s page %d: %s",
                page.doc_id,
                page.page_num,
                e,
            )
            return []
        except Exception:
            log.exception("Unexpected error for %s page %d", page.doc_id, page.page_num)
            return []

    log.error("Max retries exceeded for %s page %d", page.doc_id, page.page_num)
    return []


def save_results(
    questions: list[GeneratedQuestion],
    output_path: Path,
) -> None:
    """Write generated questions to CSV in dev_questions.csv format."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Question_ID",
                "Domain",
                "N_Pages",
                "Question",
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "Correct_Answer",
                "Doc_ID",
                "Page_Num",
            ]
        )
        for i, q in enumerate(questions):
            writer.writerow(
                [
                    i,
                    q.domain,
                    q.n_pages,
                    q.question,
                    q.a,
                    q.b,
                    q.c,
                    q.d,
                    q.e,
                    q.f,
                    q.correct_answer,
                    q.doc_id,
                    q.page_num,
                ]
            )
    log.info("Saved %d questions to %s", len(questions), output_path)


_file_lock = threading.Lock()


def save_progress(questions: list[GeneratedQuestion], buffer_path: Path) -> None:
    """Append questions to a JSONL buffer for crash recovery (thread-safe)."""
    with _file_lock:
        with open(buffer_path, "a", encoding="utf-8") as f:
            for q in questions:
                f.write(json.dumps(vars(q), ensure_ascii=False) + "\n")


def load_progress(buffer_path: Path) -> list[GeneratedQuestion]:
    """Load previously generated questions from the JSONL buffer."""
    if not buffer_path.exists():
        return []
    questions: list[GeneratedQuestion] = []
    with open(buffer_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            questions.append(GeneratedQuestion(**data))
    return questions


def pages_already_done(questions: list[GeneratedQuestion]) -> set[tuple[str, int]]:
    """Return set of (doc_id, page_num) already processed."""
    return {(q.doc_id, q.page_num) for q in questions}


def main() -> None:
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="Generate Ukrainian MCQ questions from domain PDFs using OpenAI.")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "generated_questions.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="Path to data directory",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Max pages to process (0 = all)",
    )
    parser.add_argument(
        "--min-page-chars",
        type=int,
        default=100,
        help="Skip pages with fewer characters than this",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=25,
        help="Number of concurrent threads for API calls (default: 25)",
    )
    args = parser.parse_args()

    if not args.api_key:
        parser.error("Provide --api-key or set OPENAI_API_KEY environment variable")

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    buffer_path = output_path.with_suffix(".jsonl")

    log.info("Extracting pages from PDFs...")
    all_pages = extract_pages(data_dir)
    log.info("Total pages extracted: %d", len(all_pages))

    pages = [p for p in all_pages if len(p.text.strip()) >= args.min_page_chars]
    log.info("Pages after filtering (>= %d chars): %d", args.min_page_chars, len(pages))

    if args.max_pages > 0:
        pages = pages[: args.max_pages]
        log.info("Limited to %d pages (--max-pages)", args.max_pages)

    existing = load_progress(buffer_path)
    done = pages_already_done(existing)
    if existing:
        log.info("Resuming: %d questions already generated, %d pages done", len(existing), len(done))

    domain_descriptions = load_domain_descriptions(data_dir)
    few_shot = load_few_shot_examples(data_dir)

    system_prompts: dict[str, str] = {}
    for domain in DOMAINS:
        system_prompts[domain] = SYSTEM_PROMPT.format(
            domain_description=domain_descriptions[domain],
            few_shot_examples=few_shot[domain],
        )

    client = openai.OpenAI(api_key=args.api_key)
    all_questions = list(existing)

    generated = 0
    empty = 0
    lock = threading.Lock()

    pages_to_process = [p for p in pages if (p.doc_id, p.page_num) not in done]
    log.info("Pages to process (excluding already done): %d", len(pages_to_process))
    log.info("Using %d concurrent workers", args.workers)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                generate_questions_for_page,
                client,
                page,
                system_prompts[page.domain],
                args.model,
            ): page
            for page in pages_to_process
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating questions"):
            new_qs = future.result()
            with lock:
                if new_qs:
                    all_questions.extend(new_qs)
                    save_progress(new_qs, buffer_path)
                    generated += len(new_qs)
                else:
                    empty += 1

    log.info(
        "Done. Generated %d new questions, %d pages yielded no questions.",
        generated,
        empty,
    )
    log.info("Total questions (including resumed): %d", len(all_questions))

    save_results(all_questions, output_path)

    if buffer_path.exists():
        buffer_path.unlink()
        log.info("Cleaned up buffer file %s", buffer_path)


if __name__ == "__main__":
    main()
