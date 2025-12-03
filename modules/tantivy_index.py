"""Thin wrapper around `tantivy` providing a simple Indexing/Search class.

Usage example:

    from modules.tantivy_index import TantivyIndex

    idx = TantivyIndex(index_dir=None)  # in-memory index

    docs = [
        {"doc_id": 1, "page_id": 1, "text": "The Old Man and the Sea"},
        {"doc_id": 2, "page_id": 1, "text": "Some other text."},
    ]
    idx.add_documents(docs)

    results = idx.search("old man", top_k=5)
    print(results)

This module expects `tantivy` to be installed (pip package `tantivy`).
"""
from typing import Any, Dict, List, Optional
import os

import tantivy


class TantivyIndex:
    """A simple indexing + search helper around `tantivy`.

    Arguments:
        index_dir: Optional path to persist the index. If None an in-memory index is used.
        schema_fields: Optional list of field definitions. Each definition is a dict with:
          - name: field name
          - type: 'text' or 'integer'
          - stored: bool (default True)
          - tokenizer_name: optional tokenizer name for text fields
        memory_limit: integer memory limit passed to the writer (if supported).
    """

    def __init__(
        self,
        index_dir: Optional[str] = None,
        optional_fields: Optional[List[str]] = None,
        schema_fields: Optional[List[Dict[str, Any]]] = None,
        memory_limit: int = 50_000_000,
    ):
        """Create an index with a default schema.

        Defaults to the schema:
            - `doc_id` (text, stored)
            - `page_number` (integer, stored)
            - `text` (text, stored)

        You can add more text fields by passing `optional_fields=["field1", ...]`.
        If `schema_fields` is provided it will be used verbatim and `optional_fields`
        will be ignored.
        """

        if schema_fields is None:
            schema_fields = [
                {"name": "doc_id", "type": "text", "stored": True},
                {"name": "page_number", "type": "integer", "stored": True},
                {"name": "text", "type": "text", "stored": True},
            ]

            if optional_fields:
                for fld in optional_fields:
                    # avoid duplicates
                    if any(f["name"] == fld for f in schema_fields):
                        continue
                    schema_fields.append({"name": fld, "type": "text", "stored": True})

        self.schema_fields = schema_fields
        self.memory_limit = memory_limit

        # Build schema
        sb = tantivy.SchemaBuilder()
        for f in schema_fields:
            fname = f["name"]
            ftype = f.get("type", "text")
            stored = bool(f.get("stored", True))
            if ftype == "text":
                tokenizer = f.get("tokenizer_name", None)
                if tokenizer:
                    sb.add_text_field(fname, stored=stored, tokenizer_name=tokenizer)
                else:
                    sb.add_text_field(fname, stored=stored)
            elif ftype == "integer":
                sb.add_integer_field(fname, stored=stored)
            else:
                raise ValueError(f"Unsupported field type: {ftype}")

        self.schema = sb.build()

        # Create index
        if index_dir:
            os.makedirs(index_dir, exist_ok=True)
            self.index = tantivy.Index(self.schema, path=str(index_dir))
        else:
            self.index = tantivy.Index(self.schema)

    def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        """Add multiple documents to the index and commit.

        Documents should be dictionaries mapping field names to values. For text
        fields a single string value or a list of strings is accepted. For
        integer fields an int is expected.
        """
        try:
            writer = self.index.writer(self.memory_limit)
        except TypeError:
            writer = self.index.writer()

        for doc in docs:
            # Build kwargs suitable for tantivy.Document
            doc_kwargs: Dict[str, Any] = {}
            for f in self.schema_fields:
                name = f["name"]
                ftype = f.get("type", "text")
                if name not in doc:
                    continue
                value = doc[name]
                if value is None:
                    continue
                if ftype == "text":
                    if isinstance(value, list):
                        doc_kwargs[name] = [str(v) for v in value]
                    else:
                        doc_kwargs[name] = [str(value)]
                elif ftype == "integer":
                    # allow strings that represent integers too
                    try:
                        doc_kwargs[name] = int(value)
                    except Exception:
                        raise ValueError(f"Field '{name}' expects integer-like values; got: {value}")

            writer.add_document(tantivy.Document(**doc_kwargs))

        writer.commit()
        # Ensure merges complete before returning (writer object is not usable after)
        writer.wait_merging_threads()

    def search(
        self,
        query_str: str,
        fields: Optional[List[str]] = None,
        top_k: int = 10,
        include_snippets: bool = False,
        snippet_field: str = "body",
    ) -> List[Dict[str, Any]]:
        """Search the index and return a list of result dicts.

        Each result dict contains the stored fields and a `_score` key. If
        `include_snippets` is True an HTML snippet will be returned under
        `_snippet` (if a snippet can be generated).
        """
        # Choose default search fields (all text fields)
        if fields is None:
            fields = [f["name"] for f in self.schema_fields if f.get("type", "text") == "text"]

        # Reload to see last commit
        self.index.reload()
        searcher = self.index.searcher()

        query = self.index.parse_query(query_str, fields, fuzzy_fields={"text": (True, 1, True)})
        search_res = searcher.search(query, top_k)

        results: List[Dict[str, Any]] = []
        for score, doc_address in search_res.hits:
            stored = searcher.doc(doc_address)
            # Convert single-element lists to scalars for convenience
            stored_clean: Dict[str, Any] = {}
            for f in self.schema_fields:
                name = f["name"]
                # Try common access patterns for tantivy Document-like objects
                v = None
                get = getattr(stored, "get", None)
                if callable(get):
                    v = get(name, None)
                else:
                    try:
                        v = stored[name]
                    except Exception:
                        v = getattr(stored, name, None)

                if v is None:
                    continue

                if isinstance(v, list) and len(v) == 1:
                    stored_clean[name] = v[0]
                else:
                    stored_clean[name] = v

            stored_clean["_score"] = score

            if include_snippets:
                try:
                    snippet_generator = tantivy.SnippetGenerator.create(searcher, query, self.schema, snippet_field)
                    snippet = snippet_generator.snippet_from_doc(stored)
                    stored_clean["_snippet"] = snippet.to_html()
                except Exception:
                    stored_clean["_snippet"] = None

            results.append(stored_clean)

        return results


__all__ = ["TantivyIndex"]
