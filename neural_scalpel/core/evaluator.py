import torch
import math
from typing import Dict, Any, List

class E2EEngineBenchmarker:
    """
    End-to-End Engine Benchmarker for Neural-Scalpel.
    Evaluates real auto-regressive generation tests and real Perplexity calculations
    over a stream of tokens, proving the surgical methodology holds up in a true
    generative environment.
    """
    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def calculate_perplexity(self, text: str, stride: int = 512) -> float:
        """
        Calculates the perplexity of the model on the given text.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        
        encodings = self.tokenizer(text, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)
        
        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + stride, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            
            outputs = self.model(input_ids, labels=target_ids)
            
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model.forward() must support labels
            neg_log_likelihood = outputs.loss
            
            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            
            if end_loc == seq_len:
                break
                
        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl.item()

    @torch.no_grad()
    def measure_kl_divergence(self, base_model: Any, input_text: str) -> float:
        """
        Measures the KL Divergence of the next-token probability distribution (logits)
        between the base model and the current (transplanted) model.
        """
        self.model.eval()
        base_model.eval()
        device = next(self.model.parameters()).device
        
        inputs = self.tokenizer(input_text, return_tensors="pt")
        if hasattr(inputs, 'to'):
            inputs = inputs.to(device)
        elif isinstance(inputs, dict):
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs_surgical = self.model(**inputs)
        outputs_base = base_model(**inputs)
        
        logits_surgical = outputs_surgical.logits
        logits_base = outputs_base.logits
        
        # P = base model, Q = surgical model
        # KL(P || Q) = \sum P(x) * log(P(x) / Q(x))
        p = torch.nn.functional.softmax(logits_base, dim=-1)
        log_q = torch.nn.functional.log_softmax(logits_surgical, dim=-1)
        
        kl_div = torch.nn.functional.kl_div(log_q, p, reduction='batchmean')
        return kl_div.item()

    @torch.no_grad()
    def generate_text(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Generates text to manually inspect logic/hallucination."""
        self.model.eval()
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(inputs, 'to'):
            inputs = inputs.to(device)
        elif isinstance(inputs, dict):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class SQLCapabilityEvaluator:
    """
    Evaluator specifically for SQL generation tasks.
    Checks syntax validity and structural correctness.
    """
    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def extract_sql(self, text: str) -> str:
        """
        Heuristically extracts the first SQL block from model output.
        """
        import re
        # 1. Check for markdown code blocks
        sql_match = re.search(r"```sql\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()
            
        sql_match = re.search(r"```\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()

        # 2. Heuristic: Look for SELECT/WITH and optional semicolon
        text_clean = text.strip()
        start_keywords = ["SELECT", "WITH", "UPDATE", "INSERT", "DELETE", "CREATE", "DROP"]
        
        best_start = -1
        for kw in start_keywords:
            idx = text_clean.upper().find(kw)
            if idx != -1:
                if best_start == -1 or idx < best_start:
                    best_start = idx
        
        if best_start != -1:
            sql_snippet = text_clean[best_start:]
            # Look for semicolon
            semi_idx = sql_snippet.find(";")
            if semi_idx != -1:
                return sql_snippet[:semi_idx+1].strip()
            return sql_snippet.strip()

        return text_clean

    @torch.no_grad()
    def generate_sql(self, prompt: str, max_new_tokens: int = 128) -> str:
        self.model.eval()
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        input_len = inputs.input_ids.shape[1]
        gen_ids = outputs[0, input_len:]
        raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return self.extract_sql(raw_text)

    def validate_syntax(self, sql: str) -> Dict[str, Any]:
        """
        Uses sqlglot to check if the generated SQL is syntactically valid.
        """
        try:
            import sqlglot
            # Basic parse check
            sqlglot.parse_one(sql, read=None)
            return {"valid": True, "error": None}
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def validate_schema(self, sql: str, expected_tables: List[str], expected_columns: List[str]) -> Dict[str, Any]:
        """
        Uses sqlglot AST to extract tables and columns for stricter validation.
        """
        try:
            import sqlglot
            from sqlglot import exp
            parsed = sqlglot.parse_one(sql)
            
            # Extract tables
            found_tables = {t.name.lower() for t in parsed.find_all(exp.Table)}
            # Extract columns
            found_cols = {c.name.lower() for c in parsed.find_all(exp.Column)}
            
            table_ok = all(t.lower() in found_tables for t in expected_tables)
            column_ok = all(c.lower() in found_cols for c in expected_columns)
            
            return {
                "table_ok": table_ok,
                "column_ok": column_ok,
                "found_tables": list(found_tables),
                "found_columns": list(found_cols)
            }
        except Exception:
            return {"table_ok": False, "column_ok": False, "found_tables": [], "found_columns": []}

    def execute_sqlite(self, sql: str, setup_script: str) -> Dict[str, Any]:
        """
        Executes the SQL against an in-memory SQLite database.
        """
        import sqlite3
        conn = sqlite3.connect(":memory:")
        try:
            cursor = conn.cursor()
            cursor.executescript(setup_script)
            cursor.execute(sql)
            results = cursor.fetchall()
            return {"success": True, "results": results, "error": None}
        except Exception as e:
            return {"success": False, "results": None, "error": str(e)}
        finally:
            conn.close()

    def evaluate_suite(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates a list of test cases with syntax, schema, and execution checks.
        """
        results = []
        failure_cases = []
        stats = {
            "total": len(test_cases),
            "syntax_valid": 0,
            "execution_success": 0,
            "exact_match": 0,
            "categories": {}
        }
        
        for case in test_cases:
            prompt = case["prompt"]
            gen_sql = self.generate_sql(prompt)
            syntax = self.validate_syntax(gen_sql)
            
            # Schema check
            schema = self.validate_schema(
                gen_sql, 
                case.get("tables", []), 
                case.get("columns", [])
            )
            
            # Execution check
            execution = {"success": None, "results": None}
            if case.get("sqlite_setup") and syntax["valid"]:
                execution = self.execute_sqlite(gen_sql, case["sqlite_setup"])
            
            # Result correctness check
            is_correct = False
            expected = case.get("expected_result")
            order_sensitive = case.get("order_sensitive", False)
            
            if execution.get("success") and expected is not None:
                actual_res = execution["results"]
                if not order_sensitive:
                    # Sort both if order doesn't matter
                    try:
                        is_correct = sorted(actual_res) == sorted(expected)
                    except:
                        # Fallback for complex results that might not sort easily
                        is_correct = actual_res == expected
                else:
                    is_correct = actual_res == expected
            
            res = {
                "id": case.get("id", "unknown"),
                "category": case.get("category", "general"),
                "prompt": prompt,
                "generated": gen_sql,
                "syntax_valid": syntax["valid"],
                "schema_ok": schema["table_ok"] and schema["column_ok"],
                "execution_success": execution.get("success"),
                "is_correct": is_correct,
                "error": syntax["error"] or execution.get("error")
            }
            
            # Update stats
            if syntax["valid"]: stats["syntax_valid"] += 1
            if execution.get("success"): stats["execution_success"] += 1
            if is_correct: stats["exact_match"] += 1
            
            cat = res["category"]
            if cat not in stats["categories"]:
                stats["categories"][cat] = {"total": 0, "pass": 0, "correct": 0}
            stats["categories"][cat]["total"] += 1
            if execution.get("success"): stats["categories"][cat]["pass"] += 1
            if is_correct: stats["categories"][cat]["correct"] += 1
            
            results.append(res)
            if not is_correct:
                failure_cases.append(res)
            
        stats["execution_success_rate"] = stats["execution_success"] / stats["total"] if stats["total"] > 0 else 0
        stats["execution_accuracy"] = stats["exact_match"] / stats["total"] if stats["total"] > 0 else 0
        
        return {
            "stats": stats,
            "results": results,
            "failure_cases": failure_cases
        }
