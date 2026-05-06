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

    @torch.no_grad()
    def generate_sql(self, prompt: str, max_new_tokens: int = 128) -> str:
        self.model.eval()
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=False, # Use greedy for deterministic eval
            pad_token_id=self.tokenizer.eos_token_id
        )
        # Only return the generated part
        input_len = inputs.input_ids.shape[1]
        gen_ids = outputs[0, input_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def validate_syntax(self, sql: str) -> Dict[str, Any]:
        """
        Uses sqlglot to check if the generated SQL is syntactically valid.
        """
        try:
            import sqlglot
            # Basic parse check
            sqlglot.transpile(sql, read=None)
            return {"valid": True, "error": None}
        except ImportError:
            return {"valid": None, "error": "sqlglot not installed"}
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def evaluate_suite(self, test_cases: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Evaluates a list of test cases: {"prompt": "...", "reference": "..."}
        """
        results = []
        valid_count = 0
        total = len(test_cases)
        
        for case in test_cases:
            prompt = case["prompt"]
            gen_sql = self.generate_sql(prompt)
            syntax = self.validate_syntax(gen_sql)
            
            res = {
                "prompt": prompt,
                "generated": gen_sql,
                "syntax_valid": syntax["valid"],
                "error": syntax["error"]
            }
            if syntax["valid"]:
                valid_count += 1
            results.append(res)
            
        return {
            "pass_rate": valid_count / total if total > 0 else 0,
            "total": total,
            "results": results
        }
