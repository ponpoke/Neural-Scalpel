# Route-Aware Weight Surgery inside vLLM: Dynamic Specialization without Model Reloading

Large Language Models (LLMs) are memory-hungry. When you have dozens of domain-specific tasks—SQL generation, Python coding, creative writing, safety filtering—running a dedicated LLM for each is financially prohibitive.

LoRA (Low-Rank Adaptation) adapters help, but existing serving engines often treat them as second-class citizens, either limiting them to specific PEFT architectures or failing to guarantee strict isolation.

**Enter Neural-Scalpel.**

Neural-Scalpel is a candidate for a new runtime serving pattern. Instead of treating models as static blocks of VRAM, Neural-Scalpel performs **route-aware weight surgery** inside vLLM's execution loop.

## How It Works
1. **Request arrives** with a `route_id`.
2. **Scheduler** groups requests homogenously by route.
3. **HotSwap Runtime** intercepts the forward pass, dynamically patching the attention and MLP layers with the specific route's weights (Swap).
4. **HotSwap Runtime** activates the route weights for a route window.
5. **Requests** belonging to that route are processed while the route remains active.
6. **When the route window ends** or the active route changes, the runtime rolls back to the base weights and verifies restoration.

By guaranteeing symmetry (Swaps == Rollbacks) and holding routes active for homogenous batches, Neural-Scalpel achieves strict route isolation with minimal latency overhead.

## Benchmarks
In a 6-hour extended soak test, Neural-Scalpel handled 1,956,000 requests across mixed routes with zero VRAM growth and zero route violations.

In controlled single-GPU validation:
- Phase 5-D measured Scalpel v2 at ~2574 tok/s vs Native LoRA at ~983 tok/s across 50 prompts × 3 runs.
- Phase 5-E-1 processed 1000 dynamically routed two-route requests with 0 route violations and 0 quarantine events.
- Phase 5-F confirmed exact Base-before/Base-after text match and 100.0% top-token logprob trace similarity after verified rollback under tested cache-reset conditions.

*This is not yet production-ready serving software. Formal Production Candidate status remains pending the final 24h persistent-route soak and broader deployment validation.*

Read more and reproduce our results on GitHub!
