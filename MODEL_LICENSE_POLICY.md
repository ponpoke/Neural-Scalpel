# Neural-Scalpel Model License Policy

Neural-Scalpel is a mathematical framework for projecting parameter differences (LoRAs/Task Vectors) from a source model to a target model.

## 1. Derivative Work Clause
When you use Neural-Scalpel to project a LoRA trained on Model A to a new base Model B, the resulting projected LoRA (the `.scalpel_route` payload) is mathematically derived from both the source LoRA and the target base model's initialization space.

Therefore, the projected payload is generally considered a **derivative work** of both:
1. The original LoRA's license.
2. The source base model's license (if the LoRA is considered a derivative of the base model).
3. The target base model's license.

## 2. Your Responsibility
It is the user's explicit responsibility to ensure that they have the legal right to combine the licenses of the source model, the source adapter, and the target model. 

For example:
- You cannot legally project a LoRA trained on a model with a non-commercial license onto a target model with a commercial license and use the result commercially. The strictest license terms generally apply.
- Neural-Scalpel provides a `license` field in the `.scalpel_route` manifest to help track this provenance.

## 3. Route Auditing
The Neural-Scalpel Hot-Swap Runtime includes an audit-logging mechanism. Operators should configure the runtime to reject `.scalpel_route` payloads that do not contain valid or approved license metadata if operating in a corporate or production environment.
