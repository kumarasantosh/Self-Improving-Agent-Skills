import json
import os
from typing import Callable, List, Optional, Dict, Any

from pydantic import BaseModel, Field

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


class FailureAnalysis(BaseModel):
    diagnosis: str = Field(description="Root cause and analytical breakdown of failed test evaluations")
    mutation_strategy: str = Field(
        description="Selected optimization strategy: add_example, add_constraint, restructure, or add_edge_case"
    )
    target_section: str = Field(description="Target section or directive of the skill specification to revise")
    suggested_change: str = Field(description="Precise technical modification proposed to resolve the failure")


class SkillMutation(BaseModel):
    description: str = Field(description="Concise description of the specific mutation applied")
    reasoning: str = Field(description="Technical rationale explaining how this modification resolves the issue")
    new_skill_md: str = Field(description="Full updated content for SKILL.md retaining structure and frontmatter")


class SkillForgeOptimizer:
    def __init__(self, api_key: str, model: str = "gemini-3-flash-preview"):
        os.environ["GOOGLE_API_KEY"] = api_key
        self.model = model
        self._session_service = InMemorySessionService()
        self._call_counter = 0

        self.executor = Agent(
            name="executor",
            model=model,
            instruction=(
                "You are the SkillForge Execution and Evaluation Agent. You operate across three distinct modes:\n\n"
                "1. EXECUTE MODE: Follow the provided skill specification meticulously to process user requests. "
                "Output strictly what the skill produces with zero extraneous commentary.\n\n"
                "2. ANALYZE MODE: Review incoming skill definitions and generate comprehensive test scenarios "
                "paired with rigorous binary evaluation criteria. Output valid, parseable JSON.\n\n"
                "3. SCORE MODE: Systematically evaluate agent execution output against specified criteria. "
                "Produce precise binary pass/fail ratings with justification in valid JSON."
            ),
        )

        self.analyst = Agent(
            name="analyst",
            model=model,
            instruction=(
                "You are the SkillForge Diagnostic Analyst Agent. "
                "Analyze failed test evaluation records to identify root failure causes, systemic prompt ambiguities, "
                "or missing constraints. Recommend an optimal mutation strategy from: "
                "add_example, add_constraint, restructure, or add_edge_case."
            ),
            output_schema=FailureAnalysis,
        )

        self.mutator = Agent(
            name="mutator",
            model=model,
            instruction=(
                "You are the SkillForge Surgical Prompt Mutator Agent. "
                "Given a SKILL.md document and an analyst diagnostic report, make exactly ONE high-impact, "
                "surgical refinement. Preserve YAML frontmatter and formatting integrity. "
                "Output the complete revised SKILL.md document."
            ),
            output_schema=SkillMutation,
        )

    async def _dispatch_agent(self, agent: Agent, prompt: str) -> str:
        self._call_counter += 1
        user_id = f"sf_session_{self._call_counter}"
        runner = Runner(
            agent=agent, app_name="skillforge_engine", session_service=self._session_service
        )
        session = await self._session_service.create_session(
            app_name="skillforge_engine", user_id=user_id
        )
        accumulated_text = ""
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=types.Content(parts=[types.Part(text=prompt)]),
        ):
            if hasattr(event, "content") and event.content:
                for part in event.content.parts or []:
                    if hasattr(part, "text") and part.text:
                        accumulated_text += part.text
        return accumulated_text

    async def _query_agent_json(self, agent: Agent, prompt: str, fallback: Any = None) -> Any:
        raw_text = await self._dispatch_agent(agent, prompt)
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            idx = raw_text.find("{")
            if idx == -1:
                idx = raw_text.find("[")
            if idx != -1:
                try:
                    result, _ = decoder.raw_decode(raw_text, idx)
                    return result
                except json.JSONDecodeError:
                    pass
            if fallback is not None:
                return fallback
            raise

    _ask = _dispatch_agent
    _ask_json = _query_agent_json

    async def analyze_skill(self, skill_files: Dict[str, str]) -> Dict[str, Any]:
        skill_md = next(
            (v for k, v in skill_files.items() if k.endswith("SKILL.md")), ""
        )
        refs = {k: v for k, v in skill_files.items() if "references/" in k}
        ref_text = ""
        if refs:
            ref_text = "\n\nReference Material:\n" + "\n---\n".join(
                f"## {k}\n{v}" for k, v in refs.items()
            )

        prompt = (
            f"Perform an architectural analysis on this agent skill specification and generate "
            f"comprehensive test scenarios with rigorous binary evaluation criteria.\n\n"
            f"# SKILL.md\n{skill_md}\n{ref_text}\n\n"
            f"Produce:\n"
            f"1. 3-4 diverse, challenging test scenarios (realistic inputs)\n"
            f"2. 4-6 unambiguous binary evaluation criteria (yes/no tests)\n\n"
            f"Return JSON adhering to schema:\n"
            f'{{"scenarios": [{{"id": 1, "name": "Short Title", "description": "Scenario context", '
            f'"input": "Exact test user prompt"}}], '
            f'"evals": [{{"id": 1, "name": "Criterion Check", "criterion": "Core expectation", '
            f'"question": "Direct yes/no validation question", '
            f'"pass_condition": "Expected positive manifestation", "fail_condition": "Failure indicators"}}]}}'
        )
        return await self._query_agent_json(self.executor, prompt)

    async def optimize(
        self,
        skill_files: Dict[str, str],
        scenarios: List[Dict[str, Any]],
        evals: List[Dict[str, Any]],
        max_rounds: int = 5,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        async def broadcast(event_payload: dict):
            if callback:
                await callback(event_payload)

        skill_md = next(
            (v for k, v in skill_files.items() if k.endswith("SKILL.md")), ""
        )
        current_md = skill_md
        score_trajectory: List[float] = []
        mutation_audit: List[Dict[str, Any]] = []

        baseline = await self._score_skill(current_md, scenarios, evals)
        baseline_score = round(100 * baseline["passed"] / max(baseline["total"], 1), 1)
        score_trajectory.append(baseline_score)

        await broadcast({
            "type": "baseline",
            "data": {
                "score": baseline_score,
                "passed": baseline["passed"],
                "total": baseline["total"],
                "per_eval": baseline["per_eval"],
            },
        })

        for round_idx in range(1, max_rounds + 1):
            await broadcast({"type": "experiment_start", "data": {"round": round_idx}})

            diagnosis = await self._analyze_failures(
                current_md, scenarios, evals, baseline["details"]
            )

            mutation = await self._mutate_skill(current_md, diagnosis)
            candidate_md = mutation.get("new_skill_md", current_md)

            benchmark = await self._score_skill(candidate_md, scenarios, evals)
            candidate_score = round(100 * benchmark["passed"] / max(benchmark["total"], 1), 1)

            is_accepted = candidate_score > baseline_score
            audit_entry = {
                "round": round_idx,
                "strategy_type": diagnosis.get("mutation_strategy", "unknown"),
                "diagnosis": diagnosis.get("diagnosis", ""),
                "description": mutation.get("description", ""),
                "score_before": baseline_score,
                "score_after": candidate_score,
                "kept": is_accepted,
            }
            mutation_audit.append(audit_entry)

            if is_accepted:
                current_md = candidate_md
                baseline = benchmark
                baseline_score = candidate_score

            score_trajectory.append(baseline_score)

            await broadcast({
                "type": "experiment_result",
                "data": {
                    "round": round_idx,
                    "score": candidate_score,
                    "kept": is_accepted,
                    "status": "kept" if is_accepted else "discarded",
                    "description": mutation.get("description", ""),
                    "strategy": diagnosis.get("mutation_strategy", ""),
                    "per_eval": benchmark["per_eval"],
                },
            })

        final_score = baseline_score
        await broadcast({
            "type": "complete",
            "data": {
                "baseline_score": score_trajectory[0],
                "final_score": final_score,
                "improved_skill_md": current_md,
                "score_history": score_trajectory,
                "mutation_log": mutation_audit,
                "strategy_stats": self._compile_strategy_stats(mutation_audit),
            },
        })

        return {
            "baseline_score": score_trajectory[0],
            "final_score": final_score,
            "improved_skill_md": current_md,
            "score_history": score_trajectory,
            "mutation_log": mutation_audit,
        }

    async def _score_skill(
        self, skill_md: str, scenarios: List[Dict[str, Any]], evals: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        all_evaluations = []
        total_passed = 0
        total_checks = 0
        per_eval = {e["id"]: {"passed": 0, "total": 0} for e in evals}

        for sc in scenarios:
            output = await self._dispatch_agent(
                self.executor,
                f"Execute this skill specification:\n\n{skill_md}\n\nUser request:\n{sc['input']}",
            )
            scoring_res = await self._query_agent_json(
                self.executor,
                (
                    f"Evaluate this execution output against the designated criteria.\n\n"
                    f"Input:\n{sc['input']}\n\n"
                    f"Output:\n{output}\n\n"
                    f"Criteria:\n{json.dumps(evals, indent=2)}\n\n"
                    f"Return JSON adhering to schema: {{\"results\": [{{\"eval_id\": 1, \"passed\": true, \"reason\": \"...\"}}]}}"
                ),
                fallback={"results": []},
            )
            scores = scoring_res.get("results", []) if isinstance(scoring_res, dict) else scoring_res

            for s in scores:
                eid = s.get("eval_id")
                passed = s.get("passed", False)
                if passed:
                    total_passed += 1
                total_checks += 1
                if eid in per_eval:
                    per_eval[eid]["total"] += 1
                    if passed:
                        per_eval[eid]["passed"] += 1
                all_evaluations.append({**s, "scenario_id": sc["id"]})

        return {
            "passed": total_passed,
            "total": total_checks,
            "per_eval": [
                {"eval_id": k, **v, "pass_rate": round(v["passed"] / max(v["total"], 1) * 100, 1)}
                for k, v in per_eval.items()
            ],
            "details": all_evaluations,
        }

    async def _analyze_failures(
        self, skill_md: str, scenarios: List[dict], evals: List[dict], details: List[dict]
    ) -> Dict[str, Any]:
        failures = [d for d in details if not d.get("passed")]
        if not failures:
            return {
                "diagnosis": "All benchmark evaluations passed successfully",
                "mutation_strategy": "add_constraint",
                "target_section": "N/A",
                "suggested_change": "none",
            }

        return await self._query_agent_json(
            self.analyst,
            (
                f"Diagnose these benchmark failures and determine the optimal single fix.\n\n"
                f"Skill Definition Preview:\n{skill_md[:2000]}\n\n"
                f"Failed Evaluation Details:\n{json.dumps(failures[:5], indent=2)}"
            ),
            fallback={
                "diagnosis": "Unable to diagnose failure pattern",
                "mutation_strategy": "add_constraint",
                "target_section": "unknown",
                "suggested_change": "unclear",
            },
        )

    async def _mutate_skill(self, skill_md: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        return await self._query_agent_json(
            self.mutator,
            (
                f"Apply this surgical fix to the skill specification. Make ONE targeted modification only.\n\n"
                f"Current SKILL.md:\n{skill_md}\n\n"
                f"Diagnosis: {analysis.get('diagnosis')}\n"
                f"Strategy: {analysis.get('mutation_strategy')}\n"
                f"Target Section: {analysis.get('target_section')}\n"
                f"Suggested Modification: {analysis.get('suggested_change')}"
            ),
            fallback={
                "description": "Failed to parse mutation output",
                "reasoning": "",
                "new_skill_md": skill_md,
            },
        )

    @staticmethod
    def _compile_strategy_stats(mutation_log: List[dict]) -> Dict[str, dict]:
        stats = {}
        for m in mutation_log:
            strat = m.get("strategy_type", "unknown")
            if strat not in stats:
                stats[strat] = {"total": 0, "kept": 0}
            stats[strat]["total"] += 1
            if m.get("kept"):
                stats[strat]["kept"] += 1
        return stats


SkillOptimizer = SkillForgeOptimizer
