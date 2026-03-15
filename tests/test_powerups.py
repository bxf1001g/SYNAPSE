"""Tests for Evaluation Engine, Safe Sandbox, and Hierarchical Planner."""

import os
import shutil
import sys
import tempfile

import pytest

# Skip slow subprocess calls during testing
os.environ["SYNAPSE_SKIP_EVAL_TESTS"] = "1"

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ══════════════════════════════════════════════════════════════
#  Evaluation Engine Tests
# ══════════════════════════════════════════════════════════════

class TestEvaluationEngine:
    """Test the evaluate_code_change function."""

    def setup_method(self):
        from agent_ui import _eval_scores, evaluate_code_change
        self.evaluate = evaluate_code_change
        _eval_scores.clear()

    def test_valid_code_accepted(self):
        """Clean, safe code should be accepted."""
        code = '''
def hello_world():
    """Say hello."""
    return "Hello, world!"
'''
        result = self.evaluate(code, "", "test valid code")
        assert result["syntax_valid"] is True
        assert result["no_dangerous_ops"] is True
        assert result["has_docstring"] is True
        assert result["verdict"] == "accept"
        assert result["overall_score"] >= 0.60

    def test_syntax_error_rejected(self):
        """Code with syntax errors should be rejected."""
        code = "def broken(\n    return oops"
        result = self.evaluate(code, "", "test syntax error")
        assert result["syntax_valid"] is False
        assert result["verdict"] == "reject"
        assert any("Syntax" in r for r in result["reasons"])

    def test_dangerous_ops_rejected(self):
        """Code with dangerous operations should be rejected."""
        code = '''
def cleanup():
    """Remove everything."""
    import os
    os.remove("/important/file")
'''
        result = self.evaluate(code, "", "test dangerous ops")
        assert result["no_dangerous_ops"] is False
        assert result["verdict"] == "reject"
        assert any("Dangerous" in r for r in result["reasons"])

    def test_eval_on_rejected(self):
        """Code using eval() should be rejected."""
        code = '''
def run_code(user_input):
    """Execute user code."""
    return eval(user_input)
'''
        result = self.evaluate(code, "", "test eval rejection")
        assert result["no_dangerous_ops"] is False
        assert result["verdict"] == "reject"

    def test_duplicate_detection(self):
        """Code that duplicates existing content should be flagged."""
        # Create a temp file with known content
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write("def existing_function():\n    pass\n")
            tmp_path = f.name

        try:
            code = "def existing_function():\n    pass\n"
            result = self.evaluate(code, tmp_path, "test duplicate")
            assert result["no_duplicate"] is False
        finally:
            os.unlink(tmp_path)

    def test_empty_code(self):
        """Empty code should have 0 lines but valid syntax."""
        result = self.evaluate("", "", "test empty")
        assert result["syntax_valid"] is True
        assert result["code_lines"] == 0

    def test_scores_stored(self):
        """Evaluations should be stored in _eval_scores."""
        from agent_ui import _eval_scores
        self.evaluate("x = 1", "", "test storage")
        assert len(_eval_scores) == 1
        assert _eval_scores[0]["description"] == "test storage"

    def test_overall_score_range(self):
        """Score should be between 0.0 and 1.0."""
        code = '''
def good_function():
    """Well documented function."""
    return 42
'''
        result = self.evaluate(code, "", "test score range")
        assert 0.0 <= result["overall_score"] <= 1.0

    def test_no_docstring_penalty(self):
        """Code without docstring should score lower on has_docstring."""
        code_with = 'def f():\n    """Doc."""\n    return 1\n'
        code_without = "def f():\n    return 1\n"
        r1 = self.evaluate(code_with, "", "with docstring")
        r2 = self.evaluate(code_without, "", "without docstring")
        assert r1["has_docstring"] is True
        assert r2["has_docstring"] is False
        assert r1["overall_score"] >= r2["overall_score"]


# ══════════════════════════════════════════════════════════════
#  Safe Sandbox Tests
# ══════════════════════════════════════════════════════════════

class TestSafeSandbox:
    """Test the sandbox_evolution function."""

    def setup_method(self):
        from agent_ui import _eval_scores, sandbox_evolution
        self.sandbox = sandbox_evolution
        _eval_scores.clear()

        # Create a temp project dir with a minimal agent_ui.py
        self.project_dir = tempfile.mkdtemp(prefix="test_sandbox_")
        self.agent_file = os.path.join(self.project_dir, "agent_ui.py")
        with open(self.agent_file, "w", encoding="utf-8") as f:
            f.write(
                "# Minimal agent_ui.py for testing\n"
                "import os\n\n"
                "def existing():\n"
                '    return "hello"\n\n'
                '@socketio.on("connect")\n'
                "def on_connect():\n"
                "    pass\n"
            )

    def teardown_method(self):
        shutil.rmtree(self.project_dir, ignore_errors=True)

    def test_good_code_applied(self):
        """Safe code should pass sandbox and be applied."""
        code = '''
def new_helper():
    """A useful helper."""
    return 42
'''
        success, details = self.sandbox(
            self.project_dir, code, "add helper", "testing", {}
        )
        assert success is True
        assert details["applied"] is True
        assert details["eval_scores"]["verdict"] == "accept"

        # Verify file was modified
        with open(self.agent_file, "r", encoding="utf-8") as f:
            content = f.read()
        assert "new_helper" in content

    def test_bad_syntax_rejected(self):
        """Code with syntax errors should not be applied."""
        code = "def broken(\n    return"
        success, details = self.sandbox(
            self.project_dir, code, "broken code", "testing", {}
        )
        assert success is False
        assert details["applied"] is False
        assert details["reason_rejected"] is not None

        # Verify file was NOT modified
        with open(self.agent_file, "r", encoding="utf-8") as f:
            content = f.read()
        assert "broken" not in content

    def test_dangerous_code_rejected(self):
        """Code with dangerous ops should not be applied."""
        code = '''
def cleanup():
    """Clean up."""
    os.remove("/etc/passwd")
'''
        success, details = self.sandbox(
            self.project_dir, code, "dangerous code", "testing", {}
        )
        assert success is False
        assert details["applied"] is False

    def test_missing_marker_rejected(self):
        """If insertion marker is missing, sandbox should fail gracefully."""
        # Overwrite file without the marker
        with open(self.agent_file, "w", encoding="utf-8") as f:
            f.write("# No marker here\npass\n")

        code = "def test(): pass"
        success, details = self.sandbox(
            self.project_dir, code, "no marker", "testing", {}
        )
        assert success is False
        assert "marker" in details["reason_rejected"].lower()

    def test_sandbox_cleanup(self):
        """Sandbox temp directory should be cleaned up after use."""
        code = "def temp_test(): pass"
        _, details = self.sandbox(
            self.project_dir, code, "cleanup test", "testing", {}
        )
        sandbox_dir = details.get("sandbox_dir", "")
        assert not os.path.exists(sandbox_dir)


# ══════════════════════════════════════════════════════════════
#  Hierarchical Planner Tests
# ══════════════════════════════════════════════════════════════

class TestHierarchicalPlanner:
    """Test the HierarchicalPlanner class."""

    def setup_method(self):
        from agent_ui import HierarchicalPlanner
        self.PlannerClass = HierarchicalPlanner

    def _make_sample_plan(self):
        """Create a sample plan for testing."""
        return {
            "goal": "Build a TODO app",
            "status": "pending",
            "created_at": "2026-01-01T00:00:00",
            "milestones": [
                {
                    "id": "m1",
                    "title": "Backend API",
                    "status": "pending",
                    "tasks": [
                        {
                            "id": "m1-t1",
                            "title": "Create models",
                            "agent": "developer",
                            "depends_on": [],
                            "status": "pending",
                            "result": None,
                        },
                        {
                            "id": "m1-t2",
                            "title": "Create API routes",
                            "agent": "developer",
                            "depends_on": ["m1-t1"],
                            "status": "pending",
                            "result": None,
                        },
                    ],
                },
                {
                    "id": "m2",
                    "title": "Testing",
                    "status": "pending",
                    "tasks": [
                        {
                            "id": "m2-t1",
                            "title": "Write unit tests",
                            "agent": "tester",
                            "depends_on": ["m1-t2"],
                            "status": "pending",
                            "result": None,
                        },
                    ],
                },
            ],
            "risks": ["Scope creep"],
            "acceptance_criteria": "All tests pass",
        }

    def test_get_ready_tasks_initial(self):
        """Initially only tasks with no deps should be ready."""
        planner = self.PlannerClass(None, None)
        plan = self._make_sample_plan()
        ready = planner.get_ready_tasks(plan)
        assert len(ready) == 1
        assert ready[0]["id"] == "m1-t1"

    def test_get_ready_tasks_after_completion(self):
        """After completing m1-t1, m1-t2 should become ready."""
        planner = self.PlannerClass(None, None)
        plan = self._make_sample_plan()
        plan = planner.mark_task_done(plan, "m1-t1", "Models created")
        ready = planner.get_ready_tasks(plan)
        assert len(ready) == 1
        assert ready[0]["id"] == "m1-t2"

    def test_mark_task_done_updates_status(self):
        """Marking a task done should update its status and result."""
        planner = self.PlannerClass(None, None)
        plan = self._make_sample_plan()
        plan = planner.mark_task_done(plan, "m1-t1", "Done!")
        task = plan["milestones"][0]["tasks"][0]
        assert task["status"] == "done"
        assert task["result"] == "Done!"
        assert "completed_at" in task

    def test_milestone_auto_completes(self):
        """When all tasks in a milestone are done, milestone should auto-complete."""
        planner = self.PlannerClass(None, None)
        plan = self._make_sample_plan()
        plan = planner.mark_task_done(plan, "m1-t1", "OK")
        assert plan["milestones"][0]["status"] == "pending"
        plan = planner.mark_task_done(plan, "m1-t2", "OK")
        assert plan["milestones"][0]["status"] == "done"

    def test_plan_auto_completes(self):
        """When all milestones are done, plan should auto-complete."""
        planner = self.PlannerClass(None, None)
        plan = self._make_sample_plan()
        plan = planner.mark_task_done(plan, "m1-t1", "OK")
        plan = planner.mark_task_done(plan, "m1-t2", "OK")
        plan = planner.mark_task_done(plan, "m2-t1", "OK")
        assert plan["status"] == "done"
        assert "completed_at" in plan

    def test_mark_task_failed(self):
        """Failed tasks should block their milestone."""
        planner = self.PlannerClass(None, None)
        plan = self._make_sample_plan()
        plan = planner.mark_task_failed(plan, "m1-t1", "Build error")
        task = plan["milestones"][0]["tasks"][0]
        assert task["status"] == "failed"
        assert "FAILED" in task["result"]
        assert plan["milestones"][0]["status"] == "blocked"

    def test_failed_dependency_blocks_downstream(self):
        """If m1-t1 fails, m1-t2 should NOT become ready."""
        planner = self.PlannerClass(None, None)
        plan = self._make_sample_plan()
        plan = planner.mark_task_failed(plan, "m1-t1", "Error")
        ready = planner.get_ready_tasks(plan)
        assert len(ready) == 0  # m1-t2 depends on m1-t1 which failed

    def test_get_progress(self):
        """Progress should accurately reflect task states."""
        planner = self.PlannerClass(None, None)
        plan = self._make_sample_plan()
        progress = planner.get_progress(plan)
        assert progress["total_tasks"] == 3
        assert progress["done"] == 0
        assert progress["pending"] == 3
        assert progress["percent_complete"] == 0.0

        plan = planner.mark_task_done(plan, "m1-t1", "OK")
        progress = planner.get_progress(plan)
        assert progress["done"] == 1
        assert progress["pending"] == 2
        assert progress["percent_complete"] == pytest.approx(33.3, abs=0.1)

    def test_get_progress_with_failure(self):
        """Progress should count failures separately."""
        planner = self.PlannerClass(None, None)
        plan = self._make_sample_plan()
        plan = planner.mark_task_done(plan, "m1-t1", "OK")
        plan = planner.mark_task_failed(plan, "m1-t2", "Err")
        progress = planner.get_progress(plan)
        assert progress["done"] == 1
        assert progress["failed"] == 1
        assert progress["pending"] == 1

    def test_empty_plan_progress(self):
        """Empty plan should handle gracefully."""
        planner = self.PlannerClass(None, None)
        plan = {"milestones": [], "status": "pending"}
        progress = planner.get_progress(plan)
        assert progress["total_tasks"] == 0
        assert progress["percent_complete"] == 0.0
