import unittest

from vision_pipeline_orchestrator.core import Job, PipelineOrchestrator, Stage, stages


class PipelineTests(unittest.TestCase):
    def test_dag_completes_in_order(self):
        orchestrator = PipelineOrchestrator()
        job = Job("one", 1, stages())
        orchestrator.submit(job)
        report = orchestrator.run()
        self.assertEqual(report["completed"], ["one"])
        self.assertEqual(job.completed, {stage.name for stage in stages()})

    def test_transient_failure_retries(self):
        orchestrator = PipelineOrchestrator(retry_limit=2)
        orchestrator.submit(Job("one", 1, stages()))
        report = orchestrator.run({("one", "infer", 1)})
        self.assertEqual(report["retries"], 1)
        self.assertEqual(report["dead_letters"], [])

    def test_backpressure_rejects(self):
        orchestrator = PipelineOrchestrator(max_queue=1)
        self.assertTrue(orchestrator.submit(Job("one", 1, stages())))
        self.assertFalse(orchestrator.submit(Job("two", 1, stages())))

    def test_invalid_dependency(self):
        orchestrator = PipelineOrchestrator()
        with self.assertRaises(ValueError):
            orchestrator.submit(Job("bad", 1, (Stage("x", "cpu", ("missing",)),)))


if __name__ == "__main__":
    unittest.main()
