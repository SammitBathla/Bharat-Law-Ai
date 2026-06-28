import os
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ip_law_agent import answer_query, classify_ip_domain


class IpLawAgentFallbackTests(unittest.TestCase):
    @patch("ip_law_agent.GoogleGenerativeAI")
    def test_classify_ip_domain_falls_back_on_quota_error(self, mock_google):
        mock_google.return_value.invoke.side_effect = Exception("429 RESOURCE_EXHAUSTED")

        domain = classify_ip_domain("How do I patent my invention?")

        self.assertEqual(domain, "Patent")

    @patch("ip_law_agent.setup_rag_chain")
    @patch("ip_law_agent.load_vector_store")
    @patch("ip_law_agent.detect_language", return_value="en")
    @patch("ip_law_agent.classify_ip_domain", return_value="Patent")
    def test_answer_query_uses_fallback_when_gemini_is_quota_exhausted(
        self,
        mock_classify,
        mock_detect_language,
        mock_load_vector_store,
        mock_setup_rag_chain,
    ):
        class DummyVectorStore:
            def similarity_search(self, query, k=3):
                return [type("Doc", (), {"page_content": "Patentable subject matter requires novelty and non-obviousness."})()]

        mock_load_vector_store.return_value = DummyVectorStore()
        mock_setup_rag_chain.return_value.invoke.side_effect = Exception("429 RESOURCE_EXHAUSTED")

        response = answer_query("How do I patent my invention?")

        self.assertEqual(response["domain"], "Patent")
        self.assertIn("fallback", response["result"].lower())
        self.assertIn("patentable", response["result"].lower())


if __name__ == "__main__":
    unittest.main()
