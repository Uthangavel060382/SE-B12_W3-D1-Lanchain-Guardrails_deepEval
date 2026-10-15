
from app import ask_ai

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams


def test_ai_response():

    # Step 1: Prepare the question
    question = "What is LangChain?"

    # Step 2: Invoke our guarded LangChain app
    actual_answer = ask_ai(question)

    print("\nAI Answer:", actual_answer)

    # Step 3: Create the evaluation test case
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_answer,
        expected_output=(
            "LangChain is a framework for building "
            "LLM-powered applications."
        )
    )

    # Step 4: Evaluate relevance
    relevancy = AnswerRelevancyMetric(
        threshold=0.7,
        model="gpt-4o-mini"
    )

    # Step 5: Evaluate correctness
    correctness = GEval(
        name="Correctness",
        criteria=(
            "Determine whether the actual output "
            "correctly explains the expected concept."
        ),
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT
        ],
        threshold=0.7,
        model="gpt-4o-mini"
    )

    # Step 6: Assert both evaluation metrics
    assert_test(
        test_case,
        [relevancy, correctness]
    )
