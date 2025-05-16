import openai


def test_openai_client_valid(openai_client: openai.OpenAI):
    models_result = openai_client.models.list()
    assert len(models_result.data) > 0
