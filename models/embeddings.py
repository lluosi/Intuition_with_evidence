from __future__ import annotations


class LocalGmeEncoder:
    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def encode(self, text: str) -> list[float]:
        inputs = self.tokenizer(text or " ", return_tensors="pt", truncation=True).to(self.device)
        with self.torch.no_grad():
            vector = self.model(**inputs).last_hidden_state[:, -1, :]
        return vector.detach().cpu().float().numpy()[0].tolist()


class ApiEmbeddingEncoder:
    def __init__(self, api_key: str, api_base: str, model: str):
        from openai import OpenAI

        if not api_base:
            raise ValueError("EMBEDDING_API_BASE is required for the API embedding backend.")
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model

    def encode(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=text or " ")
        return list(response.data[0].embedding)


def create_embedding_encoder(settings) -> LocalGmeEncoder | ApiEmbeddingEncoder:
    if settings.embedding_backend == "api":
        return ApiEmbeddingEncoder(
            settings.embedding_api_key,
            settings.embedding_api_base,
            settings.embedding_model,
        )
    if settings.embedding_backend != "local":
        raise ValueError("EMBEDDING_BACKEND must be local or api.")
    return LocalGmeEncoder(settings.embedding_model)
