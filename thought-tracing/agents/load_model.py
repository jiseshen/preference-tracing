from .gpt import AsyncConversationalGPTBaseAgent, ConversationalGPTBaseAgent, O1BaseAgent, O1MiniAgent

def load_model(model_name, num_gpus=2, mode="async", **kwargs):
    if model_name in ["o1-preview-2024-09-12", "o1-2024-12-17", "o3-mini-2025-01-31"]:
        model = O1BaseAgent({'model': model_name, **kwargs})
    elif model_name in ["o1-mini-2024-09-12"]:
        model = O1MiniAgent({'model': model_name, **kwargs})
    elif model_name in ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06", "gpt-4o-2024-05-13", "gpt-4-turbo-2024-04-09", "gpt-3.5-turbo-0301", "gpt-4-0314", "gpt-4-0125-preview", "gpt-4-0613", "gpt-3.5-turbo-0125"]:
        if mode == "async":
            model = AsyncConversationalGPTBaseAgent({'model': model_name, **kwargs})
        elif mode == "non-async":
            model = ConversationalGPTBaseAgent({'model': model_name, **kwargs})
    elif model_name in ["gpt-4-turbo-nonasync", "gpt-4o-nonasync"]:
        model = ConversationalGPTBaseAgent({'model': model_name.removesuffix("-nonasync"), **kwargs})
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

    return model
