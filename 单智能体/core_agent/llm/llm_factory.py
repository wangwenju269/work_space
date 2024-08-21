def get_llm_cls(llm_type):
    if llm_type == 'openai':
        from .openai import OpenAi
        return OpenAi
    elif llm_type == 'open_model':
        from .model_llm import ModelLLM
        return ModelLLM
    else:
        raise ValueError(f'Invalid llm_type {llm_type}')


class LLMFactory:

    @staticmethod
    def build_llm(model_name, cfg):
        llm_type = cfg[model_name].pop('type')
        llm_cls = get_llm_cls(llm_type)
        llm_cfg = cfg[model_name]
        return llm_cls(cfg=llm_cfg)
