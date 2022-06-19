class OneLineALGO:

    def __init__(self, name: str, hyper_params: dict, cache: dict):
        # name是一条时间线的唯一标识。由于多条时间线会公用一个cache，所以一条时间线保存的变量会以name作为key
        self.name = name
        self.model_params = dict()
        self.hyper_params = hyper_params
        self.cache = cache


    # 训练模型，类似于offline retrain，第一次train
    def fit(self, df: pd.DataFrame):
        pass


    # 用于做一次性的异常检测/预测，不涉及增量训练
    def predict(self, df: pd.DataFrame):
        pass

    # 用于预测/检测的同时更新模型，先进行预测（和df无关），
    # 然后利用_update_cache结合df和上一次的模型参数更新model
    # df包含了最新的数据。用于增量训练。
    # 可以设计触发器（比如预测了指定长度的数据）触发_update_cache）
    def predict_update(self, df: pd.DataFrame):
        A, L, timestamp = self._get_cache()#取出模型参数（A，L）和最新的时间索引timestamp（暂时用不上）
        self.model_params["params"] = (A, L)
        #预测代码
        result = XXX
        #更新模型参数
        self._update_cache(self.model_params, df)
        return result


    # 用于从cache中拿出中间变量（已缓存时序索引和模型参数）（可用于增量训练模型，predict_update中调用）
    def _get_cache(self):
        A, L = self.cache.get("model_params")
        last_timestamp = self.cache.get("timestamp")
        return A, L, last_timestamp


    # 用于更新模型参数等中间变量（最新的时序索引和模型参数）并保存回cache,
    # i.e.,更新self.cache（增量训练主代码，predict_update中调用）
    def _update_cache(self, params: dict, df: pd.DataFrame):
        #更新模型参数流程
        A, L = XXX
        self.cache["model_params"] = (A, L)
        self.cache["timestamp"] =
        return self.cache


    # 预测任务中，用于生成预测结果的index，给预测出的数据构造timestamp(时间戳)，给df的构造成dataframe增加index，和历史数据无关
    def _infer_start_index(self, ts: pd.DataFrame):
        pass


    # 从dict输入中载入模型，从cache中载入？predict_update中调用，用于offline train场景，结合fit，可暂时不写这个函数
    def load_model(self, model_dict: dict):
        pass


    # 把模型保存到dict中，保存到cache？predict_update中调用，用于offline train场景，结合fit，可暂时不写这个函数
    def dump_model(self) -> dict:
        pass

