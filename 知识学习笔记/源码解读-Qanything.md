+ **`qanything` 源码解读**

  + `Shell` 启动命令：

    ```shell
  bash scripts/run_for_7B_in_Linux_or_WSL.sh
    ```

    打开 `shell`  脚本，调用执行 `shell` 和参数信息：
  
    ```shell
  bash scripts/base_run.sh -s 'LinuxOrWSL' -m 19530 -q 8777 -M 7B
    ```

    命令行参数解析：
  
    ```shell
    while getopts ":s:m:q:M:cob:k:n:l:w:" opt; do
      case $opt in
      s) system="$OPTARG"
        ;;
        m) milvus_port="$OPTARG"
        ;;
        q) qanything_port="$OPTARG"
        ;;
        M) model_size="$OPTARG"
        ;;
        c) use_cpu=true
        ;;
        o) use_openai_api=true
        ;;
        b) openai_api_base="$OPTARG"
        ;;
        k) openai_api_key="$OPTARG"
        ;;
        n) openai_api_model_name="$OPTARG"
        ;;
        l) openai_api_context_length="$OPTARG"
        ;;
        w) workers="$OPTARG"
        ;;
        \?) echo "Invalid option -$OPTARG" >&2
        ;;
      esac
    done
    ```
  
  + **准备工作**
  
    安装调试工具`debugpy`, 修改`shell` 启动脚本文件如下：
  
    ```shell
    # 启动qanything-server服务
    CUDA_VISIBLE_DEVICES=1 python3 -m debugpy --listen 5881 --wait-for-client  qanything_kernel/qanything_server/sanic_api.py
    ```
  
    端口`5881` 应于`vscode` 中 `lauch.json ` 文件配置一致。命令行启动脚本 `bash scripts/run_for_7B_in_Linux_or_WSL.sh`  在 python 脚本里按下`F5` 进入调试状态。 
  
  + **执行Python代码**
  
    服务的入口地址是 `sanic_api` 脚本，先获取各种路径信息 (当前目录、父目录、根目录 ，后 `model_config` 导入必要的配置参数、提示词模板等信息；
  
    （如：设置rerank的batch大小、设置embed的batch大小、设置多线程worker数量、`nltk_data`,` pdf_to_markdown,` `ocr_models`文件解析模型路径，和向量检索库、文件分块、召回参数、模型下载路径等；）
  
    `general_utils` 是一些基础函数（检查、校验、下载）等，函数集合：
  
    ```python
    __all__ = ['write_check_file', 'isURL', 'format_source_documents', 'get_time', 'safe_get', 'truncate_filename',
               'read_files_with_extensions', 'validate_user_id', 'get_invalid_user_id_msg', 'num_tokens', 'download_file', 
               'get_gpu_memory_utilization', 'check_package_version', 'simplify_filename', 'check_and_transform_excel',
               'export_qalogs_to_excel', 'get_table_infos']
    ```
    
    记录下载依赖的脚本代码：
    
    ```python
    def download_file(url, filename):
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(filename, 'wb') as file:
            for data in response.iter_content(chunk_size=1024):
                progress_bar.update(len(data))
                file.write(data)
    
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
    """
    url :  'https://aiinfra.pkgs.visualstudio.com/PublicPackages/_apis/packaging/feeds/9387c3aa-d9ad-4513-968c-383f6f7f53b8/pypi/packages/onnxruntime-gpu/versions/1.17.1/onnxruntime_gpu-1.17.1-cp310-cp310-manylinux_2_28_x86_64.whl/content'      
    filename : 'onnxruntime_gpu-1.17.1-cp310-cp310-manylinux_2_28_x86_64.whl'
    """        
    ```
    
    采用加速推理框架 `vllm` ,   安装命令如下：
    
    ```python
    ! pip install vllm==0.2.7 -i https://pypi.mirrors.ustc.edu.cn/simple/ --trusted-host pypi.mirrors.ustc.edu.cn
    ```
    
    将原有参数(`ArgumentParser`)  重新封装一层:
    
    ```python
            from vllm.engine.arg_utils import AsyncEngineArgs
            parser = AsyncEngineArgs.add_cli_args(parser)
            args = parser.parse_args()
    ```
    
    如果模型不存在, 下载模型:
    
    ```python
    from modelscope import snapshot_download
    model_download_params = {
                                'model_id': 'netease-youdao/Qwen-7B-QAnything', 
                                'revision': 'master',
                                'cache_dir': './path'
                              }
    snapshot_download(**model_download_params)
    ```
    
    ```python
    # 文件夹里子文件 copy 操作, 更替文件名
    import subprocess
    subprocess.check_output(['ln', '-s', cache_dir, LOCAL_RERANK_PATH], text=True)
    ```
    
    `Qanything`  针对不同文档加载各自`Loader` , 本质还是`langchain_community` 的 `UnstructuredFileLoader`. 例如：
    
    ```python
    + __all__ = [
    
        "UnstructuredPaddleImageLoader",
    
        "UnstructuredPaddlePDFLoader",
    
        "UnstructuredPaddleAudioLoader",
    
        ''MyRecursiveUrlLoader''
    
      ]
    
    ```
    
    ```mermaid
    flowchart LR
        subgraph langchain_community.document_loaders
        direction LR
        UnstructuredFileLoader --> UnstructuredPaddleImageLoader 
        UnstructuredFileLoader --> UnstructuredPaddlePDFLoader 
        UnstructuredFileLoader --> UnstructuredPaddleAudioLoader 
        UnstructuredFileLoader --> UnstructuredWordDocumentLoader 
        UnstructuredFileLoader --> UnstructuredExcelLoader 
        UnstructuredFileLoader --> UnstructuredPDFLoader 
        UnstructuredFileLoader --> UnstructuredEmailLoader 
        UnstructuredFileLoader --> UnstructuredPowerPointLoader 
        end
        
        subgraph langchain_community.document_loaders.base
        BaseLoader --> MyRecursiveUrlLoader 
        BaseLoader --> TextLoader 
        BaseLoader --> CSVLoader 
        end
        
        
    ```
    
    `LocalDocQA` : **核心主体**
    
    ```mermaid
    flowchart LR
        direction LR
        LocalDocQA --o llm -.class.- OpenAICustomLLM
        LocalDocQA --o embeddings  -.class.- EmbeddingOnnxBackend
        LocalDocQA --o faiss_client -.class.- FaissClient
        LocalDocQA --o mysql_client -.class.- KnowledgeBaseManager
        LocalDocQA --o local_rerank_backend -.class.- RerankBackend
        LocalDocQA --o ocr_reader -.class.- OCRQAnything
        FaissClient --o embeddings
        FaissClient --o mysql_client
    
    ```
    
    
    
    ```python
    class LocalDocQA:
        def __init__(self):
            self.llm: object = None
            self.embeddings: EmbeddingBackend = None
            self.top_k: int = VECTOR_SEARCH_TOP_K
            self.chunk_size: int = CHUNK_SIZE
            self.score_threshold: int = VECTOR_SEARCH_SCORE_THRESHOLD
            self.faiss_client: FaissClient = None
            self.mysql_client: KnowledgeBaseManager = None
            self.local_rerank_backend: RerankBackend = None
            self.ocr_reader: OCRQAnything = None
            self.mode: str = None
            self.use_cpu: bool = True
            self.model: str = None
            
            
        def init_cfg(self, args=None):
            self.rerank_top_k = int(args.model_size[0])
            self.use_cpu = args.use_cpu
            if args.use_openai_api:
                self.model = args.openai_api_model_name
            else:
                self.model = args.model.split('/')[-1]
            if platform.system() == 'Linux':
                if args.use_openai_api:
                    self.llm: OpenAILLM = OpenAILLM(args)
                else:
                    from qanything_kernel.connector.llm.llm_for_fastchat import OpenAICustomLLM
                    self.llm: OpenAICustomLLM = OpenAICustomLLM(args)
                from qanything_kernel.connector.rerank.rerank_onnx_backend import RerankOnnxBackend
                from qanything_kernel.connector.embedding.embedding_onnx_backend import EmbeddingOnnxBackend
                self.local_rerank_backend: RerankOnnxBackend = RerankOnnxBackend(self.use_cpu)
                self.embeddings: EmbeddingOnnxBackend = EmbeddingOnnxBackend(self.use_cpu)
            else:
                if args.use_openai_api:
                    self.llm: OpenAILLM = OpenAILLM(args)
                else:
                    from qanything_kernel.connector.llm.llm_for_llamacpp import LlamaCPPCustomLLM
                    self.llm: LlamaCPPCustomLLM = LlamaCPPCustomLLM(args)
                from qanything_kernel.connector.rerank.rerank_torch_backend import RerankTorchBackend
                from qanything_kernel.connector.embedding.embedding_torch_backend import EmbeddingTorchBackend
                self.local_rerank_backend: RerankTorchBackend = RerankTorchBackend(self.use_cpu)
                self.embeddings: EmbeddingTorchBackend = EmbeddingTorchBackend(self.use_cpu)
            self.mysql_client = KnowledgeBaseManager()
            self.ocr_reader = OCRQAnything(model_dir=OCR_MODEL_PATH, device="cpu")  # 省显存
            debug_logger.info(f"OCR DEVICE: {self.ocr_reader.device}")
            self.faiss_client = FaissClient(self.mysql_client, self.embeddings)
    ```
    
    
    
    `handler.py` 前端按钮执行函数：
    
    ```python
    __all__ = ["new_knowledge_base", "upload_files", "list_kbs", "list_docs", "delete_knowledge_base", "delete_docs",
               "rename_knowledge_base", "get_total_status", "clean_files_by_status", "upload_weblink", "local_doc_chat",
               "document", "new_bot", "delete_bot", "update_bot", "get_bot_info", "upload_faqs", "get_file_base64",
               "get_qa_info"]
    ```
    
    