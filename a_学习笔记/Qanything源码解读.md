**`qanything` 源码解读**

+ **准备工作**

  安装调试工具`debugpy`, 修改`shell` 启动脚本文件如下：

  ```shell
  # 启动qanything-server服务
  CUDA_VISIBLE_DEVICES=1 python3 -m debugpy --listen 5881 --wait-for-client  qanything_kernel/qanything_server/sanic_api.py
  ```

  端口`5881` 应于`vscode` 中  lauch.json 文件配置一致。命令行启动脚本 `bash scripts/run_for_7B_in_Linux_or_WSL.sh`  在 python 脚本里按下`F5` 进入调试状态。 

+ **执行代码**

  `model_config` 配置参数、模型数据路径、提示词模板等信息；

  `general_utils` 是一些基础函数（检查、校验、下载）等；

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

  