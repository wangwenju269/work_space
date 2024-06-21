+ 外部`API`说明

  ```python
  # 脚本文件名：必须准确该出api的名字信息，eg:safe_detech.py   api.py
  # 代码格式 格式如下：
  
  class your_api_name: #api的名字，首字母大写
      def __init__(self) -> None:
          self.access_token_url = "" #可以访问的 url
      
      def run(self, *args, **kwargs):
          """
          api 调用接口: 详细说明 api 输入类型，输出的结果。（必须要有）
          each api should implement this function to generate response
          Args:
              prompt (str): prompt
          Returns:
              str: response
         
          """
     def excute(self, *args, **kwargs):
         """
         中间执行的代码，可选
         """       
  
  ```
  
  **格式：**

  ```PYTHON
    # 工具注册的模板  
          {
          'name_for_human':
              'math',          # 工具名
          'name_for_model':
              'math',          # 工具名
          'description_for_model':
              'Useful for when you need to answer questions about Math.',    # 工具描述
          'parameters': [                                                    # 工具参数
                  {
                  "name": "query",                                            # 参数名 
                  "type": "string",                                           # 参数类型 
                  "description": "the problem to solved by math",             # 参数描述
                  'required': True                                            # 外部 api 调用是否需要该参数 
                  }
              ],
          }, 
  ```
  
  



