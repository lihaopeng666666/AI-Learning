import os
from dotenv import load_dotenv #读取项目根目录的 .env 文件，把里面的 KEY=VALUE 加载成环境变量
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("ALIYUN_BAILIAN_API_KEY"), # 从环境配置中读取密钥
    base_url="https://ws-it60fjx06fvy8l40.cn-beijing.maas.aliyuncs.com/compatible-mode/v1" # 改掉请求地址
)

response = client.chat.completions.create(
    model="qwen-max",
    messages=[
        {"role": "system", "content": "你是一个土木工程领域的助手，回答要简洁、准确"}, # 系统提示词，设定模型角色和行为风格
        # {"role": "user", "content": "用三句话解释什么是混凝土的水灰比。"},
    ],
    temperature=1.0 # 控制输出的随机性，范围通常是0到2 值越低，回答越稳定、保守；值越高，越有创造性
)

print(response.choices[0].message.content)
