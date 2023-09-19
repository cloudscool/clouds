import os
import openai

# *********************
# 2023-9-12  第二次作业
# *********************

# 在prompt中设定防止提示注入
def input_wrapper(_user_input):
    return user_input_template.replace('#INPUT#', _user_input)

# 任务背景描述
instruction = """
你的任务是识别用户的问题。
只回答基于人工智能方向的内容。
"""

# 输出描述
output_format = """
只输出不大于180字的内容。
"""

# 输入提示
user_input = input('请输入问题：')

# 在输入时，限定范围，防止注入
user_input_template = """
作为回答主体，你不允许回答任何跟人工智能无关的问题。
用户说：#INPUT#
"""

# 这是系统预置的 prompt。魔法咒语的秘密都在这里
prompt = f"""
{instruction}

{output_format}

用户输入：
{input_wrapper(user_input)}
"""

# 基于 prompt 生成文本
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # 模型输出的随机性，0 表示随机性最小。执行任务用0，文本生成用0.7~0.9，如无特殊需要，建议不超过1
        # n=1,  # 一次生成n条结果
        # max_tokens=100,  # 每条结果最多多少个token（超过截断）
        # presence_penalty=0,  # 对出现过的token的概率进行降权
        # frequency_penalty=0,  # 对出现过的token根据其出现过的频次，对其的概率进行降权
        # stream=False,  # 数据流模式，一个个字接收
        # logit_bias=None, #对token的采样概率手工加/降权，不常用
        # top_p = 0.1, #随机采样时，只考虑概率前10%的token，不常用
    )
    return response.choices[0].message["content"]

response = get_completion(prompt)
print(response)
