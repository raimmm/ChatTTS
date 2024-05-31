import torch
import ChatTTS
from IPython.display import Audio

model_dict = {
    'vocos_config_path': "/root/ranpengcheng/project/ChatTTS/config/config_vocos.yaml",
    'vocos_ckpt_path': "/root/ranpengcheng/project/ChatTTS/asset/Vocos.pt",
    'dvae_config_path': "/root/ranpengcheng/project/ChatTTS/config/config_dvae.yaml",
    'dvae_ckpt_path': "/root/ranpengcheng/project/ChatTTS/asset/DVAE.pt",
    'gpt_config_path': "/root/ranpengcheng/project/ChatTTS/config/config_gpt.yaml",
    'gpt_ckpt_path': "/root/ranpengcheng/project/ChatTTS/asset/GPT.pt",
    'decoder_config_path': "/root/ranpengcheng/project/ChatTTS/config/config_decoder.yaml",
    'decoder_ckpt_path': "/root/ranpengcheng/project/ChatTTS/asset/Decoder.pt",
    'tokerizer_path': "/root/ranpengcheng/project/ChatTTS/asset/tokenizer.pt"
}

chat = ChatTTS.Chat()
chat.load_models()


# texts = ["free memory equals to props's total memory minus torch dot cuda dot memory reserved",]*3 \
#         + ["中国人民银行，国家金融监督管理总局发布通知。通知提出，对于贷款购买商品住房的居民家庭，首套住房商业性个人住房贷款最低首付款比例调整为不低于百分之十五",]*3

# wavs = chat.infer(texts, use_decoder=True)

# Audio(wavs[0], rate=24_000, autoplay=True)

# ChatGPT
from ChatTTS.experimental.llm import llm_api
API_KEY = 'sk-571b0e0042d64a329adb0925c82fa28d'  # https://www.deepseek.com/ 生成对应的api_key
client = llm_api(api_key=API_KEY,
        base_url="https://api.deepseek.com",
        model="deepseek-chat")

user_question = '四川有哪些好吃的美食呢?'
text = client.call(user_question, prompt_version = 'deepseek')
text = client.call(text, prompt_version = 'deepseek_TN')

# TTS
spk_stat = torch.load('asset/spk_stat.pt')  # https://huggingface.co/2Noise/ChatTTS/tree/main huggingface下载model
torch.manual_seed(100)
rand_spk = torch.randn(768) * spk_stat.chunk(2)[0] + spk_stat.chunk(2)[1]

params_infer_code = {'spk_emb' : rand_spk, 'temperature':.3}
params_refine_text = {'prompt':'[oral_2][laugh_0][break_6]'}
# wav = chat.infer('四川美食可多了，有麻辣火锅、宫保鸡丁、麻婆豆腐、担担面、回锅肉、夫妻肺片等，每样都让人垂涎三尺。', params_refine_text=params_refine_text, params_infer_code=params_infer_code)
# wav = chat.infer('四川美食确实以辣闻名，但也有不辣的选择。比如甜水面、赖汤圆、蛋烘糕、叶儿粑等，这些小吃口味温和，甜而不腻，也很受欢迎。', params_refine_text=params_refine_text, params_infer_code=params_infer_code)
# wav = chat.infer("""大家好，我叫李明，今年22岁，刚从四川大学计算机科学专业毕业。
#                  [uv_break] 我热爱编程，尤其是人工智能方向。[laugh]我性格开朗，喜欢团队合作，曾在校内编程大赛中获奖。
#                  [uv_break] 平时喜欢打篮球、听音乐和旅游。[laugh]希望在未来的工作中与大家一起成长！
#                  [uv_break] 谢谢！""".replace('\n', ''))
# wav = chat.infer("""chat T T S 是一款强大的对话式文本转语音模型。它有中英混读和多说话人的能力。
#                 chat T T S 不仅能够生成自然流畅的语音，还能控制[laugh]笑声啊[laugh]，
#                 停顿啊[uv_break]语气词啊等副语言现象[uv_break]。这个韵律超越了许多开源模型[uv_break]。
#                 请注意，chat T T S 的使用应遵守法律和伦理准则，避免滥用的安全风险。[uv_break]'
#                 """.replace('\n', ''))
wav = chat.infer("""人们靠近一棵大树[uv_break]，总是会赞美它的枝繁叶茂[uv_break]、繁华硕果[uv_break]，
                 [uv_break]人们总是会看见它的参天之姿[uv_break]，却从来没有人去关注他那庞大而又沉默的树根[uv_break]，
                 [uv_break]那树根埋在那阴冷[uv_break]而又黑暗的泥土里[uv_break]，无怨无悔的[uv_break]深深扎进坚硬的大地，
                 [uv_break]却也是这些无人问津的根系[uv_break]，支撑起了所有向上的力量[uv_break]和枯荣。""".replace('\n', ''), params_refine_text=params_refine_text, params_infer_code=params_infer_code)


# Audio(wav[0], rate=24_000, autoplay=True)

import soundfile as sf
sf.write("data/test16.wav", wav[0][0], 24_000, format="wav")