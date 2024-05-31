import torch
import ChatTTS
from IPython.display import Audio
import soundfile as sf

# 加载本地模型目录
model_dict = {
    'vocos_config_path': "/root/ranpengcheng/project/ChatTTS/config/config_vocos.yaml",
    'vocos_ckpt_path': "/root/ranpengcheng/project/ChatTTS/asset/Vocos.pt",
    'dvae_config_path': "/root/ranpengcheng/project/ChatTTS/config/config_dvae.yaml",
    'dvae_ckpt_path': "/root/ranpengcheng/project/ChatTTS/asset/DVAE.pt",
    'gpt_config_path': "/root/ranpengcheng/project/ChatTTS/config/config_gpt.yaml",
    'gpt_ckpt_path': "/root/ranpengcheng/project/ChatTTS/asset/GPT.pt",
    'decoder_config_path': "/root/ranpengcheng/project/ChatTTS/config/config_decoder.yaml",
    'decoder_ckpt_path': "/root/ranpengcheng/project/ChatTTS/asset/Decoder.pt",
    'tokenizer_path': "/root/ranpengcheng/project/ChatTTS/asset/tokenizer.pt"
}

chat = ChatTTS.Chat()
chat.load_models(model_dict, source='local')

# TTS
# spk_stat的size:[1536]
spk_stat = torch.load('asset/spk_stat.pt')  # https://huggingface.co/2Noise/ChatTTS/tree/main huggingface下载model
# torch.manual_seed(100)
# rand_spk = torch.randn(768) * spk_stat.chunk(2)[0] + spk_stat.chunk(2)[1]
rand_spk = torch.full(size=(768, ), fill_value=0.1) * spk_stat.chunk(2)[0] + spk_stat.chunk(2)[1]

input_content = """
                chat T T S 是一款强大的对话式文本转语音模型。它有中英混读和多说话人的能力。
                chat T T S 不仅能够生成自然流畅的语音，还能控制[laugh]笑声啊[laugh]，
                停顿啊[uv_break]语气词啊等副语言现象[uv_break]。这个韵律超越了许多开源模型[uv_break]。
                请注意，chat T T S 的使用应遵守法律和伦理准则，避免滥用的安全风险。[uv_break]'
                """.replace('\n', '')
params_infer_code = {'spk_emb' : rand_spk, 'temperature':.3}
params_refine_text = {'prompt':'[oral_2][laugh_0][break_4]'}
wav = chat.infer(input_content, params_refine_text=params_refine_text, params_infer_code=params_infer_code, use_decoder=True)
output_file = "output/test20.wav"
sf.write(output_file, wav[0][0], 24_000, format="wav")