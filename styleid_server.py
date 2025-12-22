import os, torch, numpy as np, io, random, copy, traceback, asyncio
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, Response
import uvicorn
from omegaconf import OmegaConf
from einops import rearrange
import torchvision.transforms as transforms
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

app = FastAPI()
gpu_lock = asyncio.Lock()

# --- CONFIGURATION ---
DEVICE = torch.device("cuda")
MODEL_CONFIG = "/workspace/Stable-Diffusion-Style-VietNam/models/ldm/stable-diffusion-v1/v1-inference.yaml"
CKPT = "/workspace/Stable-Diffusion-Style-VietNam/models/ldm/stable-diffusion-v1/model.ckpt"
PRECISION_SCOPE = torch.cuda.amp.autocast
DDIM_STEPS = 50
ATTN_LAYERS = [6, 7, 8, 9, 10, 11]

MODEL, SAMPLER, UC = None, None, None

# Class giả lập 'opt' để tương thích với hàm feat_merge chuẩn của bạn
class Opt:
    def __init__(self, gamma, T):
        self.gamma = gamma
        self.T = T

def load_model():
    global MODEL, SAMPLER, UC
    config = OmegaConf.load(MODEL_CONFIG)
    pl_sd = torch.load(CKPT, map_location="cpu")
    MODEL = instantiate_from_config(config.model)
    MODEL.load_state_dict(pl_sd["state_dict"], strict=False)
    MODEL.to(DEVICE).eval()
    SAMPLER = DDIMSampler(MODEL)
    SAMPLER.make_schedule(ddim_num_steps=DDIM_STEPS, ddim_eta=0.0, verbose=False)
    UC = MODEL.get_learned_conditioning([""])
    print("✅ StyleID Server Ready with Standard feat_merge.")

# --- SỬ DỤNG HÀM FEAT_MERGE CHUẨN CỦA BẠN ---
def feat_merge(opt, cnt_feats, sty_feats, start_step=0):
    feat_maps = [{'config': {
                'gamma':opt.gamma,
                'T':opt.T,
                'timestep':_,
                }} for _ in range(50)]

    for i in range(len(feat_maps)):
        if i < (50 - start_step):
            continue
        cnt_feat = cnt_feats[i]
        sty_feat = sty_feats[i]
        ori_keys = sty_feat.keys()

        for ori_key in ori_keys:
            if ori_key[-1] == 'q':
                feat_maps[i][ori_key] = cnt_feat[ori_key]
            if ori_key[-1] == 'k' or ori_key[-1] == 'v':
                feat_maps[i][ori_key] = sty_feat[ori_key]
    return feat_maps

def preprocess(pil_img):
    image = transforms.CenterCrop(min(pil_img.size))(pil_img).resize((512, 512), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    return (2. * torch.from_numpy(image) - 1.).to(DEVICE)

def adain(cnt_feat, sty_feat):
    cnt_m, cnt_s = cnt_feat.mean(dim=[0,2,3], keepdim=True), cnt_feat.std(dim=[0,2,3], keepdim=True)
    sty_m, sty_s = sty_feat.mean(dim=[0,2,3], keepdim=True), sty_feat.std(dim=[0,2,3], keepdim=True)
    return ((cnt_feat - cnt_m) / cnt_s) * sty_s + sty_m

@app.post("/transform")
async def transform(
    file: UploadFile = File(...), 
    style_name: str = Form(...), 
    gamma: float = Form(0.75), 
    T: float = Form(1.5),
    start_step: int = Form(49)
):
    content_bytes = await file.read()
    
    async with gpu_lock:
        local_feat_maps = [{} for _ in range(50)]
        
        try:
            content_pil = Image.open(io.BytesIO(content_bytes)).convert("RGB")
            init_cnt = preprocess(content_pil)
            style_dir = f"/workspace/Stable-Diffusion-Style-VietNam/data/sty/{style_name}"
            style_path = os.path.join(style_dir, random.choice(os.listdir(style_dir)))
            init_sty = preprocess(Image.open(style_path).convert("RGB"))

            with torch.no_grad(), PRECISION_SCOPE():
                def cb(i):
                    # Lấy danh sách timesteps và đảo ngược theo định dạng list của Python
                    t_list = SAMPLER.ddim_timesteps[::-1].tolist()
                    
                    if i in t_list:
                        t_idx = t_list.index(i)
                        for b_idx in ATTN_LAYERS:
                            # Truy cập vào UNet output blocks để lấy đặc trưng Attention
                            block = MODEL.model.diffusion_model.output_blocks[b_idx]
                            if len(block) > 1:
                                # Lấy Q, K, V từ lớp Self-Attention đầu tiên của Transformer block
                                qkv = block[1].transformer_blocks[0].attn1
                                local_feat_maps[t_idx][f"output_block_{b_idx}_self_attn_q"] = qkv.q
                                local_feat_maps[t_idx][f"output_block_{b_idx}_self_attn_k"] = qkv.k
                                local_feat_maps[t_idx][f"output_block_{b_idx}_self_attn_v"] = qkv.v

                # Inversion - Content
                z_cnt = MODEL.get_first_stage_encoding(MODEL.encode_first_stage(init_cnt))
                cnt_z_enc, _ = SAMPLER.encode_ddim(z_cnt, num_steps=50, unconditional_conditioning=UC, callback_ddim_timesteps=50, img_callback=lambda pred, xt, i: cb(i))
                cnt_feats = copy.deepcopy(local_feat_maps)
                
                # Inversion - Style
                local_feat_maps = [{} for _ in range(50)]
                z_sty = MODEL.get_first_stage_encoding(MODEL.encode_first_stage(init_sty))
                sty_z_enc, _ = SAMPLER.encode_ddim(z_sty, num_steps=50, unconditional_conditioning=UC, callback_ddim_timesteps=50, img_callback=lambda pred, xt, i: cb(i))
                sty_feats = copy.deepcopy(local_feat_maps)

                # Sampler với hàm trộn chuẩn
                # Tạo đối tượng opt từ tham số nhận được qua Form
                current_opt = Opt(gamma=gamma, T=T)
                merged = feat_merge(current_opt, cnt_feats, sty_feats, start_step=start_step)
                
                samples, _ = SAMPLER.sample(S=50, batch_size=1, shape=[4, 64, 64], unconditional_conditioning=UC, x_T=adain(cnt_z_enc, sty_z_enc), injected_features=merged, start_step=start_step)
                
                decoded = torch.clamp((MODEL.decode_first_stage(samples) + 1.0) / 2.0, min=0.0, max=1.0)
                out_np = (rearrange(decoded[0].cpu().numpy(), 'c h w -> h w c') * 255).astype(np.uint8)
                final_img = Image.fromarray(out_np).resize((1024, 1024), resample=Image.Resampling.LANCZOS)
                
            buf = io.BytesIO()
            final_img.save(buf, format='PNG')
            return Response(content=buf.getvalue(), media_type="image/png")

        except Exception as e:
            print(f"❌ GPU Error: {str(e)}")
            traceback.print_exc()
            return Response(content=content_bytes, media_type="image/png")

if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)