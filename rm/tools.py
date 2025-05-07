import os
import requests, imageio, cv2
import json, sys, numpy as np
from PIL import Image
from tqdm import tqdm

def crop_image_to_square(img):
    width, height = img.size
    if width > height:
        left = (width - height) / 2
        right = width - left
        top = 0
        bottom = height
    else:
        left = 0
        right = width
        top = (height - width) / 2
        bottom = height - top
    return img.crop((left, top, right, bottom))



def story_concat():
    vpath = '/apdcephfs_cq10/share_1367250/wellszhou/code/vrag/Hunyuan_vRAG_720p_infer/backup/videos/singleref'
    from decord import VideoReader
    height, width = 1104, 832
    frames = 129
    pad = 10
    imgpath = f'{vpath}/man.png'
    vlist = [f'{vpath}/man_{i}.mp4' for i in range(1,6)]
    imgpath = f'{vpath}/woman.png'
    vlist = [f'{vpath}/woman_{i}.mp4' for i in range(1,6)]
    
    res = []
    for text in ['Reference', 'Eating breakfast','In a subway','Working at desk','Walking a dog', 'Sleeping at night']:
        w2= width
        right_put = np.ones((90, w2+pad, 3), dtype=np.uint8)*255
        color = (0,0,0)
        cv2.putText(right_put, text, (20, 60), cv2.FONT_HERSHEY_COMPLEX, 2, color, 2)
        res.append(right_put)
    text_img = np.hstack(res)
    cv2.imwrite(imgpath.replace('.png', '_text.png'), text_img)
    # return
    
    img = Image.open(imgpath).convert('RGB');   img = np.array(img.resize((width, height)))
    # img = crop_image_to_square(img)
    # img = cv2.resize(np.array(img), (width, width))
    # tmp = np.ones((height, width, 3), dtype=np.uint8)*255
    # tmp[:832 ] = img ; img = tmp
    vres=[]
    for vname in vlist:
        video = VideoReader(vname)
        sample_ids = list(range(len(video)))    
        frames = video.get_batch(sample_ids).asnumpy()
        if len(frames) < 129:
            frames = np.concatenate([frames]+[frames[-1][None]]*(129-len(frames)), axis=0)
        frames = frames[:129]
        if frames.shape[1]!=height:
            frames = np.array([cv2.resize(frame, (width, height)) for frame in frames])
        vres.append(frames)
        vres.append(np.ones_like(frames[:,:,:pad])*255)
    res = np.concatenate(vres, axis=2)

    res2 = []
    for i in range(res.shape[0]):
        tmp = np.concatenate([img, np.ones_like(img[:,:pad])*255, res[i]], axis=1)
        # breakpoint()
        tmp = np.vstack([text_img, tmp])
        # breakpoint()
        res2.append( tmp[:,:-pad])
    res = np.array(res2)

    # breakpoint()
    print('write video...')
    imageio.mimwrite(imgpath.replace('.png', '_concat.mp4'), res, fps=25, quality=5)
    # imageio.mimwrite(imgpath.replace('.png', '_concat.gif'), res, 'GIF', fps=25, loop=0)
    


def concat_video():
    vpath = '/apdcephfs_cq10/share_1367250/wellszhou/code/vrag/Hunyuan_vRAG_720p_infer/hymm.github.io/static/videos/single_diff'
    from decord import VideoReader
    height, width = 720, 1280
    frames = 129
    pad = 20
    imgpath = f'{vpath}/human_002.png'
    vlist = [f'{vpath}/{x}_002.mp4' for x in ['hunyuan', 'keling', 'vidu', 'pika', 'hailuo', 'skyreels']]
    imgpath = f'{vpath}/032.png'
    vlist = [f'{vpath}/{x}_032.mp4' for x in ['hunyuan', 'keling', 'vidu', 'pika', 'hailuo', 'skyreels']]
    # imgpath = f'{vpath}/042.png'
    # vlist = [f'{vpath}/{x}_042.mp4' for x in ['hunyuan', 'keling', 'vidu', 'pika', 'hailuo', 'skyreels']]
    
    # imgpath = f'{vpath}/item_c1.png'
    # vlist = [f'{vpath}/{x}_c1.mp4' for x in ['hunyuan', 'keling', 'vidu', 'pika', 'skyreels', 'vace']]
    # imgpath = f'{vpath}/cloth_02.png'
    # vlist = [f'{vpath}/{x}_c2.mp4' for x in ['hunyuan', 'keling', 'vidu', 'pika', 'skyreels', 'vace']]
    # imgpath = f'{vpath}/item_c3.png'
    # vlist = [f'{vpath}/{x}_c3.mp4' for x in ['hunyuan', 'keling', 'vidu', 'pika', 'skyreels', 'vace']]
    
    res = []
    for text in ['Ref', 'Ours', 'Keling', 'Vidu', 'Pika', 'Hailuo', 'Skyreels']:
    # for text in ['Ref', 'Ours', 'Keling', 'Vidu', 'Pika', 'Skyreels', 'VACE']:
        w2=height if text=='Ref' else width
        
        right_put = np.ones((180, w2+pad, 3), dtype=np.uint8)*255
        if text=='Ours':
            color = (255, 0, 0)
        else:
            color = (0,0,0)
        cv2.putText(right_put, text, (w2//3, 120), cv2.FONT_HERSHEY_COMPLEX, 5, color, 5)
        res.append(right_put)
    text_img = np.hstack(res)
    cv2.imwrite(imgpath.replace('.png', '_text.png'), text_img)
    # return
    
    img = Image.open(imgpath).convert('RGB')
    img = crop_image_to_square(img)
    img = cv2.resize(np.array(img), (height, height))
    vres=[]
    for vname in vlist:
        video = VideoReader(vname)
        sample_ids = list(range(len(video)))    
        frames = video.get_batch(sample_ids).asnumpy()
        if len(frames) < 129:
            frames = np.concatenate([frames]+[frames[-1][None]]*(129-len(frames)), axis=0)
        frames = frames[:129]
        if frames.shape[1]!=height:
            frames = np.array([cv2.resize(frame, (width, height)) for frame in frames])
        vres.append(frames)
        vres.append(np.ones_like(frames[:,:,:pad])*255)
    res = np.concatenate(vres, axis=2)

    res2 = []
    for i in range(res.shape[0]):
        tmp = np.concatenate([img, np.ones_like(img[:,:pad])*255, res[i]], axis=1)
        tmp = np.vstack([text_img, tmp])
        # breakpoint()
        res2.append( tmp[:,:-pad])
    res = np.array(res2)

    # breakpoint()
    print('write video...')
    imageio.mimwrite(imgpath.replace('.png', '_concat.mp4'), res, fps=25, quality=5)



def concat_image():
    vpath = '/apdcephfs_cq10/share_1367250/wellszhou/code/vrag/Hunyuan_vRAG_720p_infer/backup/videos/single_diff'
    from decord import VideoReader
    height, width = 720, 1280
    frames = 129
    pad = 20
    num_imgs = 4
    idx = [0,25,50,100]
    model_name = ['Ours', 'Keling', 'Vidu', 'Pika', 'Hailuo', 'Skyreels', 'VACE']
    imgpath = f'{vpath}/032.png'
    vlist = [f'{vpath}/{x}_032.mp4' for x in ['hunyuan', 'keling', 'vidu', 'pika', 'hailuo', 'skyreels', 'vace']]
    prompt = 'Prompt: In a music rehearsal room, a woman holds a violin, concentrating on playing.'
    # imgpath = f'{vpath}/new_006.png'
    # vlist = [f'{vpath}/{x}_006.mp4' for x in ['hunyuan', 'keling', 'vidu', 'pika', 'hailuo', 'skyreels', 'vace']]
    # prompt = 'Prompt: A man selects fresh fruits and vegetables from a stall at the market.'

    # model_name = ['Ours', 'Keling', 'Vidu', 'Pika', 'Skyreels', 'VACE']
    # imgpath = f'{vpath}/item_c1.png'
    # vlist = [f'{vpath}/{x}_c1.mp4' for x in ['hunyuan', 'keling', 'vidu', 'pika', 'skyreels', 'vace']]
    # prompt = 'Prompt: A dog is chasing a cat in the park.'
    # imgpath = f'{vpath}/cloth_02.png'
    # vlist = [f'{vpath}/{x}_c2.mp4' for x in ['hunyuan', 'keling', 'vidu', 'pika', 'skyreels', 'vace']]
    # prompt = 'Prompt: In an outdoor cafe, a woman wearing a dress is enjoying a delicious dessert.'
    
    img = Image.open(imgpath).convert('RGB')
    img = crop_image_to_square(img)
    img = cv2.resize(np.array(img), (height, height))
    white = (np.ones((height+pad, width*num_imgs, 3))*255).astype(np.uint8)
    white[:height, (width*num_imgs-height)//2:(width*num_imgs-height)//2+height] = img;  img = white
    cv2.putText(img, 'Reference Image', (20, 120), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,0), 3)
    vres=[img]; print(img.shape)
    

    for vid, vname in enumerate(vlist[:]):
        video = VideoReader(vname)
        sample_ids = list(range(len(video)))    
        frames = video.get_batch(sample_ids).asnumpy()
        if len(frames) < 129:
            frames = np.concatenate([frames]+[frames[-1][None]]*(129-len(frames)), axis=0)
        frames = frames[:129]
        if frames.shape[1]!=height:
            frames = np.array([cv2.resize(frame, (width, height)) for frame in frames])
        rows = []
        for i in idx:
            if i==0:
                color = (255, 0,0) if vid==0 else (0,0,0)
                frames[i][50: 160, 20:450] = 255
                cv2.putText(frames[i], model_name[vid], (20, 120), cv2.FONT_HERSHEY_COMPLEX, 3, color, 3)
            rows.append(frames[i])
        vres.append(np.hstack(rows))
        print(vid, vres[-1].shape)
        # if vid!=len(vlist)-1:
        vres.append(np.ones_like(vres[-1][:pad, :])*255)
    img = (np.ones((120, width*num_imgs, 3))*255).astype(np.uint8)
    prompt_img = cv2.putText(img, prompt, (20, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,0), 3); vres.append(prompt_img)
    res = np.vstack(vres)
    cv2.imwrite(imgpath.replace('.png', '_imgcat.jpg'), res[:,:,::-1])
    
    
def pad_image(crop_img, size, color=(255, 255, 255), resize_ratio=1):
    crop_h, crop_w = crop_img.shape[:2]
    target_w, target_h = size
    scale_h, scale_w = target_h / crop_h, target_w / crop_w
    if scale_w > scale_h:
        resize_h = int(target_h*resize_ratio)
        resize_w = int(crop_w / crop_h * resize_h)
    else:
        resize_w = int(target_w*resize_ratio)
        resize_h = int(crop_h / crop_w * resize_w)
    # print('bbox', bbox, crop_img.shape, (resize_w, resize_h))
    crop_img = cv2.resize(crop_img, (resize_w, resize_h))
    pad_left = (target_w - resize_w) // 2
    pad_top = (target_h - resize_h) // 2
    pad_right = target_w - resize_w - pad_left
    pad_bottom = target_h - resize_h - pad_top
    crop_img = cv2.copyMakeBorder(crop_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color)
    return crop_img


def multiref_concat_video():
    vpath = '/apdcephfs_cq10/share_1367250/wellszhou/code/vrag/Hunyuan_vRAG_720p_infer/backup/videos/multi_diff'
    from decord import VideoReader
    height, width = 720, 1280
    frames = 129
    pad = 20
    cls_id = 8
    imgpath = f'{vpath}/image/{cls_id}-human.jpg'
    imgpath2= f'{vpath}/image/{cls_id}.jpg'
    vlist = [f'{vpath}/{x}_{cls_id}.mp4' for x in ['hunyuan', 'keling', 'vidu', 'pika', 'skyreels', 'vace']]

    img = Image.open(imgpath).convert('RGB')
    img1 = pad_image(np.array(img), ( width//2, height))
    img2 = pad_image(np.array(Image.open(imgpath2).convert('RGB')), ( width//2, height))
    img = np.hstack([img1, img2]); print(img.shape)
    # fw, fh = img.size
    # img = cv2.resize(np.array(img), (int(height/fh*fw), height))

    res = [];  mlist = ['Ref', 'Ours', 'Keling',  'Vidu', 'Pika', 'Skyreels', 'VACE']
    for text in mlist[:1]:
        w2=img.shape[1]
        
        right_put = np.ones((180, w2+pad, 3), dtype=np.uint8)*255
        if text=='Ours':
            color = (255, 0, 0)
        else:
            color = (0,0,0)
        cv2.putText(right_put, 'Ref1   Ref2', (140, 120), cv2.FONT_HERSHEY_COMPLEX, 5, color, 5)
        res.append(right_put)
    
    
    vres=[]
    for idx, vname in enumerate(vlist):
        video = VideoReader(vname)
        sample_ids = list(range(len(video)))    
        frames = video.get_batch(sample_ids).asnumpy()
        # if idx==0:
        #     frames = frames[:, :512]
        if len(frames) < 129:
            frames = np.concatenate([frames]+[frames[-1][None]]*(129-len(frames)), axis=0)
        frames = frames[:129]
        fh, fw = frames.shape[1:3]
        if frames.shape[1]!=height:
            frames = np.array([cv2.resize(frame, (int(height/fh*fw), height)) for frame in frames])
        vres.append(frames)
        vres.append(np.ones_like(frames[:,:,:pad])*255)

        w2 = int(height/fh*fw)
        
        right_put = np.ones((180, int(height/fh*fw)+pad, 3), dtype=np.uint8)*255
        if idx==0:
            color = (255, 0, 0)
        else:
            color = (0,0,0)
        cv2.putText(right_put, mlist[idx+1], (w2//5, 120), cv2.FONT_HERSHEY_COMPLEX, 5, color, 5)
        res.append(right_put)
        print(idx, w2, frames.shape, right_put.shape)
    text_img = np.hstack(res)
    cv2.imwrite(imgpath.replace('.jpg', '_text.png'), text_img)

    res = np.concatenate(vres, axis=2)

    res2 = []
    for i in range(res.shape[0]):
        # breakpoint()
        tmp = np.concatenate([img, np.ones_like(img[:,:pad])*255, res[i]], axis=1)
        tmp = np.vstack([text_img, tmp])
        # breakpoint()
        res2.append( tmp[:,:-pad])
    res = np.array(res2)

    # breakpoint()
    print('write video...')
    imageio.mimwrite(imgpath.replace('.jpg', '_concat.mp4'), res, fps=25, quality=5)



def audio_concat_video():
    vpath = '/apdcephfs_cq10/share_1367250/wellszhou/code/vrag/Hunyuan_vRAG_720p_infer/hymm.github.io/static/videos/audioref'
    vpath = '/apdcephfs_cq10/share_1367250/wellszhou/code/vrag/Hunyuan_vRAG_720p_infer/backup/videos/audio2'
    from decord import VideoReader
    height, width = 720, 1280
    frames = 129
    pad = 10
    imgpath = f'{vpath}/woman04.png';   vlist = [f'{vpath}/cake.mp4']
    imgpath = f'{vpath}/woman03.png';   vlist = [f'{vpath}/lipstick.mp4']
    # imgpath = f'{vpath}/woman07.png';   vlist = [f'{vpath}/yujitan.mp4']
    # imgpath = f'{vpath}/man04.png';   vlist = [f'{vpath}/chengdu.mp4']
    # imgpath = f'{vpath}/man03.png';   vlist = [f'{vpath}/watch.mp4']
    # imgpath = f'{vpath}/man02.png';   vlist = [f'{vpath}/damingwangchao.mp4']
    
    img = Image.open(imgpath).convert('RGB')
    if 0:
        print(imgpath)
        w,h = img.size 
        img = img.crop((0,0,w, h//2+h//6))
    img = crop_image_to_square(img)
    img = cv2.resize(np.array(img), (height//2, height//2))
    white = (np.ones((height, height//2+pad, 3))*255).astype(np.uint8)
    white[:height//2, :height//2] = img;  img = white
    
    res = [img]*129
    imageio.mimwrite(imgpath.replace('.png', '_ref.mp4'), res, fps=25, quality=5)

    # cmd = f'ffmpeg -y -i {imgpath.replace(".png", "_ref.mp4")}  -i {vlist[0]} -filter_complex "[0:v][1:v]vstack=inputs=2[v]" -map "[v]" -map 1:a -c:v libx264 -c:a aac -strict experimental {imgpath.replace(".png", "_concat.mp4")}'
    cmd = f'ffmpeg -y -i {imgpath.replace(".png", "_ref.mp4")}  -i {vlist[0]} -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -map 1:a -c:v libx264 -c:a aac -strict experimental {imgpath.replace(".png", "_concat.mp4")}'
    
    os.system(cmd)


def resize_video():
    vpath = '/apdcephfs_cq10/share_1367250/wellszhou/code/vrag/Hunyuan_vRAG_720p_infer/hymm.github.io/static/videos/goodedit/2_video.mp4'
    vpath = '/apdcephfs_cq10/share_1367250/wellszhou/code/vrag/Hunyuan_vRAG_720p_infer/hymm.github.io/static/videos/bigone/m4.mp4'
    from decord import VideoReader
    video = VideoReader(vpath) 
    frames = video.get_batch(list(range(129))).asnumpy()
    h,w = frames.shape[1:3]
    # res = np.array([cv2.resize(x, (w//4, h//4)) for x in frames])
    res = np.array([x[:, -1280:] for x in frames])
    imageio.mimwrite(vpath.replace('.mp4', '_resize.mp4'), res, fps=25, quality=5)


def edit_resize_img():
    vpath = '/apdcephfs_cq10/share_1367250/wellszhou/code/vrag/Hunyuan_vRAG_720p_infer/hymm.github.io/static/videos/goodedit'
    from decord import VideoReader
    
    imgpath = f'{vpath}/1.png'
    vlist = [f'{vpath}/1_{x}.mp4' for x in ['video']]

    for idx, vname in enumerate(vlist):
        video = VideoReader(vname) 
        frames = video.get_batch([0,1]).asnumpy()
        h,w = frames.shape[1:3]

    img = Image.open(imgpath).convert('RGB')
    w2, h2 = img.size
    img = cv2.resize(np.array(img), (w, int(h2*w/w2)))
    # img[img<1]=255
    cv2.imwrite(imgpath.replace('.png', '_resize.png'), img[:,:,::-1])


if __name__ == "__main__":
    # story_concat()
    # concat_video()
    concat_image()
    # multiref_concat_video()
    # audio_concat_video()
    # edit_resize_img()
    # resize_video()
