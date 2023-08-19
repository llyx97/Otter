import mimetypes
import os
from typing import Union
import cv2
import requests
import torch
import transformers
from PIL import Image
import sys
from tqdm import tqdm
import random, argparse, re, json
import numpy as np

sys.path.append("../..")
# make sure you can properly access the otter folder
from otter.modeling_otter import OtterForConditionalGeneration

# Disable warnings
requests.packages.urllib3.disable_warnings()

# ------------------- Utility Functions -------------------


def get_content_type(file_path):
    content_type, _ = mimetypes.guess_type(file_path)
    return content_type


# ------------------- Image and Video Handling Functions -------------------


def extract_frames(video_path, num_frames=16):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = total_frames // num_frames
    frames = []

    for i in range(num_frames):
        video.set(cv2.CAP_PROP_POS_FRAMES, i * frame_step)
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame).convert("RGB")
            frames.append(frame)

    video.release()
    return frames


def get_image(url: str) -> Union[Image.Image, list]:
    if "://" not in url:  # Local file
        content_type = get_content_type(url)
    else:  # Remote URL
        content_type = requests.head(url, stream=True, verify=False).headers.get("Content-Type")

    if "image" in content_type:
        if "://" not in url:  # Local file
            return Image.open(url)
        else:  # Remote URL
            return Image.open(requests.get(url, stream=True, verify=False).raw)
    elif "video" in content_type:
        video_path = "temp_video.mp4"
        if "://" not in url:  # Local file
            video_path = url
        else:  # Remote URL
            with open(video_path, "wb") as f:
                f.write(requests.get(url, stream=True, verify=False).content)
        frames = extract_frames(video_path)
        if "://" in url:  # Only remove the temporary video file if it was downloaded
            os.remove(video_path)
        return frames
    else:
        raise ValueError("Invalid content type. Expected image or video.")


# ------------------- OTTER Prompt and Response Functions -------------------


def get_formatted_prompt(prompt: str) -> str:
    return f"<image>User: {prompt} GPT:<answer>"


def get_response(input_data, prompt: str, model=None, image_processor=None, tensor_dtype=None) -> str:
    if isinstance(input_data, Image.Image):
        vision_x = image_processor.preprocess([input_data], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
    elif isinstance(input_data, list):  # list of video frames
        vision_x = image_processor.preprocess(input_data, return_tensors="pt")["pixel_values"].unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError("Invalid input data. Expected PIL Image or list of video frames.")

    lang_x = model.text_tokenizer(
        [
            get_formatted_prompt(prompt),
        ],
        return_tensors="pt",
    )

    # Get the data type from model's parameters
    model_dtype = next(model.parameters()).dtype

    # Convert tensors to the model's data type
    vision_x = vision_x.to(dtype=model_dtype)
    lang_x_input_ids = lang_x["input_ids"]
    lang_x_attention_mask = lang_x["attention_mask"]

    bad_words_id = model.text_tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
    generated_text = model.generate(
        vision_x=vision_x.to(model.device),
        lang_x=lang_x_input_ids.to(model.device),
        attention_mask=lang_x_attention_mask.to(model.device),
        max_new_tokens=512,
        num_beams=3,
        no_repeat_ngram_size=3,
        bad_words_ids=bad_words_id,
    )
    parsed_output = (
        model.text_tokenizer.decode(generated_text[0])
        .split("<answer>")[-1]
        .lstrip()
        .rstrip()
        .split("<|endofchunk|>")[0]
        .lstrip()
        .rstrip()
        .lstrip('"')
        .rstrip('"')
    )
    return parsed_output

def extract_qca(llm_output):
    # extract question, choices, answer and element from llm output
    questions, choices, answers, element = [], [], [], None
    for line in llm_output.split('\n'):
        if line.startswith('About'):
            element = line.replace('About ', '')
        elif line.startswith('Q:'):
            questions.append(line.replace('Q: ', ''))
        elif line.startswith('Choices: '):
            choices_ = line.replace('Choices: ', '').split(', ')
            choices.append(choices_)
        elif line.startswith('A:'):
            answers.append(line.replace('A: ', ''))
    if not (len(questions)==len(answers) and len(questions)==len(choices)) or len(questions)==0 or len(element)==0:
        print(f"Fail to extract question, choice, answer from the LLM output \n{llm_output}")
    return questions, choices, answers, element

def load_vqa(args):
    # load the dataset of questions, choices and answers from the llm outputs
    file_path = args.videoqa_file
    fetv_file = args.eval_data_path
    video_ids = []
    with open(fetv_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            video_ids.append(str(data["video_id"]))
    datas = {}
    qid = 0
    caption_ids = []
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            for llm_output in data['gen_questions'].values():
                questions, choices, answers, element = extract_qca(llm_output)
                for q, c, a in zip(questions, choices, answers):
                    datas[qid] = {'caption_id': data['input_id'], 'question': q}
                    caption_ids.append(data['input_id'])
                    qid += 1
    return datas, list(set(caption_ids)), video_ids

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


video_paths = {
        "videofusion": '/home/liuyuanxin/Ask-Anything/video_chat/manual_eval/damo-text2video_outputs/16frames',
        "text2video-zero": '/home/liuyuanxin/Ask-Anything/video_chat/manual_eval/text2video-zero_outputs/8fps_16frames',
        "cogvideo": '/home/liuyuanxin/Ask-Anything/video_chat/manual_eval/cogvideo_outputs/videos',
        "ground-truth": '/home/liuyuanxin/Ask-Anything/video_chat/manual_eval/real_videos'
    }

def build_video_files(args, video_ids):
    video_root_path = video_paths[args.video_model]
    if args.video_model=='ground-truth':
        video_files = {}
        for i, vid in enumerate(video_ids):
            video_files[i] = os.path.join(video_root_path, f"{vid}.mp4") if vid!='None' else None
    else:
        video_files = sorted(os.listdir(video_root_path))
        video_files = [os.path.join(video_root_path, file) for file in video_files 
                                if file.endswith('.gif') or file.endswith('.mp4') or file.endswith('.avi')]
        video_files = {int(re.findall("\d+", os.path.split(fn)[1])[0]): fn for fn in video_files}
    return video_files

def save_answer(answers, save_file):
    # answers (Dict): {data_id (int): answer (str)}
    with open(save_file, 'w') as f:
        for id in answers:
            dumped = json.dumps({id: answers[id]})
            f.write(dumped+'\n')

def ask_dataset(questions, model, video_files, save_file):
    """
        Ask a dataset of questions with corresponding videos
        Args
            questions (Dict): a dict of textual questions, indexed by data_id
            video_files (List): a list of strings of the video paths
            save_file (Str): the file to save the results
    """
    answers = {}
    for data_id, question in tqdm(questions.items()):
        video_file = video_files[question['caption_id']]
        frames_list = get_image(video_file)
        print(video_file)
        print(f"\nPrompt: {question['question']}")
        response = get_response(frames_list, question['question'], model, image_processor, tensor_dtype)
        print(f"Response: {response}")
        answers[data_id] = {'caption_id': question['caption_id'], 'answer': response}
        save_answer(answers, save_file)
    return answers


# ------------------- Main Function -------------------
if "__main__" == __name__:
    # config
    parser = argparse.ArgumentParser()     
    parser.add_argument('--eval_data_path', default='../../data/fetv_data.json')        
    parser.add_argument('--save_path', default=None)        
    parser.add_argument('--video_model', default='videofusion')   
    parser.add_argument('--videoqa_file', default='../../data/question_yesno_000_099.json', help='the file containing Video QA questions')
    parser.add_argument('--multi_round', default=1, type=int, help='whether to run multiple times with different seeds')    
    args = parser.parse_args()

    args.device = torch.device("cuda:0")

    questions, caption_ids, video_ids = load_vqa(args)
    video_files = build_video_files(args, video_ids)
    video_files = {id: vf for id, vf in video_files.items() if id in caption_ids}

    if args.save_path is None:
        id_range = args.videoqa_file.split('/')[-1].replace('question_yesno_', '').replace('.json', '')
        save_path = f"answers/{args.video_model}/{id_range}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    load_bit = "fp32"
    if load_bit == "fp16":
        precision = {"torch_dtype": torch.float16}
    elif load_bit == "bf16":
        precision = {"torch_dtype": torch.bfloat16}
    elif load_bit == "fp32":
        precision = {"torch_dtype": torch.float32}

    # This model version is trained on MIMIC-IT DC dataset.
    model = OtterForConditionalGeneration.from_pretrained("../../ckpts", device_map="auto", **precision)
    tensor_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[load_bit]

    model.text_tokenizer.padding_side = "left"
    tokenizer = model.text_tokenizer
    image_processor = transformers.CLIPImageProcessor()
    model.eval()

    for seed in range(args.multi_round):
        set_seed(seed)
        save_file = os.path.join(save_path, f"seed{seed}.json")
        if os.path.isfile(save_file):
            continue
        print(f"\nSeed{seed}\n")
        answers = ask_dataset(questions, model, video_files, save_file)
