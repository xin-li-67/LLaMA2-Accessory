import sys
import os
sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0])

#from accessory.model.meta import MetaModel
from model.meta import MetaModel
import argparse
import torch


from PIL import Image

from util import misc
from fairscale.nn.model_parallel import initialize as fs_init

from data.alpaca import transform_val, format_prompt
from util.tensor_parallel import load_tensor_parallel_model_list
from util.quant import quantize

import torch
import os
import json
from tqdm import tqdm
import shortuuid

from LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from LLaVA.llava.conversation import conv_templates, SeparatorStyle
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math


def get_args_parser():
    parser = argparse.ArgumentParser('llava_benchmark evaluation', add_help=False)
    # Model parameters
    parser.add_argument('--llama_type', default='llama_qformerv2', type=str, metavar='MODEL',
                        help='type of llama')
    parser.add_argument('--llama_config', default='/path/to/params.json', type=str, nargs="+",
                        help='Path to llama model config')
    parser.add_argument('--tokenizer_path', type=str, default="../tokenizer.model",
                        help='path to tokenizer.model')

    parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str, nargs="+",
                        help='directory containing pre-trained checkpoints')

    parser.add_argument('--device', default='cuda',
                        help='device for inference')
    parser.add_argument('--model_parallel_size', default=1, type=int)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--quant', action="store_true", default=False,
                        help="enable quantization")
    
    #llava benchmark setting
    parser.add_argument("--image_folder", type=str, default="path/to/images")
    parser.add_argument("--conv_mode", type=str, default="llava_llama_2")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--max_gen_len", type=int, default=516)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--question_file", type=str, default="path/to/questions.jsonl")
    parser.add_argument("--answers_file", type=str, default="yourpath/tosave/answers.jsonl")
    
    return parser

args = get_args_parser().parse_args()



def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

# define the model
misc.init_distributed_mode(args)
fs_init.initialize_model_parallel(args.model_parallel_size)
model = MetaModel(args.llama_type, args.llama_config, args.tokenizer_path, with_visual=True)
print(f"load pretrained from {args.pretrained_path}")
load_tensor_parallel_model_list(model, args.pretrained_path)

if args.quant:
    print("Quantizing model to 4bit!")

    from transformers.utils.quantization_config import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig.from_dict(
        config_dict={
            "load_in_8bit": False,
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
        },
        return_unused_kwargs=False,
    )
    quantize(model, quantization_config)
    
#print("Model = %s" % str(model))
model.bfloat16().cuda()

@ torch.inference_mode()
def generate_output(img_path,prompt):
   
    if img_path is not None:
        image = Image.open(img_path).convert('RGB')
        
        image = transform_val(image).unsqueeze(0)
    else:
        image = None

    _prompt = format_prompt(prompt)

    if image is not None:
        image = image.cuda()

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        results = model.generate([_prompt], image, max_gen_len=512, temperature=0.1, top_p=0.7)
    text_output = results[0].strip()
    
    return text_output

def eval_llava_benchmark(args):
    model_name='model_name'
    disable_torch_init()
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        print("__idx,qs____")
        print(idx,qs)
        cur_prompt = qs

        image_path = os.path.join(args.image_folder, image_file)

        prompt = qs

        times=0

        output_text = generate_output(
            img_path=image_path,
            prompt=prompt)


        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": output_text,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()
eval_llava_benchmark(args)
