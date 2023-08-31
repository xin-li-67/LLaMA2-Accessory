import os
import json
import argparse

import torch
from tqdm import tqdm
import numpy as np
import random
from PIL import Image
import math
import time
import sys
sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0])



from model.meta import MetaModel
from PIL import Image

from util import misc
from fairscale.nn.model_parallel import initialize as fs_init


from util.quant import quantize
from data.alpaca import transform_val, format_prompt
from util.tensor_parallel import load_tensor_parallel_model_list



# SEED bench only for image_test. Please refer to the complete SEED bench (including video detection, etc.) https://github.com/AILab-CVC/SEED-Bench/tree/main

cc3m_dir = "path/to/SEED-Bench-image"

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_args_parser():
    parser = argparse.ArgumentParser('Single-turn (conversation) demo', add_help=False)
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

    parser.add_argument("--max_gen_len", type=int, default=516)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.7)
    
    # SEED settings
    parser.add_argument('--anno_path', type=str, default='path/to/SEED-Bench.json')
    parser.add_argument('--output_dir', type=str, default='yourpath/tosave/results')
    parser.add_argument('--task', type=str, default='image')
    
    return parser

args = get_args_parser().parse_args()


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
    
print("Model = %s" % str(model))
model.bfloat16().cuda()


def process_prompt(question):
    return f"""Question: {question}\nAnswer:"""

def is_integer_string(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def filter_questions(data, task='image'):
    if task == "image":
        return [q for q in data if 1 <= q["question_type_id"] <= 9]
    else:
        raise ValueError(f"Invalid task: {task}")
    
    
    
@ torch.inference_mode()
def generate_output(img_path,prompt):

    print("image path:",img_path)
    # plt.imshow(img_path)
    if img_path is not None:
        image = Image.open(img_path).convert('RGB')
        
        image = transform_val(image).unsqueeze(0)
    else:
        image = None

    _prompt = prompt

    if image is not None:
        image = image.cuda()
    
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        results = model.generate([_prompt], image, max_gen_len=12, temperature=0.1, top_p=0.7)
    text_output = results[0].strip()
    
    return text_output


def eval_seed(qa_anno, output_dir):
    start = time.time()
    total_qa_num = len(qa_anno)
    answer_list = []
    output_f = open(os.path.join(output_dir, "results.json"), "a")
    step = 0
    for qa_item in tqdm(qa_anno):

        data_info = {
            'question': qa_item['question'],
            'choices': [qa_item['choice_a'], qa_item['choice_b'], qa_item['choice_c'], qa_item['choice_d']],
            'data_type': qa_item['data_type'],
        }

        if qa_item['data_type'] == 'image':
            data_path = os.path.join(cc3m_dir, qa_item['data_id'])
        else:
            raise ValueError("The data type is not valid.")
        data_info['data_path'] = data_path


        #整理出来
        question=qa_item['question']
        choices=[qa_item['choice_a'], qa_item['choice_b'], qa_item['choice_c'], qa_item['choice_d']]
        data_type=qa_item['data_type']
        data_path = data_path
        
        
        prompt="Question:"+ question+ "\nthe choices are:" + "A." +qa_item['choice_a'] + "B." +qa_item['choice_b'] + "C." +qa_item['choice_c'] + "D." +qa_item['choice_d']+"\n please find the best choice for the question.\n The answer is:"
        

        output_text = generate_output(
                    img_path=data_path,
                    prompt=prompt)

        pred_id=''
        for i in range(len(output_text)):
            if output_text[i]=='A'  or output_text[i]=='a':
                pred_id='A'
                break
            elif output_text[i]=='B' or output_text[i]=='b':
                pred_id='B'
                break
            elif output_text[i]=='C' or output_text[i]=='c':
                pred_id='C'
                break
            elif output_text[i]=='D' or output_text[i]=='d':
                pred_id='D'
                break

        
        gt = qa_item['answer']
        answer_record = {
            'question_id': qa_item['question_id'],
            'prediction': pred_id,
            'gt':gt,
            'q_type_id':qa_item['question_type_id']
        }
        answer_list.append(answer_record)
        # output prediction record for each question
        output_f.write(json.dumps(answer_record) + "\n")
        step += 1
        
    end = time.time()
    
    
    print("evaluation finished! Calculating accuracy...")
    type_counts = {}
    correct_counts = {}

    for item in answer_list:
        pred, gt, data_type = item['prediction'], item['gt'], item['q_type_id']

        type_counts[data_type] = type_counts.get(data_type, 0) + 1
        if pred == gt:
            correct_counts[data_type] = correct_counts.get(data_type, 0) + 1

    print("Accuracy for each data type:")
    total_count = 0
    total_correct = 0
    for data_type in type_counts.keys():
        accuracy = correct_counts[data_type] / type_counts[data_type] * 100
        print(f"Data type {data_type}: {accuracy:.2f}%")

        total_count += type_counts[data_type]
        total_correct += correct_counts[data_type]

    total_accuracy = total_correct / total_count * 100
    print(f"Total accuracy: {total_accuracy:.2f}%")
    print("time cost:%.2f s"%(end-start))
    
    
    
qa_anno = json.load(open(args.anno_path, 'rb'))
if 'questions' in qa_anno.keys():
    qa_anno = qa_anno['questions']
qa_anno = filter_questions(qa_anno, args.task)

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

print(f'evaluating..')


eval_seed(qa_anno, args.output_dir)
