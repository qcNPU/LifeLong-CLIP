import json
from openai import OpenAI
from tqdm import tqdm
import os.path as osp



global_api_key = "sk-PdEqIkKM9Nu6EnkJ101672DaE0954d95A7F64eF62083Af01" # Enter Your API Key!!!
#class description缓存
attributes = None

instructions = [
"I will give you two question-anwser examples,please follow them and give the anwser of question 3,just output the anwser pieces split by | and every piece need be less than 20 words.Do not output any Unicode words and anything else except anwser pieces.\
    Q: Describe what an animal giraffe looks like in a photo,list 6 pieces?\
    A: There are 6 useful visual features for a giraffe in a photo:\
    covered with a spotted coat|has a short, stocky body|has a long neck|owns a small neck to its body|is yellow or brown in color|have a black tufted tail\
    Q: Describe what an equipment laptop looks like in a photo, list 4 pieces?\
    A: There are 4 useful visual features for a laptop in a photo:\
    has a built-in touchpad below the keyboard|has a black screen|attached with charging ports|owns a QWERTY keyboard\
    Q: Describe what a {} {} looks like in a photo, list {} pieces?\
    A: There are {} useful visual features for a {} in a photo:",

"I will give you two question-anwser examples,please follow them and give the anwser of question 3,just output the anwser pieces split by | and every piece need be less than 20 words.Do not output any Unicode words and anything else except anwser pieces.\
    Q: Visually describe a giraffe, a type of animal, list 6 pieces?\
    A: There are 6 useful visual features for a giraffe in a photo:\
    covered with a spotted coat|has a short,stocky body|has a long neck|owns a small neck to its body|is yellow or brown in color|have a black tufted tail\
    Q: Visually describe a laptop, a type of equipment, list 4 pieces?\
    A: There are 4 useful visual features for a laptop in a photo:\
    has a built-in touchpad below the keyboard| has a black screen|attached with charging ports|owns a QWERTY keyboard\
    Q: Visually describe a {}, a type of {}, list {} pieces?\\\
    A: There are {} useful visual features for a {} in a photo:",

"I will give you two question-anwser examples,please follow them and give the anwser of question 3,just output the anwser pieces split by | and every piece need be less than 20 words.Do not output any Unicode words and anything else except anwser pieces.\
    Q: How to distinguish a giraffe which is an animal, list 6 pieces?\
    A: There are 6 useful visual features for a giraffe in a photo:\
    covered with a spotted coat|has a short, stocky body|has a long neck|owns a small neck to its body|is yellow or brown in color|have a black tufted tail\
    Q: How to distinguish a laptop which is an equipment, list 4 pieces?\
    A: There are 4 useful visual features for a laptop in a photo:\
    has a built-in touchpad below the keyboard|has a black screen|attached with charging ports|owns a QWERTY keyboard\
    Q: How to distinguish a {} which is a {}, list {} pieces?\
    A: There are {} useful visual features for a {} in a photo:"]
    #class,type,num,num，class

def get_completion(client, prompt, model="gpt-3.5-turbo", temperature=1):
    """
    调用openai的api接口，给出text提示，返回description
    Args:
        client:
        prompt:
        model:
        temperature:
    Returns:
    """
    messages = [{"role": "system", "content": "You are good at image classification."}, {"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()

def get_All_Descriptions(args):
    if args is None:
        gpt_dir = '/home/qc/project/LifeLong-CLIP/datasets/gpt/gpt_data'
        db_name = 'cifar100'
    else:
        gpt_dir = args.get("gpt_dir")
        db_name = args.get("dataset")
    # 要使用全局变量并赋值，需要使用global关键字
    global attributes
    # 1. 若没有缓存
    if attributes is None:
        path = osp.join(gpt_dir ,'attribute', db_name + '.json')
        # 1.1 若文件存在，则直接读取
        if osp.isfile(path):
            with open(path, 'r') as f:
                attributes = json.load(f)
        # 1.2 若文件不存在，则调用gpt生成description
        else:
            # 1.2.1 ——1.0.0版本之后的openai接口调用代码
            client = OpenAI(
                api_key=global_api_key,
                base_url="https://free.gpt.ge/v1/"
            )
            # 1.2.2 先读取该数据集中所有的class, | 之后是大类，构建gpt提示用
            cls_names_path = osp.join(gpt_dir , 'classType' , db_name + '.txt')
            with open(cls_names_path, 'r') as f:
                lines = [line.strip().split("|") for line in f]

            attributes = {}
            cls_descrip_path = osp.join(gpt_dir, 'attribute', db_name + '.json')
            # 1.2.3 对于所有class，逐个生成description
            for classname,ctype in tqdm(lines):
                prompts = [instruction.format(classname,ctype, 5,5,classname) for instruction
                           in instructions]
                responses = [get_completion(client, prompt) for prompt in prompts]
                # 放入缓存中，classname为key
                attributes[classname] = responses
                # 1.2.4 将缓存内容写入json文件——indent，根据数据格式缩进显示，读起来更加清晰
                with open(cls_descrip_path, 'w') as f:
                    json.dump(attributes, f, indent=4)

    return attributes

def get_Classes_Attributes(args, classnames):
    descriptions = get_All_Descriptions(args)
    # 2. 从缓存中读取对应classes的descriptions
    reuslt=[descriptions[i] for i in classnames]
    return reuslt


if __name__ == "__main__":

    get_All_Descriptions(None)
    # get_Classes_Descriptions(args,["face", "leopard", "motorbike"])