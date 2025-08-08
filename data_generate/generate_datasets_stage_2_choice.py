import os
import json
import random

# --- 1. 配置您的路径 ---
# (请根据您的实际情况修改这些路径)

# 输入: 包含所有病例的总纲JSON文件路径
MASTER_JSON_PATH = './test.json'  # <--- 修改这里
'''
这个json文件的结构示例:
[
  {
    "idx": 0,
    "status": "stable_stage",
    "patient_id": "001",
    "images": {
      "clinical": [
        "stable_stage/0/clinical/0.jpg"
      ],
      "wood_lamp": [
        "stable_stage/0/wood_lamp/0.jpg"
      ]
    }
  },
  ...]
'''
# 输出: 所有生成的数据集将被保存在这个文件夹中
OUTPUT_DIR = ''

# --- (代码实现部分，通常无需修改) ---

# 定义四种图片类型及其属性
IMAGE_TYPES = [
    {
        'name': "original clinical image",
        'prefix': 'original',
        'key': 'clinical',
        'initial': 'OC'
    },
    {
        'name': "original Wood's lamp image",
        'prefix': 'original',
        'key': 'wood_lamp',
        'initial': 'OW'
    },
    {
        'name': "edge-enhanced clinical image",
        'prefix': 'edge_enhanced',
        'key': 'clinical',
        'initial': 'EC'
    },
    {
        'name': "edge-enhanced Wood's lamp image",
        'prefix': 'edge_enhanced',
        'key': 'wood_lamp',
        'initial': 'EW'
    }
]

def generate_mcq_prompt(image_permutation, option_a_text, option_b_text):
    """Dynamically generates the multiple-choice question prompt."""
    
    if len(image_permutation) == 1:
        img_desc = f"the {image_permutation[0]['name']} <image>"
    else:
        name_list = [f"the {img['name']} <image>" for img in image_permutation]
        img_desc = ", ".join(name_list[:-1]) + f", and {name_list[-1]}"
    
    question = (
        f"Based on the input of {img_desc}, analyze and determine the vitiligo "
        f"progression phase:\n"
        f"A. {option_a_text}\n"
        f"B. {option_b_text}\n"
        f"Please answer option A or B directly."
    )
    return question


def create_multiple_choice_dataset(master_data, permutation, output_path, doot_dir):
    """Creates a complete multiple-choice dataset for a specific permutation."""
    
    labeled_samples = []
    for case in master_data:
        for i in range(len(case['images']['clinical'])):
            image_paths = []
            
            # 检查这个案例是否包含所有需要的图片类型
            missing_image = False
            for img_type in permutation:
                if not case['images'][img_type['key']]:
                    missing_image = True
                    break
            
            # 如果有任何一种图片缺失，就跳过这个案例
            if missing_image:
                continue
            
            for img_type in permutation:
                relative_path = case['images'][img_type['key']][i]
                base_name = os.path.basename(relative_path) 
                #将base_name和img_type['prefix']连接起来
                final_path = img_type['prefix'] + "_" + base_name
                final_path = os.path.join(doot_dir, final_path)
                image_paths.append(final_path)
            
            labeled_samples.append({
                'label': case['status'],
                'images': image_paths
            })

    new_dataset = []
    total_samples = len(labeled_samples)
    if total_samples == 0:
        return

    half = total_samples // 2
    
    random.shuffle(labeled_samples)
    a_is_stable_flags = [True] * half + [False] * (total_samples - half)
    random.shuffle(a_is_stable_flags)

    for idx, (sample, a_is_stable) in enumerate(zip(labeled_samples, a_is_stable_flags)):
        label = sample['label']
        img_paths = sample['images']

        option_A_text = "stable stage vitiligo" if a_is_stable else "active stage vitiligo"
        option_B_text = "active stage vitiligo" if a_is_stable else "stable stage vitiligo"

        is_stable_case = (label == "stable_stage")
        answer = "A" if is_stable_case == a_is_stable else "B"

        question = generate_mcq_prompt(permutation, option_A_text, option_B_text)

        new_entry = {
            "id": idx,
            "conversations": [
                {"from": "human", "value": question},
                # 不需要answer的话，注释掉下面这行
                # {"from": "gpt", "value": answer}
            ],
            "image": img_paths
        }
        new_dataset.append(new_entry)

    random.shuffle(new_dataset)
    #假如没有图片，则不生成数据集        
    if len(new_dataset) == 0:
        return
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_dataset, f, indent=2, ensure_ascii=False)


def two_choice(master_json_path, output_dir, doot_dir):
    """主函数，根据指定的6种组合生成选择题数据集。"""
    
    if not os.path.exists(master_json_path):
        print(f"Error: Master JSON file not found at '{master_json_path}'")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"All specified Multiple-Choice datasets will be saved in: '{output_dir}'")

    with open(master_json_path, 'r', encoding='utf-8') as f:
        master_data = json.load(f)

    # --- 代码修改核心 ---

    # 1. 创建一个从首字母缩写到完整图片信息字典的映射
    image_type_map = {img_type['initial']: img_type for img_type in IMAGE_TYPES}

    # 2. 直接定义我们想要生成的6种组合
    target_permutations = [
        ['OC'],
        ['OW'],
        ['OC', 'EC'],
        ['OW', 'EW'],
        ['OC', 'OW'],
        ['OC', 'EC', 'OW', 'EW']
    ]

    print("Starting specified multiple-choice dataset generation...")
    dataset_counter = 0

    # 3. 遍历我们直接定义好的目标列表
    for initials_list in target_permutations:
        try:
            # 根据首字母列表，构建生成数据集所需的完整信息列表(perm)
            perm = [image_type_map[initial] for initial in initials_list]
        except KeyError as e:
            print(f"Warning: Skipping {initials_list} due to unrecognized initial: {e}.")
            continue

        # 4. 生成文件名并创建数据集
        k = len(perm)
        initials_str = "_".join(initials_list)
        # 注意：这里使用了新的文件名格式 "test_choice_..."
        output_filename = f'test_choice_{k}img_{initials_str}.json'
        output_filepath = os.path.join(output_dir, output_filename)
        
        # 调用选择题数据集的生成函数
        create_multiple_choice_dataset(master_data, perm, output_filepath, doot_dir)
        
        print(f"  - Generated: {output_filename}")
        dataset_counter += 1

    print(f"\nTask Complete! Generated a total of {dataset_counter} specified multiple-choice datasets.")

