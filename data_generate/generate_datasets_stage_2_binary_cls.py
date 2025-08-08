import os
import json
import argparse

# --- 1. 配置您的路径 ---
# (请根据您的实际情况修改这些路径)

# 输入: 包含所有病例的总纲JSON文件路径
MASTER_JSON_PATH = './train.json'  # <--- 修改这里
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

def generate_prompt(image_permutation):
    """根据图片的排列顺序，动态生成提问文本(prompt)。"""
    if len(image_permutation) == 1:
        img_desc = f"the {image_permutation[0]['name']} <image>"
    else:
        name_list = [f"the {img['name']} <image>" for img in image_permutation]
        img_desc = ", ".join(name_list[:-1]) + f", and {name_list[-1]}"
    
    return (f"Based on the input of {img_desc}, analyze and determine the vitiligo "
            f"progression phase: stable stage vitiligo or active stage vitiligo?")

def create_dataset_for_permutation(master_data, permutation, output_path, doot_dir):
    """为一种特定的图片排列组合生成一个完整的数据集文件。"""
    new_dataset = []
    prompt_template = generate_prompt(permutation)
    
    conversation_id = 0
    for case in master_data:
        num_images_in_case = len(case['images']['clinical'])
        for i in range(num_images_in_case):
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

            gpt_response = f"{case['status'].replace('_', ' ')} vitiligo"

            new_entry = {
                "id": conversation_id,
                "conversations": [
                    {"from": "human", "value": prompt_template},
                    # 不需要answer的话，注释掉下面这行
                    # {"from": "gpt", "value": gpt_response}

                ],
                "image": image_paths
            }
            new_dataset.append(new_entry)
            conversation_id += 1
    #假如数据集为空，则不生成数据集
    if len(new_dataset) == 0:
        return
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_dataset, f, indent=2, ensure_ascii=False)


def two_binary_cls(master_json_path, output_dir, doot_dir):
    """主函数，执行所有操作。"""
    
    if not os.path.exists(master_json_path):
        print(f"错误: 总纲文件未找到 '{master_json_path}'")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"所有生成的数据集将被保存在: '{output_dir}'")

    with open(master_json_path, 'r', encoding='utf-8') as f:
        master_data = json.load(f)

    # --- 代码修改核心 ---

    # 1. 创建一个从首字母缩写到完整图片信息字典的映射，方便后续查找
    image_type_map = {img_type['initial']: img_type for img_type in IMAGE_TYPES}

    # 2. 直接定义我们想要生成的6种组合，每个组合都是一个列表，包含大写的首字母
    target_permutations = [
        ['OC'],
        ['OW'],
        ['OC', 'EC'],
        ['OW', 'EW'],
        ['OC', 'OW'],
        ['OC', 'EC', 'OW', 'EW']
    ]

    print("开始根据指定的组合生成数据集...")
    dataset_counter = 0

    # 3. 遍历我们直接定义好的目标列表
    for initials_list in target_permutations:
        # 4. 根据首字母列表，构建生成数据集所需的完整信息列表(perm)
        #    这步是必需的，因为 create_dataset_for_permutation 需要完整的字典信息
        try:
            perm = [image_type_map[initial] for initial in initials_list]
        except KeyError as e:
            print(f"警告: 跳过 {initials_list} 因为其中包含无法识别的首字母: {e}。")
            continue

        # 5. 生成文件名并创建数据集
        k = len(perm)
        filename_initials_part = "_".join(initials_list) # 直接使用定义好的列表
        output_filename = f'train_binary_cls_{k}img_{filename_initials_part}.json'
        output_filepath = os.path.join(output_dir, output_filename)
        
        create_dataset_for_permutation(master_data, perm, output_filepath, doot_dir)
        
        print(f"  - 已生成: {output_filename}")
        dataset_counter += 1

    print(f"\n任务完成！总共生成了 {dataset_counter} 个指定的数据集。")

