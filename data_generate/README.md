# Vitiligo Dataset Generation 

包含两个 Python 脚本，用于基于病例总纲 JSON 文件生成不同任务格式的白癜风图像数据集，适用于多模态训练和评估。

## 文件说明

### 1. `generate_datasets_stage_2_choice.py`
- **功能**：从病例总纲文件生成 **二选一选择题** 格式的数据集（Multiple-Choice）。
- **输入**：病例总纲 JSON 文件（示例见下方）。
- **输出**：多种图像组合形式的选择题 JSON 文件。
- **输出样例结构**：
  ```json
  {
    "id": 0,
    "conversations": [
      {"from": "human", "value": "Based on the input of ... A. stable stage vitiligo\nB. active stage vitiligo"},
      {"from": "gpt", "value": "A"} # 只测试的话可以在代码里注释掉
    ],
    "image": [
      "original/stable_stage/0/clinical/0.jpg",
      "edge_enhanced/stable_stage/0/clinical/0.jpg"
    ]
  }

### 2. `generate_datasets_stage_2_binary_cls.py`

- **功能**：从病例总纲文件生成 **二分类问答** 格式的数据集（Binary Classification）。

- **输入**：病例总纲 JSON 文件。

- **输出**：多种图像组合形式的二分类 JSON 文件。

- **输出样例结构**：

  ```json
  {
    "id": 0,
    "conversations": [
      {"from": "human", "value": "Based on the input of ... stable stage vitiligo or active stage vitiligo?"},
      {"from": "gpt", "value": "stable stage vitiligo"}# 只测试的话可以在代码里注释掉
    ],
    "image": [
      "original/stable_stage/0/clinical/0.jpg"
    ]
  }
  ```

## 病例总纲 JSON 格式

两个脚本共用的输入文件格式示例：

```json
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
  }
]
```

字段说明：

- `status`：病情阶段（`stable_stage` 或 `active_stage`）。
- `images.clinical`：临床照片路径列表。
- `images.wood_lamp`：伍德灯照片路径列表。

## 图像类型编码

| 编码 | 含义                            |
| ---- | ------------------------------- |
| `OC` | original clinical image         |
| `OW` | original Wood's lamp image      |
| `EC` | edge-enhanced clinical image    |
| `EW` | edge-enhanced Wood's lamp image |

## 固定组合列表

两个脚本都会生成以下 6 种图像组合：

1. `['OC']`
2. `['OW']`
3. `['OC', 'EC']`
4. `['OW', 'EW']`
5. `['OC', 'OW']`
6. `['OC', 'EC', 'OW', 'EW']`

## 使用方法

1. **准备输入文件**

   - 修改脚本顶部的 `MASTER_JSON_PATH` 为你的病例总纲文件路径。
   - 修改 `OUTPUT_DIR` 为输出目录。

2. **输出文件命名规则**
   - 选择题：`test_choice_{图片数量}img_{组合编码}.json`
   - 二分类：`train_binary_cls_{图片数量}img_{组合编码}.json`
   
