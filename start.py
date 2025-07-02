#!/usr/bin/env python3
"""
白癜风AI诊断系统启动脚本
"""

import os
import sys
import subprocess
import importlib.util

def check_dependencies():
    """检查依赖包是否安装"""
    required_packages = [
        'flask',
        'flask_cors', 
        'torch',
        'torchvision',
        'PIL',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if package == 'PIL':
            package_name = 'Pillow'
        elif package == 'flask_cors':
            package_name = 'flask-cors'
        else:
            package_name = package
            
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n请运行以下命令安装:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def check_model():
    """检查模型文件是否存在"""
    model_path = 'model/best_model.pth'
    if os.path.exists(model_path):
        print(f"✅ 找到模型文件: {model_path}")
        return True
    else:
        print(f"⚠️  模型文件不存在: {model_path}")
        print("   系统将使用模拟数据运行")
        print("   请参考 model/README.md 了解如何添加你的模型")
        return False

def check_directories():
    """检查必要的目录是否存在"""
    required_dirs = ['uploads', 'model', 'change_cup']
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"创建目录: {dir_name}")
            os.makedirs(dir_name, exist_ok=True)
        else:
            print(f"✅ 目录存在: {dir_name}")

def start_server():
    """启动Flask服务器"""
    print("\n" + "="*50)
    print("🚀 启动白癜风AI诊断系统")
    print("="*50)
    
    try:
        # 导入并启动app
        from app import app
        print("\n📍 服务地址:")
        print("   本地访问: http://localhost:8080")
        print("   网络访问: http://0.0.0.0:8080")
        print("\n📊 API接口:")
        print("   预测接口: http://localhost:8080/predict")
        print("   健康检查: http://localhost:8080/health")
        print("\n💡 使用说明:")
        print("   1. 在浏览器中打开上述地址")
        print("   2. 上传临床皮损照片和伍德灯照片")
        print("   3. 点击'生成结果'进行AI分析")
        print("\n按 Ctrl+C 停止服务")
        print("-"*50)
        
        app.run(debug=True, host='0.0.0.0', port=8080)
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请确保 app.py 文件存在且正确")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

def main():
    """主函数"""
    print("白癜风AI诊断系统 - 启动检查")
    print("-"*40)
    
    # 检查目录
    check_directories()
    print()
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    print()
    
    # 检查模型
    check_model()
    
    # 启动服务器
    start_server()

if __name__ == '__main__':
    main() 