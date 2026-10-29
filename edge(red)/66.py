import os
import cv2
import numpy as np
from tqdm import tqdm

def optimize_for_metal_defects(img):
    """专为金属表面缺陷检测优化的预处理流程 - 增强对比度版"""
    # 转换为灰度图（确保是单通道）
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # 1. 多阶段对比度增强
    # 第一阶段：自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(12, 12))  # 增强参数
    enhanced = clahe.apply(gray)
    
    # 第二阶段：全局对比度拉伸
    min_val = np.min(enhanced)
    max_val = np.max(enhanced)
    stretched = cv2.convertScaleAbs(enhanced, alpha=255.0/(max_val-min_val), beta=-min_val*255.0/(max_val-min_val))
    
    # 2. 缺陷特异性增强 - 改进的锐化
    blurred = cv2.GaussianBlur(stretched, (3, 3), 0)  # 更小的核尺寸
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    sharpened = cv2.addWeighted(stretched, 2.0, blurred, -1.0, 0)  # 更强的锐化
    
    # 3. 多阈值缺陷提取 - 对比度增强版
    # 针对暗色缺陷 - 使用局部对比度增强
    black_enhanced = cv2.subtract(sharpened, cv2.GaussianBlur(sharpened, (25, 25), 0))
    _, black_defects = cv2.threshold(black_enhanced, 30, 255, cv2.THRESH_BINARY_INV)
    
    # 针对亮色缺陷 - 使用局部对比度增强
    white_enhanced = cv2.subtract(cv2.GaussianBlur(sharpened, (25, 25), 0), sharpened)
    _, white_defects = cv2.threshold(white_enhanced, 30, 255, cv2.THRESH_BINARY)
    
    # 自适应阈值增强
    adaptive_thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 15, 5)  # 调整参数
    
    # 组合多种阈值结果
    defects = cv2.bitwise_or(black_defects, white_defects)
    defects = cv2.bitwise_or(defects, adaptive_thresh)
    
    # 4. 改进的形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    defects = cv2.morphologyEx(defects, cv2.MORPH_OPEN, kernel, iterations=1)
    defects = cv2.morphologyEx(defects, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 5. 边缘增强 - 使用对比度敏感参数
    edges = cv2.Canny(sharpened, 70, 200)  # 调整阈值以增强边缘对比度
    defects = cv2.bitwise_or(defects, edges)
    
    # 6. 结果合成 - 增强对比度的混合方式
    # 先对缺陷图进行对比度增强
    defects_enhanced = cv2.convertScaleAbs(defects, alpha=1.5, beta=0)
    result = cv2.addWeighted(sharpened, 0.7, defects_enhanced, 0.5, 0)
    
    # 7. 最终对比度调整 - 更强烈的对比度增强
    # 使用gamma校正增强对比度
    gamma = 0.9
    result = np.power(result/255.0, gamma) * 255.0
    result = result.astype(np.uint8)
    
    # 额外的局部对比度增强
    lab = cv2.cvtColor(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    final_result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    final_result = cv2.cvtColor(final_result, cv2.COLOR_BGR2GRAY)
    
    return final_result

def process_image(input_path, output_path, show_comparison=False):
    """处理单个图像并可选显示对比"""
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图像: {input_path}")
        return
    
    processed = optimize_for_metal_defects(img)
    
    if show_comparison:
        # 增强对比度的比较显示
        resized_original = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
        resized_processed = cv2.resize(processed, (0,0), fx=0.5, fy=0.5)
        comparison = np.hstack((resized_original, resized_processed))
        
        # 添加文字说明
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, 'Original', (10,30), font, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(comparison, 'Processed', (resized_original.shape[1]+10,30), font, 1, (255,255,255), 2, cv2.LINE_AA)
        
        cv2.imshow('Original vs Enhanced Contrast', comparison)
        cv2.waitKey(1000)  # 显示1秒
        cv2.destroyAllWindows()
    
    cv2.imwrite(output_path, processed)

def batch_process_images(input_folder, output_folder, show_comparison=False):
    """批量处理文件夹中的所有图像"""
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有图像文件
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(supported_formats)]
    
    if not image_files:
        print(f"在文件夹 {input_folder} 中没有找到支持的图像文件")
        return
    
    print(f"开始处理 {len(image_files)} 张金属表面图像...")
    print("检测类型: 划痕/黑点/褶皱等表面缺陷 (增强对比度模式)")
    
    # 使用进度条显示处理进度
    for filename in tqdm(image_files, desc="处理进度"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        process_image(input_path, output_path, show_comparison)
    
    print(f"所有图像处理完成! 结果已保存到: {output_folder}")

if __name__ == "__main__":
    # 配置路径
    base_dir = r"D:\app\pythonMODEDEMO"  # 修改为你的实际路径
    input_dir = os.path.join(base_dir, "input_images")
    output_dir = os.path.join(base_dir, "output_images")
    
    # 检查输入目录
    if not os.path.exists(input_dir):
        print(f"错误：输入目录不存在 {input_dir}")
        print("请创建 input_images 文件夹并放入待检测的金属表面图片")
        input("按回车键退出...")
        exit()
    
    # 执行批量处理(show_comparison=True可显示处理前后对比)
    batch_process_images(input_dir, output_dir, show_comparison=False)
    
    # 防止窗口闪退
    input("按回车键退出程序...")
