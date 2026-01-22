import torch
import numpy as np
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')
from tqdm import tqdm
import time


def evaluate_model(model, test_loader, device, threshold=0.5):
    """
    评估模型在测试集上的性能
    返回Micro F1分数
    """
    model.eval()

    all_preds = []
    all_labels = []
    inference_times = []

    print("开始评估模型...")

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='评估中', unit='batch'):
            # 移动数据到设备
            images = images.to(device)

            # 记录推理时间
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()

            # 计算推理时间（秒）
            inference_time = (end_time - start_time) / images.shape[0]  # 每个样本的平均时间
            inference_times.append(inference_time)

            # 将输出转换为二值预测
            preds = (outputs > threshold).float()

            # 收集预测和标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 计算Micro F1分数（主评估指标）
    micro_f1 = f1_score(all_labels, all_preds, average='micro')

    # 计算其他评估指标
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    # 计算每个类别的F1分数
    class_f1 = f1_score(all_labels, all_preds, average=None)

    # 计算准确率
    accuracy = (all_preds == all_labels).mean()

    # 计算精确率、召回率、F1分数的详细报告
    detailed_report = classification_report(
        all_labels,
        all_preds,
        target_names=['No Defect', 'Chipped Edge', 'Scratch', 'Stain'],
        output_dict=True
    )

    # 计算推理统计
    avg_inference_time = np.mean(inference_times) * 1000  # 转换为毫秒
    fps = 1.0 / np.mean(inference_times)  # 每秒处理的帧数

    # 打印评估结果
    print("\n" + "=" * 60)
    print("模型评估结果")
    print("=" * 60)
    print(f"Micro F1分数: {micro_f1:.4f}")
    print(f"Macro F1分数: {macro_f1:.4f}")
    print(f"整体准确率: {accuracy:.4f}")
    print(f"平均推理时间: {avg_inference_time:.2f} ms/样本")
    print(f"处理速度: {fps:.2f} FPS")
    print("-" * 60)

    # 打印每个类别的F1分数
    print("每个类别的F1分数:")
    for i, (class_name, f1) in enumerate(zip(['No Defect', 'Chipped Edge', 'Scratch', 'Stain'], class_f1)):
        print(f"  {class_name}: {f1:.4f}")

    # 打印详细分类报告
    print("\n详细分类报告:")
    print(f"  No Defect - 精确率: {detailed_report['No Defect']['precision']:.4f}, "
          f"召回率: {detailed_report['No Defect']['recall']:.4f}, "
          f"F1: {detailed_report['No Defect']['f1-score']:.4f}")
    print(f"  Chipped Edge - 精确率: {detailed_report['Chipped Edge']['precision']:.4f}, "
          f"召回率: {detailed_report['Chipped Edge']['recall']:.4f}, "
          f"F1: {detailed_report['Chipped Edge']['f1-score']:.4f}")
    print(f"  Scratch - 精确率: {detailed_report['Scratch']['precision']:.4f}, "
          f"召回率: {detailed_report['Scratch']['recall']:.4f}, "
          f"F1: {detailed_report['Scratch']['f1-score']:.4f}")
    print(f"  Stain - 精确率: {detailed_report['Stain']['precision']:.4f}, "
          f"召回率: {detailed_report['Stain']['recall']:.4f}, "
          f"F1: {detailed_report['Stain']['f1-score']:.4f}")
    print("=" * 60)

    # 可视化混淆矩阵（可选）
    plot_confusion_matrix(all_labels, all_preds)

    # 返回Micro F1分数（TA评估脚本需要）
    return micro_f1


def plot_confusion_matrix(labels, preds, save_path='confusion_matrix.png'):
    """
    绘制多标签分类的混淆矩阵
    由于是多标签，我们绘制每个类别的预测情况
    """
    from sklearn.metrics import multilabel_confusion_matrix

    # 计算每个类别的混淆矩阵
    cm = multilabel_confusion_matrix(labels, preds)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('多标签分类混淆矩阵（每个类别）', fontsize=16)

    class_names = ['No Defect', 'Chipped Edge', 'Scratch', 'Stain']

    for idx, ax in enumerate(axes.flatten()):
        if idx < 4:
            # 提取TN, FP, FN, TP
            tn, fp, fn, tp = cm[idx].ravel()

            # 创建热力图数据
            heatmap_data = np.array([[tn, fp], [fn, tp]])

            # 绘制热力图
            im = ax.imshow(heatmap_data, cmap='Blues', interpolation='nearest')

            # 添加数值标签
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{heatmap_data[i, j]}",
                            ha="center", va="center", color="black", fontsize=12)

            # 设置坐标轴标签
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Pred 0', 'Pred 1'])
            ax.set_yticklabels(['True 0', 'True 1'])

            ax.set_title(f'{class_names[idx]} (F1: {f1_score(labels[:, idx], preds[:, idx]):.3f})')

            # 添加颜色条
            plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存到 {save_path}")
    plt.close()


def analyze_predictions(model, test_loader, device, num_samples=5):
    """
    分析模型预测结果，可视化一些样本
    """
    model.eval()

    print(f"\n分析 {num_samples} 个样本的预测结果...")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            if batch_idx >= num_samples:
                break

            images = images.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()

            # 转换回CPU
            image_np = images[0].cpu().numpy().transpose(1, 2, 0)
            label_np = labels[0].cpu().numpy()
            pred_np = preds[0].cpu().numpy()

            # 反归一化图像
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = std * image_np + mean
            image_np = np.clip(image_np, 0, 1)

            # 打印结果
            print(f"\n样本 {batch_idx + 1}:")
            print(f"  真实标签: {label_np}")
            print(f"  预测标签: {pred_np}")
            print(f"  预测概率: {outputs[0].cpu().numpy()}")

            # 检查哪些预测正确
            correct = (pred_np == label_np)
            print(f"  正确预测: {correct}")

            if not all(correct):
                print(f"  错误预测的类别: {np.where(correct == 0)[0]}")