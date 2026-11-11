# Physical-coordinate radiance-error dithering / 物理坐标反射误差抖动

> 主程序现已增加原生 .NET 实验模式，见[原生功能说明](../../docs/PHYSICAL_DITHER_NATIVE.md)。本文保留为 Python 参考原型的设计与用法；原生功能不依赖 Python。
一个可运行的 **DVD 实验性图片预处理工具**。它先用两种刻录纹理的有效反射模型预测多个灯位下的亮度，再求每个盘面位置的纹理混合比例，最后在物理坐标上扩散反射亮度误差。输出不透明二值 PNG，可交给现有 DVD `dispersion` 路径。

它尚未集成主程序 UI，不修改现有默认算法。附带参数是明确标记的假设模型，**不是威宝 AZO DVD-R 43523 的实测材料参数**；不能据示例预览声称实盘对比度提高或可见角度扩大。原理、校准方案和后续精化见 [设计说明](../../docs/PHYSICAL_DITHER.md)。

## 运行

依赖 Python 3.10+、NumPy、SciPy、Pillow；不依赖 Windows/WPF。不包含第三方源码或数据。建议使用项目外的虚拟环境安装 `requirements.txt`。

输入必须是已经排好版的**正方形盘面图**，边长对应配置的 `2 × canvas_radius_mm`；不会自动裁切人物。照片先线性化 sRGB，再按 Rec.709 转亮度；透明区在线性空间合成白底。

```bash
python tools/PhysicalDither/physical_dither.py \
  --image square-disc-layout.png \
  --profile tools/PhysicalDither/example-profile.json \
  --output-dir artifacts/physical-dither \
  --cell-um 40
```

输出目录必须为空，以防覆盖之前的对照结果。默认格距80 µm；上例40 µm对应116 mm画布上的2901×2901采样。二者都不是已知最佳尺度。原型限制输出到33–4097像素；当前全盘不能直接选择20 µm，需要后续分块实现。光学模型先在97×97网格计算，再双线性插值；使用很窄的反射瓣时，应增加 `--map-size` 并做收敛对照。

参数：

- `--cell-um`：物理采样间隔，与照片原始像素数无关。实际间隔因端点对齐略有取整，写入报告。
- `--tone-exponent`：独立的线性亮度影调曲线，默认1。改变它不会改变抖动算法。
- `--seed`：非周期阈值微扰及随机对照的种子；相同输入、参数和环境可复现。
- `--map-size`：光学响应图的计算分辨率，默认97。

## 输出

| 文件 | 意义 |
|---|---|
| `binary.png` | 真正送入转换链的0/255、不透明二值图；0对应Q=A5，255对应Q=92 |
| `random-baseline.png` | 同一求解后混合比例的随机抖动对照 |
| `mixture-preview.png` | 连续混合比例的可视化，不能作为最终二值刻录输入 |
| `drawing-mask.png` | 有效绘制环带；独立遮罩，不是binary.png的alpha通道 |
| `view-N-predicted.png` | 假设模型下、经过两格标准差高斯模糊的亮度预览 |
| `view-N-target.png` | 对应灯位下的目标亮度；与predicted使用同一显示尺度 |
| `report.json` | 输入哈希、完整模型、灯位、种子、格距、不可观测/混合比例裁剪比例、连续拟合残差及三种方法的多尺度误差 |

预览是归一化的**线性亮度灰度图**，不预测盘片紫色、彩虹色或相机的自动曝光。不同灯位可有不同显示尺度，不应用PNG亮度比较灯具绝对效率；数值比较用报告中的线性辐亮度。报告中的低误差也不等于实盘验证。

## 接入现有DVD路径

保持二值图原样，不缩放、不旋转、不再次抖动、不再做gamma。`canvas_radius_mm` 必须与DVD求解器的图片采样外半径一致；示例为58 mm。实际起止半径、总扇区和螺旋参数仍须使用自己的盘片几何标定，示例参数不替代该标定。

输出包含绘图区外的黑色像素，以维持**整张图不透明且只含0/255**。这使现有 `RasterImage.HasBinaryGrayscalePalette` 与最近邻采样路径生效。盘片内圈、文件区域和实际输出范围由DVD生成器决定；不要将黑色画布外区误当成透明留白。

本地32扇区管线冒烟命令（仅测试数据流，**不是完整刻录盘镜像**）：

```bash
dotnet run --project src/DiscImageStudio.Cli -c Release -- \
  solve --image artifacts/physical-dither/binary.png \
  --total-sectors 2297888 --inner-radius-mm 24.5 --outer-radius-mm 58 \
  --lba 0 --fill-sectors 32 --algorithm dispersion \
  --iso-output artifacts/physical-dither/smoke.iso \
  --tone-dither 0 --tone-gamma false --tone-brightness 0
```

总扇区数在这里仅用于演示传入几何；不能认为它是43523的已确认可写容量。完整输出还需设置实际范围。CLI快速输出会由刻录机补全派生编码，模拟器本身没有回放全盘ECC/EFMPlus。若启用读取兼容性替换、混入文件或修改码字配方，实际光学响应会变化，需要对应的新模型或标定；主程序正常的可读性规则不应为了追求模拟效果而绕过。

## 测试

```bash
python -m unittest discover -s tools/PhysicalDither -p 'test_*.py' -v
```

覆盖：互易性、半球反射能量上界、非法参数、有限光源积分与光学网格收敛、反相纹理混合、多角度不可兼得、无对比度区域、确定性/平均灰阶/低频噪声、孔洞与纯色端点。

这些测试验证数值模型和算法性质，不证明模型与真实材料吻合。源代码只把两种纹理当作有效光学单元；没有声称计算出染料复折射率、刻录坑形状或独立控制坑点。
