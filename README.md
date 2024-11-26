
这不刚换了一个笔记本电脑，Thinkpad T14P，带有Intel ARC GPU，今天我们来尝试用这个GPU来推理ONNX模型。


## 环境安装


查阅了相关文档，最好使用py310环境，其他版本可能存在兼容性问题，然后按照以下命令安装：



```
Copy# conda 环境
conda activate py310
# libuv
conda install libuv

conda install -c conda-forge libjpeg-turbo libpng

# torch
python -m pip install torch==2.3.1.post0+cxx11.abi torchvision==0.18.1.post0+cxx11.abi torchaudio==2.3.1.post0+cxx11.abi intel-extension-for-pytorch==2.3.110.post0+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/lnl/cn/

# onnxruntime
pip install onnxruntime-openvino openvino


```

## 测试



```
Copypython -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__); [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())];"

```

2\.3\.1\.post0\+cxx11\.abi
2\.3\.110\.post0\+xpu
\[0]: \_XpuDeviceProperties(name\='Intel(R) Arc(TM) Graphics', platform\_name\='Intel(R) Level\-Zero', type\='gpu', driver\_version\='1\.3\.31441', total\_memory\=16837MB, max\_compute\_units\=112, gpu\_eu\_count\=112, gpu\_subslice\_count\=14, max\_work\_group\_size\=1024, max\_num\_sub\_groups\=128, sub\_group\_sizes\=\[8 16 32], has\_fp16\=1, has\_fp64\=1, has\_atomic64\=1\)


## 加载detr模型


我们现在测试一下，使用DETR模型（[https://github.com/facebookresearch/detr），我们先将训练好的模型转成onnx格式，然后使用onnxruntime进行推理。](https://github.com)


### 先detr转onnx



```
Copydef main(args):

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, _, _ = build_model(args)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    
    dynamic_axes={
        "inputs": {0: "batch_size", 2: "height", 3: "width"},  # 改成 "inputs"，以匹配 input_names
        "pred_logits": {0: "batch_size"},                      # 改成 "pred_logits" 和 "pred_boxes"
        "pred_boxes": {0: "batch_size"}
    }

    torch.onnx.export(
        model, 
        torch.randn(1, 3, 800, 1200).to(device),   # 示例输入大小
        "model.onnx",
        do_constant_folding=True,
        opset_version=12,
        dynamic_axes=dynamic_axes,   
        input_names=["inputs"],
        output_names=["pred_logits", "pred_boxes"]
    )

```

注意`dynamic_axes` 设置支持动态大小图片输入。


### onnxruntime 推理


先转换为FP16模型，使用OpenVINOExecutionProvider作为推理后端。



```
Copyfrom onnxruntime_tools import optimizer
from onnxconverter_common import float16

# 输入和输出模型路径
input_model_path = "./model.onnx"
fp16_model_path = "./model_fp16.onnx"

# 加载 ONNX 模型
from onnx import load_model, save_model

if not os.path.exists(fp16_model_path):
    model = load_model(input_model_path)
    # 转换为 FP16
    model_fp16 = float16.convert_float_to_float16(model)
    # 保存为 FP16 格式
    save_model(model_fp16, fp16_model_path)
    print(f"FP16 模型已保存至 {fp16_model_path}")

ort_session = onnxruntime.InferenceSession(fp16_model_path, providers=['OpenVINOExecutionProvider'])


# 公共方法：进行图像预处理和模型推理
def predict_image(image: Image.Image):
    w, h = image.size
    target_sizes = torch.as_tensor([int(h), int(w)]).unsqueeze(0)

    # 预处理图片
    _trans = transform()
    image, _ = _trans(image, target=None)
    
    # 记录推理的开始时间
    start_time = time.time()
    
    # 进行 ONNX 推理 
    ort_inputs = {"inputs": image.unsqueeze(0).numpy().astype(np.float16)}
    outputs = ort_session.run(None, ort_inputs)
    
    # 记录推理的结束时间
    end_time = time.time()
    inference_time = end_time - start_time  # 推理耗时

    # 解析输出
    out_logits = torch.as_tensor(outputs[0])
    out_bbox = torch.as_tensor(outputs[1])
    
    prob = F.softmax(out_logits, -1)
    scores, labels = prob[..., :-1].max(-1)
    
    # 转换坐标
    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]
    
    # 组织推理结果
    results = [{'score': s, 'label': l, 'boxes': b, 'category': categories[l-1]['name']} 
               for s, l, b in zip(scores[0].tolist(), labels[0].tolist(), boxes[0].tolist()) if s > 0.9]
    
    print(f'predict cost {inference_time}')
    
    return results, inference_time

```

这里有个坑, onnxruntime\-openvino 推理需要额外添加动态库, 否则报错`onnxruntime::ProviderLibrary::Get [ONNXRuntimeError] : 1 : FAIL : LoadLibrary failed with error 126 "" when trying to load "onnxruntime\capi\onnxruntime_providers_openvino.dll" when using ['OpenVINOExecutionProvider'] Falling back to ['CPUExecutionProvider'] and retrying.`，这里我使用的是Windows系统，所以需要添加动态库。



```
Copy
import platform

# ref https://github.com/microsoft/onnxruntime-inference-examples/issues/117
if platform.system() == "Windows":
    import onnxruntime.tools.add_openvino_win_libs as utils
    utils.add_openvino_libs_to_path()


```

测试下:



```
CopyINFO:     127.0.0.1:64793 - "POST /predict HTTP/1.1" 200 OK
predict cost 0.3524954319000244

```

0\.35秒，还行，马马虎虎！


## 关注作者

欢迎关注作者微信公众号, 一起交流软件开发：:[slower加速器](https://jisuanqi.org)[![欢迎关注作者微信公众号](https://img2020.cnblogs.com/blog/38465/202101/38465-20210118185238294-1640651808.png)](https://github.com)

